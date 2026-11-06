"""
Trains the conditional DDPM on CelebA (64x64, attributes Male / Smiling / Young).

Usage (from the repository root):
    python -m diffusion.train --data-root /path/to/data

--data-root is the folder that contains celeba/ in the torchvision layout
(img_align_celeba/, list_attr_celeba.txt, list_eval_partition.txt, ...).
Training resumes automatically from the latest periodic checkpoint in
<output-dir>/checkpoints.
"""

import argparse
import glob
import itertools
import os

import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless / HPC runs
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import v2
from torchvision.utils import make_grid

from diffusion.model import DIFFUSION_STEPS, IMAGE_SIZE, Network, NoiseSchedule, ddpm_sample

ATTR_INDICES = [20, 31, 39]  # Male, Smiling, Young in the CelebA attribute list


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=os.path.expanduser("~/CELEBA"))
    p.add_argument("--output-dir", default="outputs/diffusion")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--cond-dropout", type=float, default=0.2, help="probability of dropping the condition (CFG)")
    p.add_argument("--guidance-scale", type=float, default=2.0, help="guidance used for the preview grids")
    p.add_argument("--checkpoint-freq", type=int, default=10)
    p.add_argument("--preview-freq", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=4)
    return p.parse_args()


def get_loader(data_root, batch_size, num_workers):
    transform = v2.Compose([
        v2.ToImage(),
        v2.CenterCrop(178),
        v2.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # range [-1, 1]
    ])
    train_set = CelebA(root=data_root, split="train", transform=transform, download=False)
    print(f"[INFO] Train samples: {len(train_set)}")
    return DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      drop_last=True, pin_memory=True)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    sample_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    loader = get_loader(args.data_root, args.batch_size, args.num_workers)
    schedule = NoiseSchedule(DIFFUSION_STEPS, device=device)
    model = Network().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    state = {"epoch": 1, "best_train_loss": float("inf"), "train_loss_history": []}

    # --- Resume from the latest periodic checkpoint ---
    checkpoints = sorted(glob.glob(os.path.join(ckpt_dir, "model_epoch_*.pth")),
                         key=lambda p: int(p.split("_")[-1].split(".")[0]))
    if checkpoints:
        print(f"[INFO] Resuming from checkpoint: {checkpoints[-1]}")
        ckpt = torch.load(checkpoints[-1], map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        for group in optimizer.param_groups:
            group["lr"] = args.lr
        state["epoch"] = ckpt["epoch"] + 1
        state["best_train_loss"] = ckpt.get("best_train_loss", float("inf"))
        state["train_loss_history"] = ckpt.get("train_loss_history", [])
    else:
        print("[INFO] No checkpoint found. Starting from scratch.")

    def save_checkpoint(epoch, is_best, periodic):
        data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_train_loss": state["best_train_loss"],
            "train_loss_history": state["train_loss_history"],
        }
        if periodic:
            torch.save(data, os.path.join(ckpt_dir, f"model_epoch_{epoch}.pth"))
        if is_best:
            torch.save(data, os.path.join(ckpt_dir, "best_model.pth"))

    def preview(epoch):
        """Saves a grid with 4 samples for each of the 8 attribute combinations."""
        model.eval()
        combos = torch.tensor(list(itertools.product([-1.0, 1.0], repeat=3)), device=device)
        cond = combos.repeat_interleave(4, dim=0)
        images = ddpm_sample(model, schedule, cond, guidance_scale=args.guidance_scale)
        grid = make_grid(images.cpu(), nrow=4)
        plt.figure(figsize=(6, 12))
        plt.imshow(grid.permute(1, 2, 0))
        plt.axis("off")
        plt.title(f"Epoch {epoch} (guidance {args.guidance_scale})")
        plt.savefig(os.path.join(sample_dir, f"epoch_{epoch}_grid.png"), bbox_inches="tight")
        plt.close()
        model.train()

    def run_epoch():
        model.train()
        running_loss = 0.0
        for x, attrs in loader:
            x = x.to(device)
            # Attributes from {0, 1} to {-1, +1}
            cond = attrs[:, ATTR_INDICES].float().to(device) * 2 - 1.0
            # Condition dropout: the zero vector teaches the unconditional model used by CFG
            mask = (torch.rand(x.shape[0], 1, device=device) > args.cond_dropout).float()
            cond = cond * mask

            t = torch.randint(0, DIFFUSION_STEPS, (x.shape[0],), device=device)
            eps = torch.randn_like(x)
            sqrt_alpha = schedule.sqrt_alpha[t].view(-1, 1, 1, 1)
            sqrt_1_alpha = schedule.sqrt_1_alpha[t].view(-1, 1, 1, 1)
            zt = sqrt_alpha * x + sqrt_1_alpha * eps

            loss = loss_function(model(zt, t, cond), eps)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
        return running_loss / len(loader)

    print(f"[INFO] Starting training. Epochs: {args.epochs}, LR: {args.lr}")
    for epoch in range(state["epoch"], args.epochs + 1):
        train_loss = run_epoch()
        state["train_loss_history"].append(train_loss)
        is_best = train_loss < state["best_train_loss"]
        if is_best:
            state["best_train_loss"] = train_loss
        print(f"[Epoch {epoch}] train loss: {train_loss:.5f}{'  (best)' if is_best else ''}")

        save_checkpoint(epoch, is_best, periodic=epoch % args.checkpoint_freq == 0)
        if epoch % args.preview_freq == 0:
            preview(epoch)

        plt.figure()
        plt.plot(state["train_loss_history"])
        plt.title("Training loss")
        plt.yscale("log")
        plt.grid(True, which="both")
        plt.savefig(os.path.join(args.output_dir, "loss_curve.png"))
        plt.close()

    print("[INFO] Training completed.")


if __name__ == "__main__":
    main()
