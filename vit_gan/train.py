"""
Trains the conditional ViT-GAN on CelebA (64x64, attributes Male / Smiling / Young).

Usage (from the repository root):
    python -m vit_gan.train --data-root /path/to/data

--data-root is the folder that contains celeba/ in the torchvision layout.
A checkpoint {netG, netD, optG, optD, epoch} is saved at the end of every
epoch and training resumes from the latest one.

Generator loss = adversarial BCE + lambda_div * diversity margin loss, where
the diversity term penalises pairs of images generated from the same
attributes and different noise that are closer than `margin` (mean L1),
to counter mode collapse.
"""

import argparse
import glob
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from vit_gan.discriminator import Discriminator
from vit_gan.generator import IMAGE_SIZE, ViTGenerator

ATTR_INDICES = [20, 31, 39]  # Male, Smiling, Young


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=os.path.expanduser("~/CELEBA"))
    p.add_argument("--output-dir", default="outputs/vit_gan")
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--lr-g", type=float, default=1e-4)
    p.add_argument("--lr-d", type=float, default=4e-4)
    p.add_argument("--lambda-div", type=float, default=2.0)
    p.add_argument("--margin", type=float, default=0.5)
    p.add_argument("--keep-checkpoints", type=int, default=0, help="keep only the last N checkpoints (0 = all)")
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def get_celeba_loader(data_root, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    dataset = datasets.CelebA(root=data_root, split="train", target_type="attr", transform=transform, download=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = os.path.join(args.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)

    netG = ViTGenerator(latent_dim=args.latent_dim).to(device)
    netD = Discriminator().to(device)

    # beta1 = 0.5 instead of the default 0.9 makes Adam more reactive, as usual for GANs.
    optG = optim.Adam(netG.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
    optD = optim.Adam(netD.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    loader = get_celeba_loader(args.data_root, args.batch_size, args.num_workers)

    # --- Resume ---
    epoch = 0
    checkpoints = sorted(glob.glob(os.path.join(ckpt_dir, "*.ckp")))
    if checkpoints:
        cp = torch.load(checkpoints[-1], map_location=device)
        netG.load_state_dict(cp["netG"])
        netD.load_state_dict(cp["netD"])
        if "optG" in cp:
            optG.load_state_dict(cp["optG"])
            optD.load_state_dict(cp["optD"])
        epoch = cp["epoch"]
        print(f"[INFO] Resumed from {checkpoints[-1]} (epoch {epoch})")

    while epoch < args.epochs:
        for i, (real_imgs, all_attrs) in enumerate(loader):
            batch_size = real_imgs.size(0)
            real_imgs = real_imgs.to(device)
            attrs = all_attrs[:, ATTR_INDICES].float().to(device)

            # --- Discriminator ---
            netD.zero_grad()
            # Noisy label smoothing: real targets in [0.9, 1.0], fake targets in [0.0, 0.1]
            label_real = (0.9 + 0.1 * torch.rand(batch_size, 1)).to(device)
            errD_real = criterion(netD(real_imgs, attrs), label_real)

            noise = torch.randn(batch_size, args.latent_dim, device=device)
            fake_imgs = netG(noise, attrs)
            label_fake = (0.1 * torch.rand(batch_size, 1)).to(device)
            errD_fake = criterion(netD(fake_imgs.detach(), attrs), label_fake)

            errD = errD_real + errD_fake
            errD.backward()
            optD.step()

            # --- Generator ---
            netG.zero_grad()
            fake_imgs1 = netG(torch.randn(batch_size, args.latent_dim, device=device), attrs)
            errG_adv = criterion(netD(fake_imgs1, attrs), label_real)  # try to fool the discriminator

            # Per-instance diversity loss: a second batch with the same attributes and new noise.
            fake_imgs2 = netG(torch.randn(batch_size, args.latent_dim, device=device), attrs)
            dists_per_image = torch.mean(torch.abs(fake_imgs1 - fake_imgs2), dim=[1, 2, 3])
            diversity_loss = torch.mean(torch.clamp(args.margin - dists_per_image, min=0.0))

            errG = errG_adv + args.lambda_div * diversity_loss
            errG.backward()
            optG.step()

            if i % 100 == 0:
                print(f"[{epoch}][{i}/{len(loader)}] Loss_D: {errD.item():.4f} "
                      f"Loss_G: {errG.item():.4f} Diversity: {diversity_loss.item():.4f}")

        epoch += 1
        state = {"netG": netG.state_dict(), "netD": netD.state_dict(),
                 "optG": optG.state_dict(), "optD": optD.state_dict(), "epoch": epoch}
        torch.save(state, os.path.join(ckpt_dir, f"{epoch:06d}.ckp"))

        if args.keep_checkpoints > 0:
            for old in sorted(glob.glob(os.path.join(ckpt_dir, "*.ckp")))[:-args.keep_checkpoints]:
                os.remove(old)


if __name__ == "__main__":
    main()
