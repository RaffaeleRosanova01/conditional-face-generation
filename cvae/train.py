"""
Trains the conditional VAE on CelebA (64x64, attributes Male / Smiling / Young).

Usage (from the repository root):
    python -m cvae.train --data-root /path/to/celeba

--data-root is the folder that directly contains list_attr_celeba.(txt|csv)
and img_align_celeba/. The full dataset is split 80/20 into train and
validation; the best model on the validation loss is saved as a plain
state dict, with early stopping.

Loss = MSE reconstruction (sum) + beta * KL(q(z|x,c) || N(0, I)).
"""

import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

from cvae.dataset import CelebADataset
from cvae.model import ConditionalVAE


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True)
    p.add_argument("--output", default="outputs/cvae/best_model.pth")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=256)
    p.add_argument("--beta", type=float, default=1.0, help="weight of the KL term")
    p.add_argument("--patience", type=int, default=10, help="early stopping patience (epochs)")
    p.add_argument("--seed", type=int, default=42, help="seed of the train/validation split")
    p.add_argument("--num-workers", type=int, default=1)
    return p.parse_args()


def kl_loss_function(mu, log_var):
    """Analytical KL divergence from N(mu, sigma^2) to N(0, I): -0.5 * sum(1 + log sigma^2 - mu^2 - sigma^2)."""
    return -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),  # range [-1, 1], matches the Tanh output
    ])

    full_dataset = CelebADataset(root_dir=args.data_root, transform=transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(args.seed))
    print(f"[INFO] Train: {len(train_set)} images | Validation: {len(val_set)} images")

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    model = ConditionalVAE(latent_dim=args.latent_dim, num_labels=3).to(device)
    if os.path.exists(args.output):
        print(f"[INFO] Resuming from {args.output}")
        model.load_state_dict(torch.load(args.output, map_location=device))

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    reconstruction_loss = nn.MSELoss(reduction="sum")  # images in [-1, 1]

    def loss_function(output, original, mu, log_var):
        return reconstruction_loss(output, original) + args.beta * kl_loss_function(mu, log_var)

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, labels) in enumerate(train_loader):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            output, mu, log_sigma = model(data, labels)
            loss = loss_function(output, data, mu, log_sigma)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 250 == 0:
                print(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}]  loss/img: {loss.item() / len(data):.4f}")
        avg_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data, labels in val_loader:
                data, labels = data.to(device), labels.to(device)
                output, mu, log_sigma = model(data, labels)
                val_loss += loss_function(output, data, mu, log_sigma).item()
        avg_val_loss = val_loss / len(val_loader.dataset)

        print(f"====> Epoch {epoch} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        # Early stopping on the validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), args.output)
            patience_counter = 0
            print(f"     New best model saved to {args.output}")
        else:
            patience_counter += 1
            print(f"     No improvement ({patience_counter}/{args.patience}), best: {best_val_loss:.4f}")
            if patience_counter >= args.patience:
                print("[INFO] Early stopping.")
                break

    print(f"[INFO] Training completed. Best model: {args.output}")


if __name__ == "__main__":
    main()
