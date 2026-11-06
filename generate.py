"""
Generates a grid of faces for all 8 combinations of [Male, Smiling, Young].

Each row is one attribute combination and each column one sample.
With --fixed-noise the same latent noise is reused on every row, so the
columns show how the same "identity" changes with the attributes.
--truncation < 1 scales the latent noise of the CVAE / ViT-GAN towards the
mean (truncation trick): fewer artifacts, slightly less diversity.

Examples:
    python generate.py --model cvae      --weights weights/cvae.pth
    python generate.py --model vit_gan   --weights weights/vit_gan_G.pth --truncation 0.7 --fixed-noise
    python generate.py --model diffusion --weights weights/diffusion.pth --guidance 2.0
"""

import argparse
import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid

ATTRIBUTE_NAMES = ("Male", "Smiling", "Young")
COMBOS = list(itertools.product([0, 1], repeat=3))  # [Male, Smiling, Young] in {0, 1}


def combo_label(combo):
    male, smiling, young = combo
    return f"{'Man' if male else 'Woman'}, {'smiling' if smiling else 'not smiling'}, {'young' if young else 'old'}"


def load_state_dict(path, device, keys):
    """Loads a plain state dict or extracts it from a training checkpoint under one of `keys`."""
    ckpt = torch.load(path, map_location=device)
    for key in keys:
        if isinstance(ckpt, dict) and key in ckpt:
            return ckpt[key]
    return ckpt


@torch.no_grad()
def generate(args, device, generator):
    n_cols = args.samples
    combos = torch.tensor(COMBOS, dtype=torch.float32, device=device)
    cond01 = combos.repeat_interleave(n_cols, dim=0)  # [8 * n_cols, 3], row-major

    if args.model == "cvae":
        from cvae.model import ConditionalVAE

        model = ConditionalVAE(latent_dim=args.latent_dim or 256, num_labels=3).to(device)
        model.load_state_dict(load_state_dict(args.weights, device, ["model_state_dict"]))
        model.eval()
        z = sample_noise(model.latent_dim, n_cols, args.fixed_noise, device, generator) * args.truncation
        images = model.decode(z, cond01)

    elif args.model == "vit_gan":
        from vit_gan.generator import ViTGenerator

        model = ViTGenerator(latent_dim=args.latent_dim or 128).to(device)
        model.load_state_dict(load_state_dict(args.weights, device, ["netG"]))
        model.eval()
        z = sample_noise(model.latent_dim, n_cols, args.fixed_noise, device, generator) * args.truncation
        images = model(z, cond01)

    else:  # diffusion
        from diffusion.model import Network, NoiseSchedule, ddpm_sample

        model = Network().to(device)
        model.load_state_dict(load_state_dict(args.weights, device, ["model_state_dict"]))
        model.eval()
        schedule = NoiseSchedule(device=device)
        # The diffusion model uses {-1, +1}; 0 is reserved for the null condition.
        return ddpm_sample(model, schedule, cond01 * 2 - 1, args.guidance, generator)

    return (images.clamp(-1, 1) + 1) / 2


def sample_noise(latent_dim, n_cols, fixed, device, generator):
    if fixed:
        z = torch.randn(n_cols, latent_dim, device=device, generator=generator)
        return z.repeat(len(COMBOS), 1)  # same noise for every attribute combination
    return torch.randn(len(COMBOS) * n_cols, latent_dim, device=device, generator=generator)


def save_figure(images, n_cols, path, title):
    grid = make_grid(images.cpu(), nrow=n_cols, padding=2, pad_value=1.0)
    fig_h = 1.0 + 1.1 * len(COMBOS)
    fig, ax = plt.subplots(figsize=(1.1 * n_cols + 2.6, fig_h))
    ax.imshow(grid.permute(1, 2, 0).numpy())
    cell = images.shape[-1] + 2
    ax.set_yticks([cell * i + cell / 2 for i in range(len(COMBOS))])
    ax.set_yticklabels([combo_label(c) for c in COMBOS], fontsize=9)
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=["cvae", "vit_gan", "diffusion"])
    p.add_argument("--weights", required=True, help="state dict or training checkpoint")
    p.add_argument("--output", default=None, help="output PNG (default: samples/<model>.png)")
    p.add_argument("--samples", type=int, default=6, help="samples per attribute combination")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--fixed-noise", action="store_true", help="reuse the same noise across combinations (cvae, vit_gan)")
    p.add_argument("--truncation", type=float, default=1.0, help="scale of the latent noise (cvae, vit_gan)")
    p.add_argument("--guidance", type=float, default=2.0, help="classifier-free guidance scale (diffusion)")
    p.add_argument("--latent-dim", type=int, default=None, help="override the latent size (cvae: 256, vit_gan: 128)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    images = generate(args, device, generator)

    output = args.output or os.path.join("samples", f"{args.model}.png")
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    names = {"cvae": "Conditional VAE", "vit_gan": "ViT-GAN", "diffusion": f"DDPM (guidance {args.guidance})"}
    title = names[args.model]
    if args.model != "diffusion":
        if args.truncation != 1.0:
            title += f" (truncation {args.truncation})"
        if args.fixed_noise:
            title += " — same noise per column"
    save_figure(images, args.samples, output, title)
    print(f"[INFO] Saved {output}")


if __name__ == "__main__":
    main()
