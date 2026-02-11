"""Download and prepare a sample image for LATINO super-resolution."""

import argparse
from pathlib import Path
from PIL import Image
from torchvision import transforms


def prepare_image(image, target_size=1024, scale_factor=4):
    """Resize and crop an image so dimensions are compatible with LATINO.

    Requirements:
        - Target dimensions must be divisible by (scale_factor * 8)
          because the SDXL VAE downsamples by 8x spatially and the
          forward operator downsamples by scale_factor.
    """
    divisor = scale_factor * 8

    # Make target_size compatible
    target_size = (target_size // divisor) * divisor
    if target_size == 0:
        target_size = divisor

    transform = transforms.Compose([
        transforms.Resize(target_size),               # resize shortest edge
        transforms.CenterCrop((target_size, target_size)),  # square crop
    ])
    return transform(image)


def download_sample(save_path="sample.png", target_size=1024, scale_factor=4):
    """Download a sample image from HuggingFace datasets."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install the `datasets` library:  pip install datasets"
        )

    print("Downloading sample image from CelebA-HQ...")
    ds = load_dataset("mattymchen/celeba-hq", split="train", streaming=True)
    sample = next(iter(ds))
    image = sample["image"]

    image = prepare_image(image, target_size, scale_factor)
    image.save(save_path)
    print(f"Saved {image.size[0]}x{image.size[1]} image to {save_path}")
    return image


def prepare_local(image_path, save_path=None, target_size=1024, scale_factor=4):
    """Load a local image and resize it to compatible dimensions."""
    image = Image.open(image_path).convert("RGB")
    image = prepare_image(image, target_size, scale_factor)

    if save_path is None:
        p = Path(image_path)
        save_path = p.parent / f"{p.stem}_{image.size[0]}x{image.size[1]}{p.suffix}"

    image.save(save_path)
    print(f"Saved {image.size[0]}x{image.size[1]} image to {save_path}")
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download or prepare an image for LATINO"
    )
    parser.add_argument("--image", type=str, default=None,
                        help="Path to a local image (omit to download a sample)")
    parser.add_argument("--output", type=str, default="sample.png",
                        help="Where to save the prepared image")
    parser.add_argument("--size", type=int, default=1024,
                        help="Target image size (will be snapped to nearest valid)")
    parser.add_argument("--scale-factor", type=int, default=4,
                        help="Downsampling scale factor (for dimension validation)")
    args = parser.parse_args()

    if args.image:
        prepare_local(args.image, args.output, args.size, args.scale_factor)
    else:
        download_sample(args.output, args.size, args.scale_factor)
