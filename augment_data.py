from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from PIL import Image, ImageEnhance, ImageFilter, ImageOps, UnidentifiedImageError

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
RESAMPLING = Image.Resampling


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _fill_color(img: Image.Image) -> tuple[int, int, int]:
    # Use an average color so empty borders blend naturally after transforms.
    return img.resize((1, 1), RESAMPLING.BILINEAR).getpixel((0, 0))


def _small_translation(img: Image.Image, rng: random.Random) -> Image.Image:
    # Tiny x/y shift to simulate camera movement while keeping the fruit centered.
    width, height = img.size
    max_dx = int(width * 0.08)
    max_dy = int(height * 0.08)
    dx = rng.randint(-max_dx, max_dx) if max_dx > 0 else 0
    dy = rng.randint(-max_dy, max_dy) if max_dy > 0 else 0
    return img.transform(
        img.size,
        Image.Transform.AFFINE,
        (1, 0, dx, 0, 1, dy),
        resample=RESAMPLING.BICUBIC,
        fillcolor=_fill_color(img),
    )


def _small_rotation(img: Image.Image, rng: random.Random) -> Image.Image:
    # Small tilt only (not large rotations) so class meaning stays realistic.
    angle = rng.uniform(-8.0, 8.0)
    return img.rotate(
        angle,
        resample=RESAMPLING.BICUBIC,
        expand=False,
        fillcolor=_fill_color(img),
    )


def _small_zoom_in(img: Image.Image, rng: random.Random) -> Image.Image:
    # Slight crop + resize acts like a soft zoom.
    width, height = img.size
    zoom = rng.uniform(0.92, 1.0)
    crop_w = max(2, int(width * zoom))
    crop_h = max(2, int(height * zoom))
    max_x0 = max(0, width - crop_w)
    max_y0 = max(0, height - crop_h)
    x0 = rng.randint(0, max_x0) if max_x0 > 0 else 0
    y0 = rng.randint(0, max_y0) if max_y0 > 0 else 0
    cropped = img.crop((x0, y0, x0 + crop_w, y0 + crop_h))
    return cropped.resize((width, height), RESAMPLING.BICUBIC)


def _small_color_jitter(img: Image.Image, rng: random.Random) -> Image.Image:
    # Mild lighting/color variation to mimic real photo conditions.
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.94, 1.06))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.94, 1.06))
    img = ImageEnhance.Color(img).enhance(rng.uniform(0.94, 1.06))
    return img


def augment_image(
    img: Image.Image,
    rng: random.Random,
    blur_probability: float,
    max_blur_radius: float,
) -> Image.Image:
    augmented = img.copy()

    # Mirror half of the time to diversify orientation.
    if rng.random() < 0.5:
        augmented = ImageOps.mirror(augmented)

    # Apply only gentle transforms to preserve fruit visual essence.
    augmented = _small_translation(augmented, rng)
    augmented = _small_rotation(augmented, rng)
    augmented = _small_zoom_in(augmented, rng)
    augmented = _small_color_jitter(augmented, rng)

    # Blur is optional and very light by design.
    if rng.random() < blur_probability:
        augmented = augmented.filter(
            ImageFilter.GaussianBlur(radius=rng.uniform(0.2, max_blur_radius))
        )

    return augmented


def _iter_source_images(input_dir: Path, output_dir: Path) -> list[Path]:
    images: list[Path] = []
    output_inside_input = _is_relative_to(output_dir, input_dir)

    for file_path in sorted(input_dir.rglob("*")):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        # Skip previously generated files so we don't augment augmentations.
        if file_path.name.startswith("aug_"):
            continue
        if output_inside_input and _is_relative_to(file_path, output_dir):
            continue
        images.append(file_path)

    return images


def _save_image(img: Image.Image, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = out_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        img.save(out_path, quality=95, optimize=True)
    elif suffix == ".png":
        img.save(out_path, optimize=True, compress_level=4)
    elif suffix == ".webp":
        img.save(out_path, quality=95, method=6)
    else:
        img.save(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply gentle data augmentation to all images in a dataset without "
            "converting images to grayscale."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data"),
        help="Root folder with class subfolders (default: data).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_augmented"),
        help="Where augmented data is written (default: data_augmented).",
    )
    parser.add_argument(
        "--augmentations-per-image",
        type=int,
        default=2,
        help="How many augmented copies to create for each original image.",
    )
    parser.add_argument(
        "--blur-probability",
        type=float,
        default=0.25,
        help="Chance to apply slight blur to each augmented image.",
    )
    parser.add_argument(
        "--max-blur-radius",
        type=float,
        default=0.7,
        help="Maximum blur radius; keep small to preserve fruit details.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible augmentations.",
    )
    parser.add_argument(
        "--copy-originals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy original images into output-dir (default: true).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if args.augmentations_per_image < 1:
        raise ValueError("--augmentations-per-image must be >= 1")
    if not (0.0 <= args.blur_probability <= 1.0):
        raise ValueError("--blur-probability must be between 0 and 1")
    if args.max_blur_radius <= 0:
        raise ValueError("--max-blur-radius must be > 0")

    rng = random.Random(args.seed)
    source_images = _iter_source_images(input_dir, output_dir)
    if not source_images:
        raise RuntimeError(f"No images found under: {input_dir}")

    copied = 0
    created = 0
    skipped = 0
    failed = 0

    for src_path in source_images:
        rel_path = src_path.relative_to(input_dir)
        dst_original = output_dir / rel_path

        # Optional: keep originals in output so one folder can train directly.
        if args.copy_originals and src_path.resolve() != dst_original.resolve():
            if dst_original.exists() and not args.overwrite:
                skipped += 1
            else:
                dst_original.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dst_original)
                copied += 1

        try:
            with Image.open(src_path) as source:
                base_image = source.convert("RGB")
        except (UnidentifiedImageError, OSError) as err:
            failed += 1
            print(f"Skipping unreadable image: {src_path} ({err})")
            continue

        for idx in range(1, args.augmentations_per_image + 1):
            # Example: Image_5.jpg -> aug_Image_5_1.jpg, aug_Image_5_2.jpg
            out_name = f"aug_{src_path.stem}_{idx}{src_path.suffix.lower()}"
            out_path = output_dir / rel_path.parent / out_name

            if out_path.exists() and not args.overwrite:
                skipped += 1
                continue

            aug = augment_image(
                base_image,
                rng=rng,
                blur_probability=args.blur_probability,
                max_blur_radius=args.max_blur_radius,
            )
            _save_image(aug, out_path)
            created += 1

    print(f"Input images found: {len(source_images)}")
    print(f"Originals copied: {copied}")
    print(f"Augmented images created: {created}")
    print(f"Skipped existing files: {skipped}")
    print(f"Unreadable images: {failed}")
    print(f"Done. Augmented dataset is in: {output_dir}")


if __name__ == "__main__":
    main()
