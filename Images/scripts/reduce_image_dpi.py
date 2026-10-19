"""Reduce the resolution metadata of raster images without changing their pixels."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Final

from PIL import Image, UnidentifiedImageError

SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset(
    {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
)
Image.MAX_IMAGE_PIXELS = None


def reduce_image_dpi(
    input_dir: Path,
    output_dir: Path,
    dpi: int,
    recursive: bool = False,
) -> tuple[int, int]:
    """Save copies of raster images with the requested DPI metadata.

    Returns:
        A tuple containing the number of converted files and skipped files.
    """
    if dpi <= 0:
        raise ValueError("dpi must be greater than zero")

    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")
    if input_dir == output_dir:
        raise ValueError("Input and output directories must be different")

    pattern = "**/*" if recursive else "*"
    converted = 0
    skipped = 0

    for source in sorted(input_dir.glob(pattern)):
        if not source.is_file() or source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        destination = output_dir / source.relative_to(input_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(source) as image:
                image.load()
                save_kwargs: dict[str, object] = {"dpi": (dpi, dpi)}

                if image.format == "JPEG":
                    image = image.convert("RGB")
                    save_kwargs["quality"] = 95
                    save_kwargs["optimize"] = True
                elif image.format == "PNG":
                    save_kwargs["optimize"] = True

                image.save(destination, **save_kwargs)
        except (OSError, UnidentifiedImageError) as error:
            print(f"Skipped {source}: {error}")
            skipped += 1
            continue

        print(f"{source} -> {destination} ({dpi} DPI)")
        converted += 1

    return converted, skipped


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create copies of raster images with lower DPI metadata."
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("Images"),
        help="Input directory (default: Images)",
    )
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("Images_low_dpi"),
        help="Output directory (default: Images_low_dpi)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI to write into the output files (default: 150)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process images in subdirectories and preserve their structure",
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line conversion."""
    args = parse_args()
    try:
        converted, skipped = reduce_image_dpi(
            args.input, args.output, args.dpi, args.recursive
        )
    except (NotADirectoryError, ValueError) as error:
        print(f"Error: {error}")
        return 2

    print(f"Converted: {converted}; skipped: {skipped}")
    return 0 if converted > 0 or skipped == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
