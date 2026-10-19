"""Create smaller copies of raster images while preserving their DPI metadata."""

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
    ratio: float = 1,
    recursive: bool = False,
    dpi: int | None = None,
    name_contains: str | None = None,
) -> tuple[int, int]:
    """Save copies with dimensions divided by ``ratio`` and optional DPI.

    Returns:
        A tuple containing the number of converted files and skipped files.
    """
    if ratio < 1:
        raise ValueError("ratio must be greater than or equal to one")
    if dpi is not None and dpi <= 0:
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
        if (
            not source.is_file()
            or source.suffix.lower() not in SUPPORTED_EXTENSIONS
            or (name_contains is not None and name_contains not in source.name)
        ):
            continue

        destination = output_dir / source.relative_to(input_dir)
        destination.parent.mkdir(parents=True, exist_ok=True)

        try:
            with Image.open(source) as image:
                image.load()
                image_format = image.format
                source_dpi = image.info.get("dpi")
                width = max(1, round(image.width / ratio))
                height = max(1, round(image.height / ratio))
                if (width, height) != image.size:
                    image = image.resize((width, height), Image.Resampling.LANCZOS)
                save_kwargs: dict[str, object] = {}
                if dpi is not None:
                    save_kwargs["dpi"] = (dpi, dpi)
                elif source_dpi is not None:
                    save_kwargs["dpi"] = source_dpi

                if image_format == "JPEG":
                    image = image.convert("RGB")
                    save_kwargs["quality"] = 95
                    save_kwargs["optimize"] = True
                elif image_format == "PNG":
                    save_kwargs["optimize"] = True

                image.save(destination, **save_kwargs)
        except (OSError, UnidentifiedImageError) as error:
            print(f"Skipped {source}: {error}")
            skipped += 1
            continue

        dpi_label = f"{dpi} DPI" if dpi is not None else "DPI unchanged"
        print(f"{source} -> {destination} ({width}x{height}, {dpi_label})")
        converted += 1

    return converted, skipped


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create smaller copies of raster images while preserving DPI metadata."
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
        "--ratio",
        type=float,
        default=1,
        help="Divide image width and height by this ratio (default: 1)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=None,
        help="DPI to write into output files (default: preserve source DPI)",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Process images in subdirectories and preserve their structure",
    )
    parser.add_argument(
        "--name-contains",
        default=None,
        help="Only process files whose name contains this text",
    )
    return parser.parse_args()


def main() -> int:
    """Run the command-line conversion."""
    args = parse_args()
    try:
        converted, skipped = reduce_image_dpi(
            args.input,
            args.output,
            args.ratio,
            args.recursive,
            args.dpi,
            args.name_contains,
        )
    except (NotADirectoryError, ValueError) as error:
        print(f"Error: {error}")
        return 2

    print(f"Converted: {converted}; skipped: {skipped}")
    return 0 if converted > 0 or skipped == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
