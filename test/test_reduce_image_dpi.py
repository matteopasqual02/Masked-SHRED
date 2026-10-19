import sys
from pathlib import Path

from PIL import Image
from pytest import approx

sys.path.insert(0, str(Path(__file__).parents[1]))

from Images.scripts.reduce_image_dpi import reduce_image_dpi


def test_reduce_image_dpi_preserves_dimensions_and_sets_metadata(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    source = input_dir / "figure.png"
    Image.new("RGBA", (32, 16), (10, 20, 30, 255)).save(source, dpi=(300, 300))

    converted, skipped = reduce_image_dpi(input_dir, output_dir, dpi=96)

    assert (converted, skipped) == (1, 0)
    with Image.open(output_dir / "figure.png") as result:
        assert result.size == (32, 16)
        assert result.info["dpi"] == approx((96, 96), abs=0.02)


def test_reduce_image_dpi_processes_nested_images_when_requested(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    nested_dir = input_dir / "nested"
    nested_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(nested_dir / "figure.jpg", dpi=(300, 300))

    converted, skipped = reduce_image_dpi(
        input_dir, output_dir, dpi=72, recursive=True
    )

    assert (converted, skipped) == (1, 0)
    assert (output_dir / "nested" / "figure.jpg").is_file()
