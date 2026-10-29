import sys
from pathlib import Path

from PIL import Image
from pytest import approx

sys.path.insert(0, str(Path(__file__).parents[1]))

from Images_final.scripts.reduce_image_dpi import reduce_image_dpi


def test_reduce_image_dpi_halves_dimensions_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()

    source = input_dir / "figure.png"
    Image.new("RGBA", (32, 16), (10, 20, 30, 255)).save(source, dpi=(300, 300))

    converted, skipped = reduce_image_dpi(input_dir, output_dir, ratio=2)

    assert (converted, skipped) == (1, 0)
    with Image.open(output_dir / "figure.png") as result:
        assert result.size == (16, 8)
        assert result.info["dpi"] == approx((300, 300), abs=0.02)


def test_reduce_image_dpi_processes_nested_images_when_requested(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    nested_dir = input_dir / "nested"
    nested_dir.mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(nested_dir / "figure.jpg", dpi=(300, 300))

    converted, skipped = reduce_image_dpi(
        input_dir, output_dir, ratio=2, recursive=True
    )

    assert (converted, skipped) == (1, 0)
    assert (output_dir / "nested" / "figure.jpg").is_file()


def test_reduce_image_dpi_can_set_output_metadata(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    Image.new("RGB", (40, 20)).save(input_dir / "figure.jpg", dpi=(600, 600))

    reduce_image_dpi(input_dir, output_dir, ratio=4, dpi=300)

    with Image.open(output_dir / "figure.jpg") as result:
        assert result.size == (10, 5)
        assert result.info["dpi"] == approx((300, 300), abs=0.02)


def test_reduce_image_dpi_can_filter_by_filename(
    tmp_path: Path,
) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    Image.new("RGB", (40, 20)).save(input_dir / "reconstruction_grid.png")
    Image.new("RGB", (40, 20)).save(input_dir / "other.png")

    converted, skipped = reduce_image_dpi(
        input_dir,
        output_dir,
        ratio=8,
        dpi=300,
        name_contains="reconstruction_grid",
    )

    assert (converted, skipped) == (1, 0)
    with Image.open(output_dir / "reconstruction_grid.png") as result:
        assert result.size == (5, 2)
        assert result.info["dpi"] == approx((300, 300), abs=0.02)
    assert not (output_dir / "other.png").exists()
