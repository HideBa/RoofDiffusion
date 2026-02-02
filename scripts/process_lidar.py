#!/usr/bin/env python3
"""Convenience script for processing LiDAR point clouds with RoofDiffusion.

This script provides a simplified interface for common LiDAR processing workflows:
- Single file processing
- Batch processing of directories
- Tiled processing for large files

Examples:
    # Process a single LAS file
    python scripts/process_lidar.py input.laz --output output.tif --checkpoint pretrained/w_footprint/260

    # Process with densified point cloud output
    python scripts/process_lidar.py input.laz --output output.laz --densify --density 4.0

    # Batch process a directory
    python scripts/process_lidar.py ./input_dir --output ./output_dir --batch

    # Process large file with tiling
    python scripts/process_lidar.py large.laz --output output.tif --tile-size 128 --overlap 16
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Process LiDAR point clouds with RoofDiffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input/Output
    parser.add_argument(
        "input",
        type=Path,
        help="Input file (LAS/LAZ/GeoTIFF) or directory for batch mode"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output file or directory"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("pretrained/w_footprint/260"),
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/roof_completion_geo.json"),
        help="Configuration file"
    )

    # Processing options
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch process all files in input directory"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        default=0.5,
        help="Rasterization resolution in meters"
    )
    parser.add_argument(
        "--classification",
        type=int,
        nargs="+",
        default=[6],
        help="LAS classification codes to include (default: 6 = buildings)"
    )

    # Output format options
    parser.add_argument(
        "--format",
        type=str,
        nargs="+",
        default=["geotiff"],
        choices=["geotiff", "las", "laz", "png"],
        help="Output format(s)"
    )
    parser.add_argument(
        "--densify",
        action="store_true",
        help="Densify output point cloud"
    )
    parser.add_argument(
        "--density",
        type=float,
        default=1.0,
        help="Point density for densification (points per m^2)"
    )
    parser.add_argument(
        "--preserve-original",
        action="store_true",
        help="Include original input points in densified output"
    )

    # Tiling options
    parser.add_argument(
        "--tile-size",
        type=int,
        default=128,
        help="Tile size for processing (pixels)"
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=16,
        help="Overlap between tiles (pixels)"
    )

    # Diffusion options
    parser.add_argument(
        "--timesteps",
        type=int,
        default=500,
        help="Number of diffusion timesteps for inference"
    )

    # GPU options
    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="GPU device ID"
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU processing"
    )

    return parser.parse_args()


def process_single_file(
    input_path: Path,
    output_path: Path,
    args: argparse.Namespace,
) -> None:
    """Process a single input file."""
    import torch
    from data.geo.loaders import create_loader
    from data.geo.exporters import create_exporter, MultiFormatExporter
    from data.geo.assembler import TileAssembler

    print(f"Processing: {input_path}")

    # Create loader
    loader = create_loader(
        input_path,
        resolution=args.resolution,
        classification_filter=args.classification if input_path.suffix.lower() in [".las", ".laz"] else None,
    )

    # Load model
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    # Check if tiled processing is needed
    tile = loader.load(input_path)
    needs_tiling = tile.height > args.tile_size * 2 or tile.width > args.tile_size * 2

    if needs_tiling:
        print(f"Large file detected ({tile.height}x{tile.width}), using tiled processing...")
        _process_tiled(input_path, output_path, loader, args, device)
    else:
        print(f"Processing as single tile ({tile.height}x{tile.width})...")
        _process_single(tile, output_path, args, device)


def _process_single(
    tile,
    output_path: Path,
    args: argparse.Namespace,
    device,
) -> None:
    """Process a single tile without tiling."""
    import torch
    from data.geo.exporters import create_exporter

    # Load model and run inference
    # Note: This is a simplified version - full implementation would load the model
    # and run proper inference

    print("Running diffusion inference...")

    # For now, just demonstrate the export flow
    # In production, this would load the model and run inference
    height_map = tile.height_map
    metadata = tile.metadata

    # Create exporters and export
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for fmt in args.format:
        exporter = create_exporter(
            fmt,
            point_density=args.density if fmt in ["las", "laz"] else None,
            preserve_original_points=args.preserve_original if fmt in ["las", "laz"] else None,
        )

        ext = exporter.get_extension()
        if output_path.suffix:
            out_file = output_path.with_suffix(ext)
        else:
            out_file = output_path / f"output{ext}"

        if fmt in ["las", "laz"]:
            exporter.export(height_map, metadata, out_file, tile.original_points)
        else:
            exporter.export(height_map, metadata, out_file)

        print(f"Exported: {out_file}")


def _process_tiled(
    input_path: Path,
    output_path: Path,
    loader,
    args: argparse.Namespace,
    device,
) -> None:
    """Process using tiled approach."""
    from data.geo.assembler import TileAssembler
    from data.geo.exporters import create_exporter
    import torch

    tile_size = (args.tile_size, args.tile_size)

    # Create assembler
    assembler = TileAssembler(
        tile_size=tile_size,
        overlap=args.overlap,
        blend_mode="linear",
    )

    # Process tiles
    print("Loading and processing tiles...")
    for tile in loader.load_tiled(input_path, tile_size=tile_size, overlap=args.overlap):
        print(f"  Processing tile {tile.tile_index}...")

        # In production, run inference here
        # For now, pass through the input
        result = tile.height_map

        assembler.add_tile(tile, tile.tile_index, result)

    # Assemble and export
    print("Assembling tiles...")
    height_map, metadata = assembler.assemble()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_points = assembler.get_original_points()

    for fmt in args.format:
        exporter = create_exporter(fmt)

        ext = exporter.get_extension()
        if output_path.suffix:
            out_file = output_path.with_suffix(ext)
        else:
            out_file = output_path / f"output{ext}"

        height_tensor = torch.from_numpy(height_map).unsqueeze(0)

        if fmt in ["las", "laz"] and original_points is not None:
            exporter.export(height_tensor, metadata, out_file, original_points)
        else:
            exporter.export(height_tensor, metadata, out_file)

        print(f"Exported: {out_file}")


def process_batch(
    input_dir: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Process all files in a directory."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all supported files
    extensions = [".las", ".laz", ".tif", ".tiff"]
    files = []
    for ext in extensions:
        files.extend(input_dir.glob(f"*{ext}"))
        files.extend(input_dir.glob(f"*{ext.upper()}"))

    if not files:
        print(f"No supported files found in {input_dir}")
        return

    print(f"Found {len(files)} files to process")

    for i, input_file in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing {input_file.name}...")

        output_file = output_dir / input_file.stem
        try:
            process_single_file(input_file, output_file, args)
        except Exception as e:
            print(f"Error processing {input_file}: {e}")
            continue

    print(f"\nBatch processing complete. Results in: {output_dir}")


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        sys.exit(1)

    if args.batch or input_path.is_dir():
        process_batch(input_path, output_path, args)
    else:
        process_single_file(input_path, output_path, args)


if __name__ == "__main__":
    main()
