"""Tile assembler for RoofDiffusion.

This module provides classes for reassembling tiled inference results
into complete outputs with seamless blending.
"""

from typing import List, Tuple, Optional, Literal
from pathlib import Path
import numpy as np
import torch

from data.geo.tile import GeoTile
from data.geo.metadata import GeoMetadata
from data.geo.exporters import BaseGeoExporter


class TileAssembler:
    """Reassembles tiled inference results into complete outputs.

    Handles overlap blending for seamless stitching between tiles.

    Example:
        >>> assembler = TileAssembler(tile_size=(128, 128), overlap=16)
        >>> for tile, result in zip(tiles, results):
        ...     assembler.add_tile(tile, tile.tile_index, result)
        >>> full_height_map, metadata = assembler.assemble()
    """

    def __init__(
        self,
        tile_size: Tuple[int, int],
        overlap: int,
        blend_mode: Literal["linear", "cosine", "none"] = "linear",
    ):
        """Initialize the assembler.

        Args:
            tile_size: Size of each tile (H, W).
            overlap: Overlap in pixels between adjacent tiles.
            blend_mode: Blending strategy for overlapping regions.
                       'linear': Linear interpolation in overlap region.
                       'cosine': Smooth cosine-based blending.
                       'none': No blending, use last-added tile values.
        """
        self.tile_size = tile_size
        self.overlap = overlap
        self.blend_mode = blend_mode

        self.tiles: dict[Tuple[int, int], torch.Tensor] = {}
        self.metadata_list: dict[Tuple[int, int], GeoMetadata] = {}
        self.original_points_list: dict[Tuple[int, int], Optional[np.ndarray]] = {}

        self._grid_shape: Optional[Tuple[int, int]] = None

    def add_tile(
        self,
        tile: GeoTile,
        tile_index: Tuple[int, int],
        result: torch.Tensor,
    ) -> None:
        """Add a processed tile to the assembly buffer.

        Args:
            tile: Original input tile (for metadata).
            tile_index: (row, col) position in tile grid.
            result: Processed height map tensor for this tile.
        """
        row, col = tile_index
        self.tiles[tile_index] = result.cpu()
        self.metadata_list[tile_index] = tile.metadata
        self.original_points_list[tile_index] = tile.original_points

        # Track grid dimensions
        if self._grid_shape is None:
            self._grid_shape = (row + 1, col + 1)
        else:
            self._grid_shape = (
                max(self._grid_shape[0], row + 1),
                max(self._grid_shape[1], col + 1),
            )

    def _create_blend_weights(self, tile_h: int, tile_w: int) -> np.ndarray:
        """Create blending weight matrix for a single tile.

        Args:
            tile_h: Tile height.
            tile_w: Tile width.

        Returns:
            2D weight array with values in [0, 1].
        """
        if self.blend_mode == "none" or self.overlap == 0:
            return np.ones((tile_h, tile_w), dtype=np.float32)

        weights = np.ones((tile_h, tile_w), dtype=np.float32)

        # Create 1D ramps for edges
        if self.blend_mode == "linear":
            ramp = np.linspace(0, 1, self.overlap)
        elif self.blend_mode == "cosine":
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, self.overlap)))
        else:
            ramp = np.ones(self.overlap)

        # Apply ramps to edges
        # Top edge
        if self.overlap <= tile_h:
            weights[: self.overlap, :] *= ramp[:, np.newaxis]
        # Bottom edge
        if self.overlap <= tile_h:
            weights[-self.overlap :, :] *= ramp[::-1, np.newaxis]
        # Left edge
        if self.overlap <= tile_w:
            weights[:, : self.overlap] *= ramp[np.newaxis, :]
        # Right edge
        if self.overlap <= tile_w:
            weights[:, -self.overlap :] *= ramp[::-1][np.newaxis, :]

        return weights

    def assemble(self) -> Tuple[np.ndarray, GeoMetadata]:
        """Assemble all tiles into final output.

        Returns:
            height_map: Complete assembled height map.
            metadata: Merged metadata for full extent.
        """
        if not self.tiles:
            raise ValueError("No tiles added to assembler")

        if self._grid_shape is None:
            raise ValueError("Grid shape not determined")

        n_rows, n_cols = self._grid_shape
        tile_h, tile_w = self.tile_size
        step_h = tile_h - self.overlap
        step_w = tile_w - self.overlap

        # Compute full output size
        full_h = (n_rows - 1) * step_h + tile_h
        full_w = (n_cols - 1) * step_w + tile_w

        # Initialize output arrays
        output = np.zeros((full_h, full_w), dtype=np.float32)
        weight_sum = np.zeros((full_h, full_w), dtype=np.float32)

        # Blend weights template
        blend_weights = self._create_blend_weights(tile_h, tile_w)

        # Accumulate tiles with blending
        for (row, col), tile_tensor in self.tiles.items():
            tile_data = tile_tensor.squeeze().numpy()

            # Handle different tile sizes (edge tiles may be smaller)
            actual_h, actual_w = tile_data.shape

            # Position in output
            row_start = row * step_h
            col_start = col * step_w
            row_end = row_start + actual_h
            col_end = col_start + actual_w

            # Adjust blend weights for edge tiles
            if actual_h != tile_h or actual_w != tile_w:
                tile_weights = self._create_blend_weights(actual_h, actual_w)
            else:
                tile_weights = blend_weights[:actual_h, :actual_w]

            # Skip NaN values in accumulation
            valid_mask = ~np.isnan(tile_data)
            tile_weights = np.where(valid_mask, tile_weights, 0)
            tile_data = np.where(valid_mask, tile_data, 0)

            output[row_start:row_end, col_start:col_end] += tile_data * tile_weights
            weight_sum[row_start:row_end, col_start:col_end] += tile_weights

        # Normalize by weight sum
        valid = weight_sum > 0
        output[valid] /= weight_sum[valid]
        output[~valid] = np.nan

        # Merge metadata
        merged_metadata = self._merge_metadata()

        return output, merged_metadata

    def _merge_metadata(self) -> GeoMetadata:
        """Merge metadata from all tiles into full extent metadata.

        Returns:
            GeoMetadata for the full assembled extent.
        """
        if not self.metadata_list:
            return GeoMetadata()

        # Use first tile's metadata as base
        first_metadata = list(self.metadata_list.values())[0]

        # Compute full bounds from all tiles
        all_bounds = [
            m.bounds
            for m in self.metadata_list.values()
            if m.bounds is not None
        ]

        if all_bounds:
            min_x = min(b[0] for b in all_bounds)
            min_y = min(b[1] for b in all_bounds)
            max_x = max(b[2] for b in all_bounds)
            max_y = max(b[3] for b in all_bounds)
            merged_bounds = (min_x, min_y, max_x, max_y)
        else:
            merged_bounds = None

        # Create new metadata with merged bounds
        merged = GeoMetadata(
            crs=first_metadata.crs,
            resolution=first_metadata.resolution,
            bounds=merged_bounds,
            height_offset=first_metadata.height_offset,
            height_scale=first_metadata.height_scale,
            height_min=first_metadata.height_min,
            height_max=first_metadata.height_max,
            nodata=first_metadata.nodata,
        )

        # Update transform for full extent
        if merged_bounds and merged.resolution:
            merged.transform = (
                merged.resolution,
                0.0,
                merged_bounds[0],
                0.0,
                -merged.resolution,
                merged_bounds[3],
            )

        return merged

    def export(
        self,
        exporter: BaseGeoExporter,
        output_path: Path,
    ) -> None:
        """Assemble and export in one step.

        Args:
            exporter: Exporter to use for output.
            output_path: Path for the output file.
        """
        height_map, metadata = self.assemble()
        height_tensor = torch.from_numpy(height_map).unsqueeze(0)
        exporter.export(height_tensor, metadata, output_path)

    def get_original_points(self) -> Optional[np.ndarray]:
        """Collect all original points from tiles.

        Returns:
            Combined original points array, or None if no points available.
        """
        all_points = [
            pts for pts in self.original_points_list.values()
            if pts is not None
        ]

        if all_points:
            return np.vstack(all_points)
        return None

    def reset(self) -> None:
        """Clear all stored tiles and reset the assembler."""
        self.tiles.clear()
        self.metadata_list.clear()
        self.original_points_list.clear()
        self._grid_shape = None


class StreamingAssembler:
    """Memory-efficient assembler for very large datasets.

    Writes tiles directly to disk, avoiding full dataset in memory.
    Uses memory-mapped files for efficient I/O.

    Example:
        >>> assembler = StreamingAssembler(
        ...     output_path='output.tif',
        ...     full_shape=(10000, 10000),
        ...     tile_size=(128, 128),
        ...     overlap=16,
        ...     metadata=metadata
        ... )
        >>> for tile, result in zip(tiles, results):
        ...     assembler.write_tile(result, tile.tile_index)
        >>> assembler.finalize()
    """

    def __init__(
        self,
        output_path: Path,
        full_shape: Tuple[int, int],
        tile_size: Tuple[int, int],
        overlap: int,
        metadata: GeoMetadata,
        dtype: np.dtype = np.float32,
        blend_mode: Literal["linear", "cosine", "none"] = "linear",
    ):
        """Initialize the streaming assembler.

        Args:
            output_path: Path to output file.
            full_shape: Total (H, W) of assembled output.
            tile_size: Size of each tile.
            overlap: Overlap between tiles.
            metadata: Metadata for full output.
            dtype: Data type for output array.
            blend_mode: Blending mode for overlaps.
        """
        self.output_path = Path(output_path)
        self.full_shape = full_shape
        self.tile_size = tile_size
        self.overlap = overlap
        self.metadata = metadata
        self.dtype = dtype
        self.blend_mode = blend_mode

        self.step_h = tile_size[0] - overlap
        self.step_w = tile_size[1] - overlap

        # Create temporary memory-mapped files
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.output_path.with_suffix(".tmp.npy")
        weight_path = self.output_path.with_suffix(".weights.npy")

        self._output_mmap = np.memmap(
            str(temp_path),
            dtype=np.float64,
            mode="w+",
            shape=full_shape,
        )
        self._weight_mmap = np.memmap(
            str(weight_path),
            dtype=np.float64,
            mode="w+",
            shape=full_shape,
        )

        self._temp_path = temp_path
        self._weight_path = weight_path
        self._finalized = False

    def _create_blend_weights(self, tile_h: int, tile_w: int) -> np.ndarray:
        """Create blending weights for a tile."""
        if self.blend_mode == "none" or self.overlap == 0:
            return np.ones((tile_h, tile_w), dtype=np.float64)

        weights = np.ones((tile_h, tile_w), dtype=np.float64)

        if self.blend_mode == "linear":
            ramp = np.linspace(0, 1, self.overlap)
        elif self.blend_mode == "cosine":
            ramp = 0.5 * (1 - np.cos(np.pi * np.linspace(0, 1, self.overlap)))
        else:
            ramp = np.ones(self.overlap)

        if self.overlap <= tile_h:
            weights[: self.overlap, :] *= ramp[:, np.newaxis]
            weights[-self.overlap :, :] *= ramp[::-1, np.newaxis]
        if self.overlap <= tile_w:
            weights[:, : self.overlap] *= ramp[np.newaxis, :]
            weights[:, -self.overlap :] *= ramp[::-1][np.newaxis, :]

        return weights

    def write_tile(
        self,
        tile_result: np.ndarray,
        tile_index: Tuple[int, int],
    ) -> None:
        """Write a single tile to the output file.

        Args:
            tile_result: Processed tile data (2D array).
            tile_index: (row, col) position in grid.
        """
        if self._finalized:
            raise RuntimeError("Assembler has been finalized")

        row, col = tile_index
        tile_data = tile_result.squeeze()
        actual_h, actual_w = tile_data.shape

        row_start = row * self.step_h
        col_start = col * self.step_w
        row_end = min(row_start + actual_h, self.full_shape[0])
        col_end = min(col_start + actual_w, self.full_shape[1])

        # Adjust for clipping
        data_h = row_end - row_start
        data_w = col_end - col_start

        blend_weights = self._create_blend_weights(actual_h, actual_w)[:data_h, :data_w]
        tile_data = tile_data[:data_h, :data_w]

        # Handle NaN values
        valid_mask = ~np.isnan(tile_data)
        blend_weights = np.where(valid_mask, blend_weights, 0)
        tile_data = np.where(valid_mask, tile_data, 0)

        # Accumulate
        self._output_mmap[row_start:row_end, col_start:col_end] += (
            tile_data * blend_weights
        )
        self._weight_mmap[row_start:row_end, col_start:col_end] += blend_weights

        # Flush periodically
        self._output_mmap.flush()
        self._weight_mmap.flush()

    def finalize(self) -> None:
        """Complete the assembly and write final output."""
        if self._finalized:
            return

        # Normalize by weights
        valid = self._weight_mmap > 0
        self._output_mmap[valid] /= self._weight_mmap[valid]
        self._output_mmap[~valid] = np.nan
        self._output_mmap.flush()

        # Write to final GeoTIFF
        try:
            import rasterio
            from rasterio.transform import Affine
            from rasterio.crs import CRS

            profile = {
                "driver": "GTiff",
                "height": self.full_shape[0],
                "width": self.full_shape[1],
                "count": 1,
                "dtype": str(self.dtype),
                "compress": "lzw",
                "tiled": True,
            }

            if self.metadata.crs:
                try:
                    profile["crs"] = CRS.from_string(self.metadata.crs)
                except Exception:
                    pass

            if self.metadata.transform:
                profile["transform"] = Affine(*self.metadata.transform[:6])

            output_data = self._output_mmap[:].astype(self.dtype)

            with rasterio.open(str(self.output_path), "w", **profile) as dst:
                dst.write(output_data, 1)

        except ImportError:
            # Fallback to numpy save
            np.save(str(self.output_path.with_suffix(".npy")), self._output_mmap[:])

        # Cleanup temp files
        del self._output_mmap
        del self._weight_mmap
        self._temp_path.unlink(missing_ok=True)
        self._weight_path.unlink(missing_ok=True)

        self._finalized = True

    def __del__(self):
        """Cleanup on destruction."""
        if not self._finalized:
            try:
                del self._output_mmap
                del self._weight_mmap
                self._temp_path.unlink(missing_ok=True)
                self._weight_path.unlink(missing_ok=True)
            except Exception:
                pass
