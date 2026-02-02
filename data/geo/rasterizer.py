"""GeoRasterizer for converting point clouds to raster grids.

This module provides functionality to convert point cloud data (e.g., from LAS/LAZ files)
into raster height maps suitable for processing by the RoofDiffusion model.
"""

from typing import Tuple, Optional, Literal
import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata

from data.geo.metadata import GeoMetadata


class GeoRasterizer:
    """Converts point clouds to raster height maps.

    Supports multiple rasterization strategies and handles
    sparse point clouds gracefully through optional interpolation.

    Example:
        >>> rasterizer = GeoRasterizer(resolution=0.5, method='max')
        >>> height_map, footprint, metadata = rasterizer.rasterize(points)
    """

    def __init__(
        self,
        resolution: float = 0.5,
        method: Literal["max", "mean", "median", "min", "count"] = "max",
        fill_method: Literal["none", "nearest", "linear", "cubic"] = "none",
        fill_radius: Optional[float] = None,
    ):
        """Initialize the rasterizer.

        Args:
            resolution: Grid cell size in world units (meters).
            method: Aggregation method when multiple points fall in same cell.
                   'max' is typical for DSM (top of canopy/building),
                   'mean' for DTM (ground level estimation).
            fill_method: Interpolation to fill gaps in sparse data.
                        'none' = no filling, keep as nodata
                        'nearest' = nearest neighbor
                        'linear' = linear interpolation
                        'cubic' = cubic interpolation
            fill_radius: Maximum distance in cells for gap filling.
                        If None, no distance limit is applied.
        """
        self.resolution = resolution
        self.method = method
        self.fill_method = fill_method
        self.fill_radius = fill_radius

    def rasterize(
        self,
        points: np.ndarray,
        bounds: Optional[Tuple[float, float, float, float]] = None,
        crs: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, GeoMetadata]:
        """Convert point cloud to raster.

        Args:
            points: Nx3+ array of (x, y, z, ...) coordinates.
                   First three columns must be x, y, z.
            bounds: Optional (min_x, min_y, max_x, max_y) bounds.
                   If None, derived from point extent.
            crs: Optional coordinate reference system string.

        Returns:
            height_map: 2D array of heights (HxW).
            footprint: 2D binary mask of valid cells (HxW).
            metadata: GeoMetadata with transform info.
        """
        if points.shape[0] == 0:
            raise ValueError("Empty point cloud provided")

        if points.shape[1] < 3:
            raise ValueError(f"Points must have at least 3 columns (x,y,z), got {points.shape[1]}")

        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]

        # Determine bounds
        if bounds is None:
            min_x, max_x = x.min(), x.max()
            min_y, max_y = y.min(), y.max()
        else:
            min_x, min_y, max_x, max_y = bounds

        # Compute grid dimensions
        width = int(np.ceil((max_x - min_x) / self.resolution))
        height = int(np.ceil((max_y - min_y) / self.resolution))

        # Ensure minimum size
        width = max(width, 1)
        height = max(height, 1)

        # Compute pixel indices for each point
        col_idx = ((x - min_x) / self.resolution).astype(np.int32)
        row_idx = ((max_y - y) / self.resolution).astype(np.int32)  # Flip y for image coords

        # Clip to valid range
        col_idx = np.clip(col_idx, 0, width - 1)
        row_idx = np.clip(row_idx, 0, height - 1)

        # Initialize output arrays
        height_map = np.full((height, width), np.nan, dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.int32)

        # Aggregate points into grid cells
        if self.method == "max":
            # Use numpy advanced indexing for max aggregation
            height_map = self._aggregate_max(row_idx, col_idx, z, height, width)
        elif self.method == "min":
            height_map = self._aggregate_min(row_idx, col_idx, z, height, width)
        elif self.method == "mean":
            height_map, count_map = self._aggregate_mean(row_idx, col_idx, z, height, width)
        elif self.method == "median":
            height_map = self._aggregate_median(row_idx, col_idx, z, height, width)
        elif self.method == "count":
            height_map = self._aggregate_count(row_idx, col_idx, height, width).astype(np.float32)
        else:
            raise ValueError(f"Unknown aggregation method: {self.method}")

        # Create footprint from valid cells
        footprint = (~np.isnan(height_map)).astype(np.float32)

        # Apply gap filling if requested
        if self.fill_method != "none" and np.any(np.isnan(height_map)):
            height_map = self._fill_gaps(height_map, footprint)

        # Create metadata
        # Transform: (x_res, x_skew, x_origin, y_skew, -y_res, y_origin)
        # Standard north-up: (res, 0, min_x, 0, -res, max_y)
        transform = (self.resolution, 0.0, min_x, 0.0, -self.resolution, max_y)

        metadata = GeoMetadata(
            crs=crs,
            transform=transform,
            bounds=(min_x, min_y, max_x, max_y),
            resolution=self.resolution,
            source_path=None,
        )

        return height_map, footprint, metadata

    def _aggregate_max(
        self, row_idx: np.ndarray, col_idx: np.ndarray, z: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        """Aggregate points using maximum value per cell."""
        height_map = np.full((height, width), np.nan, dtype=np.float32)

        # Sort by z descending, then use unique to keep first (max)
        sort_idx = np.argsort(-z)
        row_sorted = row_idx[sort_idx]
        col_sorted = col_idx[sort_idx]
        z_sorted = z[sort_idx]

        # Create linear indices
        linear_idx = row_sorted * width + col_sorted

        # Find unique indices (keeps first occurrence, which is max due to sorting)
        _, unique_idx = np.unique(linear_idx, return_index=True)

        height_map.flat[linear_idx[unique_idx]] = z_sorted[unique_idx]
        return height_map

    def _aggregate_min(
        self, row_idx: np.ndarray, col_idx: np.ndarray, z: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        """Aggregate points using minimum value per cell."""
        height_map = np.full((height, width), np.nan, dtype=np.float32)

        # Sort by z ascending
        sort_idx = np.argsort(z)
        row_sorted = row_idx[sort_idx]
        col_sorted = col_idx[sort_idx]
        z_sorted = z[sort_idx]

        linear_idx = row_sorted * width + col_sorted
        _, unique_idx = np.unique(linear_idx, return_index=True)

        height_map.flat[linear_idx[unique_idx]] = z_sorted[unique_idx]
        return height_map

    def _aggregate_mean(
        self, row_idx: np.ndarray, col_idx: np.ndarray, z: np.ndarray, height: int, width: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate points using mean value per cell."""
        sum_map = np.zeros((height, width), dtype=np.float64)
        count_map = np.zeros((height, width), dtype=np.int32)

        # Accumulate sums and counts
        np.add.at(sum_map, (row_idx, col_idx), z)
        np.add.at(count_map, (row_idx, col_idx), 1)

        # Compute mean where count > 0
        height_map = np.full((height, width), np.nan, dtype=np.float32)
        valid = count_map > 0
        height_map[valid] = (sum_map[valid] / count_map[valid]).astype(np.float32)

        return height_map, count_map

    def _aggregate_median(
        self, row_idx: np.ndarray, col_idx: np.ndarray, z: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        """Aggregate points using median value per cell."""
        height_map = np.full((height, width), np.nan, dtype=np.float32)

        # Create linear indices
        linear_idx = row_idx * width + col_idx

        # Group z values by cell
        unique_cells, inverse = np.unique(linear_idx, return_inverse=True)

        for i, cell in enumerate(unique_cells):
            cell_z = z[inverse == i]
            row, col = divmod(cell, width)
            height_map[row, col] = np.median(cell_z)

        return height_map

    def _aggregate_count(
        self, row_idx: np.ndarray, col_idx: np.ndarray, height: int, width: int
    ) -> np.ndarray:
        """Count points per cell."""
        count_map = np.zeros((height, width), dtype=np.int32)
        np.add.at(count_map, (row_idx, col_idx), 1)
        return count_map

    def _fill_gaps(self, height_map: np.ndarray, footprint: np.ndarray) -> np.ndarray:
        """Fill gaps in height map using interpolation.

        Args:
            height_map: Height map with NaN gaps.
            footprint: Binary mask of valid cells.

        Returns:
            Height map with gaps filled.
        """
        if self.fill_method == "none":
            return height_map

        valid_mask = ~np.isnan(height_map)
        if not np.any(valid_mask):
            return height_map

        # Get coordinates of valid and invalid points
        rows, cols = np.indices(height_map.shape)
        valid_points = np.column_stack([rows[valid_mask], cols[valid_mask]])
        valid_values = height_map[valid_mask]

        invalid_mask = np.isnan(height_map)
        if not np.any(invalid_mask):
            return height_map

        invalid_points = np.column_stack([rows[invalid_mask], cols[invalid_mask]])

        # Apply fill radius constraint if specified
        if self.fill_radius is not None:
            from scipy.spatial import cKDTree

            tree = cKDTree(valid_points)
            distances, _ = tree.query(invalid_points, k=1)
            within_radius = distances <= self.fill_radius
            invalid_points_to_fill = invalid_points[within_radius]
        else:
            invalid_points_to_fill = invalid_points

        if len(invalid_points_to_fill) == 0:
            return height_map

        # Interpolate
        try:
            filled_values = griddata(
                valid_points,
                valid_values,
                invalid_points_to_fill,
                method=self.fill_method if self.fill_method != "none" else "nearest",
            )

            result = height_map.copy()
            if self.fill_radius is not None:
                result[invalid_mask] = np.nan
                fill_rows = invalid_points_to_fill[:, 0]
                fill_cols = invalid_points_to_fill[:, 1]
                result[fill_rows, fill_cols] = filled_values
            else:
                result[invalid_mask] = filled_values

            return result
        except Exception:
            # Fall back to nearest neighbor on interpolation failure
            if self.fill_method != "nearest":
                filled_values = griddata(
                    valid_points,
                    valid_values,
                    invalid_points_to_fill,
                    method="nearest",
                )
                result = height_map.copy()
                result[invalid_mask] = filled_values
                return result
            return height_map

    def compute_footprint(
        self,
        points: np.ndarray,
        resolution: Optional[float] = None,
        density_threshold: int = 1,
        bounds: Optional[Tuple[float, float, float, float]] = None,
    ) -> Tuple[np.ndarray, GeoMetadata]:
        """Generate footprint mask from point density.

        Args:
            points: Nx2+ array of (x, y, ...) coordinates.
            resolution: Grid cell size. If None, uses self.resolution.
            density_threshold: Minimum points per cell to be considered valid.
            bounds: Optional (min_x, min_y, max_x, max_y) bounds.

        Returns:
            footprint: Binary mask where 1 = sufficient points, 0 = sparse/empty.
            metadata: GeoMetadata with transform info.
        """
        if resolution is None:
            resolution = self.resolution

        x = points[:, 0]
        y = points[:, 1]

        if bounds is None:
            min_x, max_x = x.min(), x.max()
            min_y, max_y = y.min(), y.max()
        else:
            min_x, min_y, max_x, max_y = bounds

        width = int(np.ceil((max_x - min_x) / resolution))
        height = int(np.ceil((max_y - min_y) / resolution))
        width = max(width, 1)
        height = max(height, 1)

        col_idx = ((x - min_x) / resolution).astype(np.int32)
        row_idx = ((max_y - y) / resolution).astype(np.int32)
        col_idx = np.clip(col_idx, 0, width - 1)
        row_idx = np.clip(row_idx, 0, height - 1)

        count_map = np.zeros((height, width), dtype=np.int32)
        np.add.at(count_map, (row_idx, col_idx), 1)

        footprint = (count_map >= density_threshold).astype(np.float32)

        transform = (resolution, 0.0, min_x, 0.0, -resolution, max_y)
        metadata = GeoMetadata(
            transform=transform,
            bounds=(min_x, min_y, max_x, max_y),
            resolution=resolution,
        )

        return footprint, metadata

    def resample(
        self,
        height_map: np.ndarray,
        metadata: GeoMetadata,
        target_resolution: float,
        method: Literal["nearest", "bilinear", "cubic"] = "bilinear",
    ) -> Tuple[np.ndarray, GeoMetadata]:
        """Resample height map to different resolution.

        Args:
            height_map: Input height map.
            metadata: Input metadata with current transform.
            target_resolution: Target resolution in world units.
            method: Resampling method.

        Returns:
            resampled: Resampled height map.
            new_metadata: Updated metadata with new transform.
        """
        if metadata.resolution is None:
            raise ValueError("Source resolution not specified in metadata")

        scale = metadata.resolution / target_resolution

        # Compute new dimensions
        new_height = int(round(height_map.shape[0] * scale))
        new_width = int(round(height_map.shape[1] * scale))
        new_height = max(new_height, 1)
        new_width = max(new_width, 1)

        # Map method to scipy order
        order_map = {"nearest": 0, "bilinear": 1, "cubic": 3}
        order = order_map.get(method, 1)

        # Handle NaN values for resampling
        mask = ~np.isnan(height_map)
        height_filled = np.where(mask, height_map, 0)

        resampled = ndimage.zoom(height_filled, scale, order=order)
        mask_resampled = ndimage.zoom(mask.astype(float), scale, order=0) > 0.5

        # Restore NaN where original was invalid
        resampled = np.where(mask_resampled, resampled, np.nan)

        # Update metadata
        if metadata.bounds is not None:
            new_metadata = metadata.with_new_transform(metadata.bounds, target_resolution)
        else:
            new_metadata = GeoMetadata(
                crs=metadata.crs,
                resolution=target_resolution,
                height_offset=metadata.height_offset,
                height_scale=metadata.height_scale,
            )

        return resampled, new_metadata
