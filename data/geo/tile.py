"""GeoTile data container for RoofDiffusion.

This module defines the GeoTile dataclass that represents a single processing
unit with its height data and geospatial metadata.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import torch

from data.geo.metadata import GeoMetadata


@dataclass
class GeoTile:
    """A single tile with height data and geospatial metadata.

    This is the primary data container passed through the processing pipeline.
    It encapsulates both the image data and all necessary geospatial information
    for accurate export.

    Attributes:
        height_map: Height map as tensor (C, H, W) - normalized to [-1, 1]
        footprint: Footprint/validity mask (1, H, W) - 1=valid, 0=invalid
        metadata: Geospatial metadata (CRS, transform, bounds, etc.)
        original_points: Original point cloud if input was LAS/LAZ (for attribute preservation)
        tile_index: Tile index (row, col) for reassembly of tiled datasets
    """

    # Height map as tensor (C, H, W) - normalized to [-1, 1]
    height_map: torch.Tensor

    # Footprint/validity mask (1, H, W) - 1=valid, 0=invalid
    footprint: torch.Tensor

    # Geospatial metadata
    metadata: GeoMetadata

    # Original point cloud (if input was LAS/LAZ)
    # Stored for point-wise attribute preservation during export
    # Shape: (N, 3+) where columns are (x, y, z, ...)
    original_points: Optional[np.ndarray] = None

    # Tile index for reassembly of large datasets
    tile_index: Optional[Tuple[int, int]] = None

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (height, width) of the tile."""
        return (self.height_map.shape[-2], self.height_map.shape[-1])

    @property
    def height(self) -> int:
        """Return height of the tile in pixels."""
        return self.height_map.shape[-2]

    @property
    def width(self) -> int:
        """Return width of the tile in pixels."""
        return self.height_map.shape[-1]

    @property
    def channels(self) -> int:
        """Return number of channels."""
        return self.height_map.shape[-3] if self.height_map.dim() >= 3 else 1

    def to_numpy(self) -> np.ndarray:
        """Convert height map to numpy array (H, W) in world height units.

        Applies denormalization using metadata height_offset and height_scale.

        Returns:
            2D numpy array of heights in world units (meters).
        """
        height_np = self.height_map.squeeze().cpu().numpy()

        # Denormalize from [-1, 1] to original height range
        if self.metadata.height_min is not None and self.metadata.height_max is not None:
            height_np = (height_np + 1) / 2  # [-1, 1] -> [0, 1]
            height_np = (
                height_np * (self.metadata.height_max - self.metadata.height_min)
                + self.metadata.height_min
            )
        elif self.metadata.height_scale != 1.0 or self.metadata.height_offset != 0.0:
            height_np = (
                height_np * self.metadata.height_scale + self.metadata.height_offset
            )

        return height_np

    def to_device(self, device: torch.device) -> "GeoTile":
        """Move tensors to specified device.

        Args:
            device: Target device (cuda, cpu, etc.)

        Returns:
            New GeoTile with tensors on the specified device.
        """
        return GeoTile(
            height_map=self.height_map.to(device),
            footprint=self.footprint.to(device),
            metadata=self.metadata,
            original_points=self.original_points,
            tile_index=self.tile_index,
        )

    def clone(self) -> "GeoTile":
        """Create a deep copy of this tile.

        Returns:
            New GeoTile with cloned tensors and copied metadata.
        """
        return GeoTile(
            height_map=self.height_map.clone(),
            footprint=self.footprint.clone(),
            metadata=self.metadata.copy(),
            original_points=self.original_points.copy()
            if self.original_points is not None
            else None,
            tile_index=self.tile_index,
        )

    def get_world_bounds(self) -> Optional[Tuple[float, float, float, float]]:
        """Get world coordinate bounds of this tile.

        Returns:
            Bounding box (min_x, min_y, max_x, max_y) or None if no metadata.
        """
        return self.metadata.bounds

    def get_valid_mask(self) -> np.ndarray:
        """Get validity mask as numpy array.

        Returns:
            2D boolean numpy array where True = valid pixel.
        """
        return self.footprint.squeeze().cpu().numpy() > 0.5

    @classmethod
    def from_numpy(
        cls,
        height_map: np.ndarray,
        footprint: Optional[np.ndarray] = None,
        metadata: Optional[GeoMetadata] = None,
        normalize: bool = True,
        height_min: Optional[float] = None,
        height_max: Optional[float] = None,
    ) -> "GeoTile":
        """Create GeoTile from numpy arrays.

        Args:
            height_map: 2D numpy array of heights in world units.
            footprint: Optional 2D binary mask (1=valid). If None, creates all-ones mask.
            metadata: Optional geospatial metadata.
            normalize: If True, normalize heights to [-1, 1] range.
            height_min: Minimum height for normalization (auto-computed if None).
            height_max: Maximum height for normalization (auto-computed if None).

        Returns:
            New GeoTile instance.
        """
        # Ensure 2D input
        if height_map.ndim == 3:
            height_map = height_map.squeeze()
        assert height_map.ndim == 2, f"Expected 2D height map, got {height_map.ndim}D"

        # Create metadata if not provided
        if metadata is None:
            metadata = GeoMetadata()

        # Handle normalization
        if normalize:
            # Find valid pixels for statistics
            if footprint is not None:
                valid_mask = footprint > 0.5
            else:
                nodata_val = metadata.nodata if metadata is not None else None
                if nodata_val is not None:
                    valid_mask = ~np.isnan(height_map) & (height_map != nodata_val)
                else:
                    valid_mask = ~np.isnan(height_map)

            valid_heights = height_map[valid_mask]
            if len(valid_heights) > 0:
                if height_min is None:
                    height_min = float(valid_heights.min())
                if height_max is None:
                    height_max = float(valid_heights.max())
            else:
                height_min = height_min if height_min is not None else 0.0
                height_max = height_max if height_max is not None else 1.0

            # Store normalization parameters
            if metadata is not None:
                metadata.height_min = height_min
                metadata.height_max = height_max

            # Normalize to [-1, 1]
            if height_max > height_min:
                height_normalized = (height_map - height_min) / (height_max - height_min)
                height_normalized = height_normalized * 2 - 1  # [0, 1] -> [-1, 1]
            else:
                height_normalized = np.zeros_like(height_map)

            height_tensor = torch.from_numpy(height_normalized.astype(np.float32))
        else:
            height_tensor = torch.from_numpy(height_map.astype(np.float32))

        # Add channel dimension (C, H, W)
        height_tensor = height_tensor.unsqueeze(0)

        # Create footprint tensor
        if footprint is None:
            footprint_arr: np.ndarray = np.ones_like(height_map)
        else:
            footprint_arr = footprint
        footprint_tensor = torch.from_numpy(footprint_arr.astype(np.float32)).unsqueeze(0)

        return cls(
            height_map=height_tensor,
            footprint=footprint_tensor,
            metadata=metadata,
        )

    def to_dict(self) -> dict:
        """Convert to dictionary for dataloader compatibility.

        Returns:
            Dictionary with keys compatible with RoofDataset output format.
        """
        return {
            "cond_image": self.height_map,
            "footprint": self.footprint,
            "geo_metadata": self.metadata.to_dict(),
            "original_points": self.original_points,
            "tile_index": self.tile_index,
        }

    def __repr__(self) -> str:
        return (
            f"GeoTile(shape={self.shape}, "
            f"has_footprint={self.footprint is not None}, "
            f"has_points={self.original_points is not None}, "
            f"tile_index={self.tile_index})"
        )
