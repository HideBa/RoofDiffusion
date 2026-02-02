"""Geospatial metadata container for RoofDiffusion.

This module defines the GeoMetadata dataclass that stores coordinate reference
system and transform information through the processing pipeline.
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import json


@dataclass
class GeoMetadata:
    """Geospatial metadata container.

    Preserves coordinate reference system and transform information
    through the processing pipeline.

    Attributes:
        crs: Coordinate Reference System (EPSG code like "EPSG:32633" or WKT string)
        transform: Affine transform tuple (x_origin, x_res, x_skew, y_origin, y_skew, -y_res)
                   Compatible with rasterio/GDAL affine format
        bounds: Bounding box in world coordinates (min_x, min_y, max_x, max_y)
        resolution: Original resolution in world units (meters)
        height_offset: Height offset for normalization recovery
        height_scale: Height scale factor for normalization recovery
        point_attributes: Original LAS/LAZ attributes to preserve (classification, intensity, etc.)
        source_path: Source file path for traceability
    """

    # Coordinate Reference System (EPSG code or WKT)
    crs: Optional[str] = None

    # Affine transform: (x_origin, x_res, x_skew, y_origin, y_skew, -y_res)
    # or as 6-tuple compatible with rasterio: (a, b, c, d, e, f)
    # where: x' = a*col + b*row + c, y' = d*col + e*row + f
    transform: Optional[Tuple[float, ...]] = None

    # Bounding box: (min_x, min_y, max_x, max_y)
    bounds: Optional[Tuple[float, float, float, float]] = None

    # Original resolution in world units (meters)
    resolution: Optional[float] = None

    # Height offset for normalization recovery
    height_offset: float = 0.0
    height_scale: float = 1.0

    # Min/max height values for denormalization
    height_min: Optional[float] = None
    height_max: Optional[float] = None

    # Original LAS/LAZ attributes to preserve
    point_attributes: Optional[Dict[str, Any]] = None

    # Source file path for traceability
    source_path: Optional[str] = None

    # No-data value
    nodata: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize metadata to dictionary for JSON storage.

        Returns:
            Dictionary representation of metadata that can be serialized to JSON.
        """
        return {
            "crs": self.crs,
            "transform": list(self.transform) if self.transform else None,
            "bounds": list(self.bounds) if self.bounds else None,
            "resolution": self.resolution,
            "height_offset": self.height_offset,
            "height_scale": self.height_scale,
            "height_min": self.height_min,
            "height_max": self.height_max,
            "point_attributes": self.point_attributes,
            "source_path": self.source_path,
            "nodata": self.nodata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeoMetadata":
        """Deserialize metadata from dictionary.

        Args:
            data: Dictionary containing metadata fields.

        Returns:
            GeoMetadata instance reconstructed from the dictionary.
        """
        return cls(
            crs=data.get("crs"),
            transform=tuple(data["transform"]) if data.get("transform") else None,
            bounds=tuple(data["bounds"]) if data.get("bounds") else None,
            resolution=data.get("resolution"),
            height_offset=data.get("height_offset", 0.0),
            height_scale=data.get("height_scale", 1.0),
            height_min=data.get("height_min"),
            height_max=data.get("height_max"),
            point_attributes=data.get("point_attributes"),
            source_path=data.get("source_path"),
            nodata=data.get("nodata"),
        )

    def to_json(self) -> str:
        """Serialize metadata to JSON string.

        Returns:
            JSON string representation of metadata.
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "GeoMetadata":
        """Deserialize metadata from JSON string.

        Args:
            json_str: JSON string containing metadata.

        Returns:
            GeoMetadata instance reconstructed from the JSON.
        """
        return cls.from_dict(json.loads(json_str))

    def get_affine(self):
        """Get affine transform compatible with rasterio.

        Returns:
            rasterio.Affine object if rasterio is available, otherwise tuple.
        """
        if self.transform is None:
            return None

        try:
            from rasterio.transform import Affine

            return Affine(*self.transform[:6])
        except ImportError:
            return self.transform

    def pixel_to_world(self, col: float, row: float) -> Tuple[float, float]:
        """Convert pixel coordinates to world coordinates.

        Args:
            col: Column (x) pixel coordinate.
            row: Row (y) pixel coordinate.

        Returns:
            Tuple of (x, y) world coordinates.
        """
        if self.transform is None:
            raise ValueError("No transform available for coordinate conversion")

        a, b, c, d, e, f = self.transform[:6]
        x = a * col + b * row + c
        y = d * col + e * row + f
        return (x, y)

    def world_to_pixel(self, x: float, y: float) -> Tuple[float, float]:
        """Convert world coordinates to pixel coordinates.

        Args:
            x: X world coordinate.
            y: Y world coordinate.

        Returns:
            Tuple of (col, row) pixel coordinates.
        """
        if self.transform is None:
            raise ValueError("No transform available for coordinate conversion")

        a, b, c, d, e, f = self.transform[:6]
        # Inverse affine: solve for col, row
        # x = a*col + b*row + c
        # y = d*col + e*row + f
        det = a * e - b * d
        if abs(det) < 1e-10:
            raise ValueError("Transform is singular, cannot invert")

        col = (e * (x - c) - b * (y - f)) / det
        row = (a * (y - f) - d * (x - c)) / det
        return (col, row)

    def copy(self) -> "GeoMetadata":
        """Create a copy of this metadata.

        Returns:
            New GeoMetadata instance with same values.
        """
        return GeoMetadata.from_dict(self.to_dict())

    def with_new_transform(
        self, new_bounds: Tuple[float, float, float, float], new_resolution: float
    ) -> "GeoMetadata":
        """Create new metadata with updated transform for different bounds/resolution.

        Args:
            new_bounds: New bounding box (min_x, min_y, max_x, max_y).
            new_resolution: New resolution in world units.

        Returns:
            New GeoMetadata with updated transform.
        """
        min_x, min_y, max_x, max_y = new_bounds
        # Standard north-up transform: (res, 0, min_x, 0, -res, max_y)
        new_transform = (new_resolution, 0.0, min_x, 0.0, -new_resolution, max_y)

        return GeoMetadata(
            crs=self.crs,
            transform=new_transform,
            bounds=new_bounds,
            resolution=new_resolution,
            height_offset=self.height_offset,
            height_scale=self.height_scale,
            height_min=self.height_min,
            height_max=self.height_max,
            point_attributes=self.point_attributes,
            source_path=self.source_path,
            nodata=self.nodata,
        )

    def __repr__(self) -> str:
        return (
            f"GeoMetadata(crs={self.crs!r}, resolution={self.resolution}, "
            f"bounds={self.bounds})"
        )
