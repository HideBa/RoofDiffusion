"""Geospatial data exporters for RoofDiffusion.

This module provides exporters for various geospatial output formats:
- GeoTIFFExporter: Export height maps as GeoTIFF raster files
- LASExporter: Export densified point clouds as LAS/LAZ files
- PNGExporter: Export as legacy PNG format (backward compatibility)
- MultiFormatExporter: Export to multiple formats at once
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Literal, Dict, List
import numpy as np
import torch

from data.geo.metadata import GeoMetadata


class BaseGeoExporter(ABC):
    """Abstract base class for geospatial data exporters."""

    @abstractmethod
    def export(
        self,
        height_map: torch.Tensor,
        metadata: GeoMetadata,
        output_path: Path,
    ) -> None:
        """Export height map to file.

        Args:
            height_map: Height map tensor (C, H, W) or (H, W), normalized to [-1, 1].
            metadata: Geospatial metadata for coordinate reference.
            output_path: Path for the output file.
        """
        pass

    @abstractmethod
    def get_extension(self) -> str:
        """Return the file extension for this format.

        Returns:
            File extension including the dot (e.g., '.tif').
        """
        pass

    def _denormalize_height(
        self, height_map: torch.Tensor, metadata: GeoMetadata
    ) -> np.ndarray:
        """Convert normalized height map to world coordinates.

        Args:
            height_map: Height tensor in [-1, 1] range.
            metadata: Metadata with height normalization parameters.

        Returns:
            Numpy array of heights in world units (meters).
        """
        height_np = height_map.squeeze().cpu().numpy()

        # Denormalize from [-1, 1]
        if metadata.height_min is not None and metadata.height_max is not None:
            height_np = (height_np + 1) / 2  # [-1, 1] -> [0, 1]
            height_np = (
                height_np * (metadata.height_max - metadata.height_min)
                + metadata.height_min
            )
        elif metadata.height_scale != 1.0 or metadata.height_offset != 0.0:
            height_np = height_np * metadata.height_scale + metadata.height_offset

        return height_np


class GeoTIFFExporter(BaseGeoExporter):
    """Export height maps as GeoTIFF raster files.

    Preserves CRS, transform, and supports various data types and compression.

    Example:
        >>> exporter = GeoTIFFExporter(dtype='float32', compress='lzw')
        >>> exporter.export(height_map, metadata, 'output.tif')
    """

    def __init__(
        self,
        dtype: Literal["float32", "float64", "uint16", "int16"] = "float32",
        nodata: Optional[float] = None,
        compress: Literal["none", "lzw", "deflate", "zstd"] = "lzw",
        tiled: bool = True,
        tile_size: int = 256,
    ):
        """Initialize the GeoTIFF exporter.

        Args:
            dtype: Output data type.
            nodata: No-data value to use. If None, uses -9999 for float, 0 for uint.
            compress: Compression algorithm.
            tiled: Use tiled storage (recommended for large files).
            tile_size: Tile size for tiled storage.
        """
        self.dtype = dtype
        self.nodata = nodata
        self.compress = compress
        self.tiled = tiled
        self.tile_size = tile_size

        # Set default nodata based on dtype
        if self.nodata is None:
            if dtype in ["float32", "float64"]:
                self.nodata = -9999.0
            elif dtype == "uint16":
                self.nodata = 0
            else:
                self.nodata = -32768

    def export(
        self,
        height_map: torch.Tensor,
        metadata: GeoMetadata,
        output_path: Path,
    ) -> None:
        """Export as GeoTIFF with full metadata.

        Args:
            height_map: Height map tensor.
            metadata: Geospatial metadata.
            output_path: Output file path.
        """
        try:
            import rasterio
            from rasterio.crs import CRS
            from rasterio.transform import Affine
        except ImportError:
            raise ImportError("rasterio is required for GeoTIFF export. Install with: pip install rasterio")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Denormalize height
        height_np = self._denormalize_height(height_map, metadata)

        # Handle NaN values
        height_np = np.where(np.isnan(height_np), self.nodata, height_np)

        # Convert to target dtype
        dtype_map = {
            "float32": np.float32,
            "float64": np.float64,
            "uint16": np.uint16,
            "int16": np.int16,
        }
        np_dtype = dtype_map[self.dtype]

        if self.dtype == "uint16":
            # Scale to uint16 range (0-65535)
            height_np = np.clip(height_np * 256, 0, 65535).astype(np_dtype)
        elif self.dtype == "int16":
            height_np = np.clip(height_np * 100, -32768, 32767).astype(np_dtype)
        else:
            height_np = height_np.astype(np_dtype)

        # Prepare rasterio profile
        height_2d = height_np.squeeze()
        profile = {
            "driver": "GTiff",
            "height": height_2d.shape[0],
            "width": height_2d.shape[1],
            "count": 1,
            "dtype": str(np_dtype),
            "nodata": self.nodata,
        }

        # Add compression
        if self.compress != "none":
            profile["compress"] = self.compress

        # Add tiling
        if self.tiled:
            profile["tiled"] = True
            profile["blockxsize"] = min(self.tile_size, height_2d.shape[1])
            profile["blockysize"] = min(self.tile_size, height_2d.shape[0])

        # Add CRS if available
        if metadata.crs:
            try:
                profile["crs"] = CRS.from_string(metadata.crs)
            except Exception:
                pass

        # Add transform if available
        if metadata.transform:
            profile["transform"] = Affine(*metadata.transform[:6])

        # Write file
        with rasterio.open(str(output_path), "w", **profile) as dst:
            dst.write(height_2d, 1)

    def get_extension(self) -> str:
        return ".tif"


class LASExporter(BaseGeoExporter):
    """Export densified point clouds as LAS/LAZ files.

    Converts raster back to point cloud with optional densification.

    Example:
        >>> exporter = LASExporter(format='laz', point_density=4.0)
        >>> exporter.export(height_map, metadata, 'output.laz', original_points=points)
    """

    def __init__(
        self,
        format: Literal["las", "laz"] = "laz",
        point_density: float = 1.0,
        densification_strategy: Literal["grid", "jitter", "regular"] = "grid",
        preserve_original_points: bool = True,
        classification: int = 6,  # Building
    ):
        """Initialize the LAS exporter.

        Args:
            format: Output format ('las' or 'laz' for compressed).
            point_density: Points per square meter for densification.
            densification_strategy: How to generate new points.
                'grid': Regular grid at cell centers.
                'jitter': Grid with random offset.
                'regular': Points at specified density.
            preserve_original_points: Include original input points if available.
            classification: LAS classification code for generated points.
        """
        self.format = format
        self.point_density = point_density
        self.densification_strategy = densification_strategy
        self.preserve_original_points = preserve_original_points
        self.classification = classification

    def export(
        self,
        height_map: torch.Tensor,
        metadata: GeoMetadata,
        output_path: Path,
        original_points: Optional[np.ndarray] = None,
    ) -> None:
        """Export as LAS/LAZ point cloud.

        Args:
            height_map: Completed height map tensor.
            metadata: Geospatial metadata with transform.
            output_path: Output file path.
            original_points: Original input points to preserve.
        """
        try:
            import laspy
        except ImportError:
            raise ImportError("laspy is required for LAS export. Install with: pip install laspy lazrs")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Denormalize height
        height_np = self._denormalize_height(height_map, metadata)

        # Generate densified points
        dense_points = self.densify(height_np, metadata, original_points)

        if len(dense_points) == 0:
            raise ValueError("No points generated for export")

        # Create LAS file
        header = laspy.LasHeader(point_format=0, version="1.2")

        # Set scale and offset for precision
        x_min, y_min = dense_points[:, 0].min(), dense_points[:, 1].min()
        header.offsets = [x_min, y_min, 0]
        header.scales = [0.001, 0.001, 0.001]  # mm precision

        # Create LAS data
        las = laspy.LasData(header)
        las.x = dense_points[:, 0]
        las.y = dense_points[:, 1]
        las.z = dense_points[:, 2]
        las.classification = np.full(len(dense_points), self.classification, dtype=np.uint8)

        # Write file
        if self.format == "laz":
            las.write(str(output_path))
        else:
            las.write(str(output_path))

    def densify(
        self,
        height_map: np.ndarray,
        metadata: GeoMetadata,
        original_points: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Generate dense point cloud from height map.

        Args:
            height_map: 2D height array in world coordinates.
            metadata: Contains transform for coordinate conversion.
            original_points: Original sparse points to include.

        Returns:
            Nx3 array of (x, y, z) points.
        """
        height_2d = height_map.squeeze()
        h, w = height_2d.shape

        all_points = []

        # Generate grid points from raster
        if metadata.transform is not None:
            a, b, c, d, e, f = metadata.transform[:6]

            if self.densification_strategy == "grid":
                # Regular grid at pixel centers
                rows, cols = np.indices((h, w))
                valid = ~np.isnan(height_2d)

                x = a * (cols + 0.5) + b * (rows + 0.5) + c
                y = d * (cols + 0.5) + e * (rows + 0.5) + f
                z = height_2d

                grid_points = np.column_stack([
                    x[valid],
                    y[valid],
                    z[valid]
                ])
                all_points.append(grid_points)

            elif self.densification_strategy == "jitter":
                # Grid with random jitter
                rows, cols = np.indices((h, w))
                valid = ~np.isnan(height_2d)

                jitter_x = np.random.uniform(-0.4, 0.4, (h, w))
                jitter_y = np.random.uniform(-0.4, 0.4, (h, w))

                x = a * (cols + 0.5 + jitter_x) + b * (rows + 0.5 + jitter_y) + c
                y = d * (cols + 0.5 + jitter_x) + e * (rows + 0.5 + jitter_y) + f
                z = height_2d

                grid_points = np.column_stack([
                    x[valid],
                    y[valid],
                    z[valid]
                ])
                all_points.append(grid_points)

            elif self.densification_strategy == "regular":
                # Points at specified density
                resolution = metadata.resolution or abs(a)
                points_per_cell = max(1, int(self.point_density * resolution * resolution))

                for row in range(h):
                    for col in range(w):
                        if np.isnan(height_2d[row, col]):
                            continue

                        for _ in range(points_per_cell):
                            # Random position within cell
                            rx = col + np.random.uniform(0, 1)
                            ry = row + np.random.uniform(0, 1)
                            x = a * rx + b * ry + c
                            y = d * rx + e * ry + f
                            z = height_2d[row, col]
                            all_points.append([[x, y, z]])

        # Include original points if requested
        if self.preserve_original_points and original_points is not None:
            all_points.append(original_points[:, :3])

        if all_points:
            return np.vstack(all_points)
        return np.empty((0, 3))

    def get_extension(self) -> str:
        return ".laz" if self.format == "laz" else ".las"


class PNGExporter(BaseGeoExporter):
    """Export as legacy PNG format (backward compatibility).

    Outputs uint16 PNG where height_m = pixel_value / 256.
    """

    def __init__(self, height_scale: float = 256.0):
        """Initialize the PNG exporter.

        Args:
            height_scale: Multiplier for converting meters to pixel values.
                         Default: 256.0 (uint16 format).
        """
        self.height_scale = height_scale

    def export(
        self,
        height_map: torch.Tensor,
        metadata: GeoMetadata,
        output_path: Path,
    ) -> None:
        """Export as uint16 PNG.

        Args:
            height_map: Height map tensor.
            metadata: Geospatial metadata (used for denormalization only).
            output_path: Output file path.
        """
        from PIL import Image

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Denormalize height
        height_np = self._denormalize_height(height_map, metadata)

        # Convert to uint16
        height_uint16 = np.clip(height_np * self.height_scale, 0, 65535).astype(np.uint16)

        # Save as PNG
        img = Image.fromarray(height_uint16, mode="I;16")
        img.save(str(output_path))

    def get_extension(self) -> str:
        return ".png"


class MultiFormatExporter:
    """Convenience class to export to multiple formats at once.

    Example:
        >>> exporters = [GeoTIFFExporter(), LASExporter()]
        >>> multi = MultiFormatExporter(exporters, output_dir='./output')
        >>> multi.export_all(height_map, metadata, 'building_001')
    """

    def __init__(
        self,
        exporters: List[BaseGeoExporter],
        output_dir: Path,
    ):
        """Initialize the multi-format exporter.

        Args:
            exporters: List of exporters to use.
            output_dir: Base directory for output files.
        """
        self.exporters = exporters
        self.output_dir = Path(output_dir)

    def export_all(
        self,
        height_map: torch.Tensor,
        metadata: GeoMetadata,
        base_name: str,
        original_points: Optional[np.ndarray] = None,
    ) -> Dict[str, Path]:
        """Export to all configured formats.

        Args:
            height_map: Height map tensor.
            metadata: Geospatial metadata.
            base_name: Base name for output files (without extension).
            original_points: Optional original points for LAS export.

        Returns:
            Dictionary mapping format name to output path.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = {}

        for exporter in self.exporters:
            ext = exporter.get_extension()
            output_path = self.output_dir / f"{base_name}{ext}"

            if isinstance(exporter, LASExporter) and original_points is not None:
                exporter.export(height_map, metadata, output_path, original_points)
            else:
                exporter.export(height_map, metadata, output_path)

            format_name = ext.lstrip(".")
            output_paths[format_name] = output_path

        return output_paths


def create_exporter(
    format: str,
    **kwargs,
) -> BaseGeoExporter:
    """Factory function to create appropriate exporter based on format.

    Args:
        format: Output format ('geotiff', 'tif', 'las', 'laz', 'png').
        **kwargs: Additional arguments passed to the exporter constructor.

    Returns:
        Appropriate exporter instance.

    Raises:
        ValueError: If format is not supported.
    """
    format = format.lower()

    if format in ["geotiff", "tif", "tiff"]:
        return GeoTIFFExporter(**kwargs)
    elif format == "las":
        return LASExporter(format="las", **kwargs)
    elif format == "laz":
        return LASExporter(format="laz", **kwargs)
    elif format == "png":
        return PNGExporter(**kwargs)
    else:
        raise ValueError(f"Unsupported export format: {format}")
