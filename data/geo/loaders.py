"""Geospatial data loaders for RoofDiffusion.

This module provides loaders for various geospatial data formats:
- LASLoader: For LAS/LAZ point cloud files
- GeoTIFFLoader: For GeoTIFF DSM/DTM raster files
- PNGLoader: For legacy PNG height maps (backward compatibility)
"""

from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple, List
from pathlib import Path
import numpy as np

from data.geo.tile import GeoTile
from data.geo.metadata import GeoMetadata
from data.geo.rasterizer import GeoRasterizer


class BaseGeoLoader(ABC):
    """Abstract base class for geospatial data loaders."""

    @abstractmethod
    def load(self, path: Path) -> GeoTile:
        """Load a single file and return as GeoTile.

        Args:
            path: Path to the input file.

        Returns:
            GeoTile containing height map, footprint, and metadata.
        """
        pass

    @abstractmethod
    def load_tiled(
        self,
        path: Path,
        tile_size: Tuple[int, int] = (128, 128),
        overlap: int = 0,
    ) -> Iterator[GeoTile]:
        """Load a large file as iterator of tiles.

        Args:
            path: Path to the input file.
            tile_size: Size of each tile (height, width).
            overlap: Overlap in pixels between adjacent tiles.

        Yields:
            GeoTile objects for each tile in the dataset.
        """
        pass

    @abstractmethod
    def supports_format(self, path: Path) -> bool:
        """Check if this loader supports the given file format.

        Args:
            path: Path to check.

        Returns:
            True if the loader can handle this file format.
        """
        pass


class LASLoader(BaseGeoLoader):
    """Loader for LAS/LAZ point cloud files.

    Uses laspy for LAS/LAZ I/O. Converts point clouds to raster
    height maps using GeoRasterizer.

    Example:
        >>> loader = LASLoader(resolution=0.5, classification_filter=[6])
        >>> tile = loader.load("building.laz")
    """

    def __init__(
        self,
        resolution: float = 0.5,
        classification_filter: Optional[List[int]] = None,
        height_attribute: str = "z",
        rasterize_method: str = "max",
        fill_method: str = "none",
        fill_radius: Optional[float] = None,
    ):
        """Initialize the LAS loader.

        Args:
            resolution: Output raster resolution in world units (meters).
            classification_filter: LAS classification codes to include.
                                   None = all points. [6] = buildings only.
                                   Common codes: 2=ground, 6=building, 3=low veg
            height_attribute: Point attribute to use as height ('z' or custom).
            rasterize_method: Aggregation method ('max', 'mean', 'min', 'median').
            fill_method: Gap filling method ('none', 'nearest', 'linear', 'cubic').
            fill_radius: Maximum distance for gap filling (in cells).
        """
        self.resolution = resolution
        self.classification_filter = classification_filter
        self.height_attribute = height_attribute
        self.rasterize_method = rasterize_method
        self.fill_method = fill_method
        self.fill_radius = fill_radius

        self.rasterizer = GeoRasterizer(
            resolution=resolution,
            method=rasterize_method,
            fill_method=fill_method,
            fill_radius=fill_radius,
        )

    def load(self, path: Path) -> GeoTile:
        """Load LAS/LAZ file and rasterize to height map.

        Args:
            path: Path to LAS/LAZ file.

        Returns:
            GeoTile with rasterized height map and metadata.
        """
        try:
            import laspy
        except ImportError:
            raise ImportError("laspy is required for LAS/LAZ support. Install with: pip install laspy lazrs")

        path = Path(path)

        # Read LAS/LAZ file
        las = laspy.read(str(path))

        # Extract coordinates
        x = np.array(las.x)
        y = np.array(las.y)

        if self.height_attribute == "z":
            z = np.array(las.z)
        else:
            z = np.array(getattr(las, self.height_attribute))

        # Apply classification filter if specified
        if self.classification_filter is not None:
            classification = np.array(las.classification)
            mask = np.isin(classification, self.classification_filter)
            x = x[mask]
            y = y[mask]
            z = z[mask]

        if len(x) == 0:
            raise ValueError(f"No points remaining after filtering in {path}")

        # Combine into points array
        points = np.column_stack([x, y, z])

        # Get CRS from LAS file
        crs = None
        try:
            if hasattr(las.header, "parse_crs") and las.header.parse_crs():
                crs = str(las.header.parse_crs())
            elif hasattr(las.header, "vlrs"):
                for vlr in las.header.vlrs:
                    if vlr.record_id == 2112:  # WKT CRS
                        crs = vlr.record_data.decode("utf-8").rstrip("\x00")
                        break
        except Exception:
            pass

        # Rasterize
        height_map, footprint, metadata = self.rasterizer.rasterize(points, crs=crs)

        # Store additional metadata
        metadata.source_path = str(path)

        # Collect point attributes for preservation
        point_attributes = {
            "total_points": len(las.points),
            "filtered_points": len(x),
        }
        if self.classification_filter is not None:
            point_attributes["classification_filter"] = self.classification_filter

        metadata.point_attributes = point_attributes

        # Create GeoTile
        tile = GeoTile.from_numpy(
            height_map=height_map,
            footprint=footprint,
            metadata=metadata,
            normalize=True,
        )

        # Store original points for potential export
        tile.original_points = points

        return tile

    def load_tiled(
        self,
        path: Path,
        tile_size: Tuple[int, int] = (128, 128),
        overlap: int = 0,
    ) -> Iterator[GeoTile]:
        """Load large point cloud as tiles using spatial indexing.

        Args:
            path: Path to LAS/LAZ file.
            tile_size: Size of each tile (height, width) in pixels.
            overlap: Overlap in pixels between adjacent tiles.

        Yields:
            GeoTile objects for each spatial tile.
        """
        try:
            import laspy
        except ImportError:
            raise ImportError("laspy is required for LAS/LAZ support. Install with: pip install laspy lazrs")

        path = Path(path)
        las = laspy.read(str(path))

        # Extract all coordinates
        x = np.array(las.x)
        y = np.array(las.y)
        z = np.array(las.z) if self.height_attribute == "z" else np.array(getattr(las, self.height_attribute))

        # Apply classification filter
        if self.classification_filter is not None:
            classification = np.array(las.classification)
            mask = np.isin(classification, self.classification_filter)
            x, y, z = x[mask], y[mask], z[mask]

        if len(x) == 0:
            return

        # Compute full extent
        min_x, max_x = x.min(), x.max()
        min_y, max_y = y.min(), y.max()

        # Compute tile world size
        tile_h, tile_w = tile_size
        tile_world_h = tile_h * self.resolution
        tile_world_w = tile_w * self.resolution
        overlap_world = overlap * self.resolution

        # Compute number of tiles
        n_cols = max(1, int(np.ceil((max_x - min_x - overlap_world) / (tile_world_w - overlap_world))))
        n_rows = max(1, int(np.ceil((max_y - min_y - overlap_world) / (tile_world_h - overlap_world))))

        # Get CRS
        crs = None
        try:
            if hasattr(las.header, "parse_crs") and las.header.parse_crs():
                crs = str(las.header.parse_crs())
        except Exception:
            pass

        # Generate tiles
        for row in range(n_rows):
            for col in range(n_cols):
                # Compute tile bounds
                tile_min_x = min_x + col * (tile_world_w - overlap_world)
                tile_max_x = min(tile_min_x + tile_world_w, max_x)
                tile_max_y = max_y - row * (tile_world_h - overlap_world)
                tile_min_y = max(tile_max_y - tile_world_h, min_y)

                # Find points in this tile
                tile_mask = (
                    (x >= tile_min_x) & (x < tile_max_x) &
                    (y > tile_min_y) & (y <= tile_max_y)
                )

                if not np.any(tile_mask):
                    continue

                tile_x = x[tile_mask]
                tile_y = y[tile_mask]
                tile_z = z[tile_mask]
                tile_points = np.column_stack([tile_x, tile_y, tile_z])

                # Rasterize tile
                tile_bounds = (tile_min_x, tile_min_y, tile_max_x, tile_max_y)
                height_map, footprint, metadata = self.rasterizer.rasterize(
                    tile_points, bounds=tile_bounds, crs=crs
                )

                # Pad to exact tile size if needed
                if height_map.shape != tile_size:
                    padded_height = np.full(tile_size, np.nan, dtype=np.float32)
                    padded_footprint = np.zeros(tile_size, dtype=np.float32)
                    h, w = min(height_map.shape[0], tile_h), min(height_map.shape[1], tile_w)
                    padded_height[:h, :w] = height_map[:h, :w]
                    padded_footprint[:h, :w] = footprint[:h, :w]
                    height_map = padded_height
                    footprint = padded_footprint

                metadata.source_path = str(path)

                tile = GeoTile.from_numpy(
                    height_map=height_map,
                    footprint=footprint,
                    metadata=metadata,
                    normalize=True,
                )
                tile.original_points = tile_points
                tile.tile_index = (row, col)

                yield tile

    def supports_format(self, path: Path) -> bool:
        """Check if this loader supports LAS/LAZ files."""
        return Path(path).suffix.lower() in [".las", ".laz"]


class GeoTIFFLoader(BaseGeoLoader):
    """Loader for GeoTIFF DSM/DTM raster files.

    Uses rasterio for GeoTIFF I/O.

    Example:
        >>> loader = GeoTIFFLoader(nodata_value=-9999)
        >>> tile = loader.load("dsm.tif")
    """

    def __init__(
        self,
        target_resolution: Optional[float] = None,
        nodata_value: Optional[float] = None,
        band: int = 1,
    ):
        """Initialize the GeoTIFF loader.

        Args:
            target_resolution: Resample to this resolution. None = native resolution.
            nodata_value: Value to treat as no-data. None = from file metadata.
            band: Which band to read (1-indexed).
        """
        self.target_resolution = target_resolution
        self.nodata_value = nodata_value
        self.band = band

    def load(self, path: Path) -> GeoTile:
        """Load GeoTIFF and convert to GeoTile.

        Args:
            path: Path to GeoTIFF file.

        Returns:
            GeoTile with height map and geospatial metadata.
        """
        try:
            import rasterio
            from rasterio.enums import Resampling
        except ImportError:
            raise ImportError("rasterio is required for GeoTIFF support. Install with: pip install rasterio")

        path = Path(path)

        with rasterio.open(str(path)) as src:
            # Read data
            data = src.read(self.band).astype(np.float32)

            # Get nodata value
            nodata = self.nodata_value if self.nodata_value is not None else src.nodata

            # Create validity mask
            if nodata is not None:
                footprint = (data != nodata).astype(np.float32)
                data = np.where(data == nodata, np.nan, data)
            else:
                footprint = (~np.isnan(data)).astype(np.float32)

            # Get geospatial info
            transform = tuple(src.transform)[:6]
            crs = str(src.crs) if src.crs else None
            bounds = tuple(src.bounds)
            resolution = src.res[0]  # Assumes square pixels

            # Resample if requested
            if self.target_resolution is not None and self.target_resolution != resolution:
                scale = resolution / self.target_resolution
                new_height = int(round(data.shape[0] * scale))
                new_width = int(round(data.shape[1] * scale))

                resampled_data = np.empty((new_height, new_width), dtype=np.float32)
                resampled_footprint = np.empty((new_height, new_width), dtype=np.float32)

                # Use bilinear for height, nearest for footprint
                from scipy import ndimage
                resampled_data = ndimage.zoom(
                    np.nan_to_num(data, nan=0), scale, order=1
                )
                resampled_footprint = ndimage.zoom(footprint, scale, order=0)

                data = resampled_data
                footprint = resampled_footprint
                resolution = self.target_resolution

                # Update transform for new resolution
                transform = (
                    self.target_resolution,
                    transform[1],
                    transform[2],
                    transform[3],
                    -self.target_resolution,
                    transform[5],
                )

        # Create metadata
        metadata = GeoMetadata(
            crs=crs,
            transform=transform,
            bounds=bounds,
            resolution=resolution,
            nodata=nodata,
            source_path=str(path),
        )

        # Create GeoTile
        tile = GeoTile.from_numpy(
            height_map=data,
            footprint=footprint,
            metadata=metadata,
            normalize=True,
        )

        return tile

    def load_tiled(
        self,
        path: Path,
        tile_size: Tuple[int, int] = (128, 128),
        overlap: int = 0,
    ) -> Iterator[GeoTile]:
        """Load large raster using windowed reading.

        Args:
            path: Path to GeoTIFF file.
            tile_size: Size of each tile (height, width).
            overlap: Overlap in pixels between adjacent tiles.

        Yields:
            GeoTile objects for each window.
        """
        try:
            import rasterio
            from rasterio.windows import Window
        except ImportError:
            raise ImportError("rasterio is required for GeoTIFF support. Install with: pip install rasterio")

        path = Path(path)
        tile_h, tile_w = tile_size
        step_h = tile_h - overlap
        step_w = tile_w - overlap

        with rasterio.open(str(path)) as src:
            height_px = src.height
            width_px = src.width

            nodata = self.nodata_value if self.nodata_value is not None else src.nodata
            crs = str(src.crs) if src.crs else None
            resolution = src.res[0]

            n_rows = max(1, int(np.ceil((height_px - overlap) / step_h)))
            n_cols = max(1, int(np.ceil((width_px - overlap) / step_w)))

            for row in range(n_rows):
                for col in range(n_cols):
                    # Compute window
                    row_off = row * step_h
                    col_off = col * step_w

                    # Adjust for edge tiles
                    win_h = min(tile_h, height_px - row_off)
                    win_w = min(tile_w, width_px - col_off)

                    window = Window(col_off, row_off, win_w, win_h)

                    # Read window data
                    data = src.read(self.band, window=window).astype(np.float32)

                    # Handle nodata
                    if nodata is not None:
                        footprint = (data != nodata).astype(np.float32)
                        data = np.where(data == nodata, np.nan, data)
                    else:
                        footprint = (~np.isnan(data)).astype(np.float32)

                    # Pad to exact tile size if edge tile
                    if data.shape != tile_size:
                        padded_data = np.full(tile_size, np.nan, dtype=np.float32)
                        padded_footprint = np.zeros(tile_size, dtype=np.float32)
                        padded_data[:data.shape[0], :data.shape[1]] = data
                        padded_footprint[:footprint.shape[0], :footprint.shape[1]] = footprint
                        data = padded_data
                        footprint = padded_footprint

                    # Compute window transform
                    window_transform = src.window_transform(window)
                    tile_transform = tuple(window_transform)[:6]

                    # Compute window bounds
                    tile_bounds = rasterio.windows.bounds(window, src.transform)

                    metadata = GeoMetadata(
                        crs=crs,
                        transform=tile_transform,
                        bounds=tile_bounds,
                        resolution=resolution,
                        nodata=nodata,
                        source_path=str(path),
                    )

                    tile = GeoTile.from_numpy(
                        height_map=data,
                        footprint=footprint,
                        metadata=metadata,
                        normalize=True,
                    )
                    tile.tile_index = (row, col)

                    yield tile

    def supports_format(self, path: Path) -> bool:
        """Check if this loader supports GeoTIFF files."""
        return Path(path).suffix.lower() in [".tif", ".tiff", ".geotiff"]


class PNGLoader(BaseGeoLoader):
    """Loader for legacy PNG height maps (backward compatibility).

    Wraps existing roof_pil_loader functionality for PNG height maps
    where height(m) = pixel_value / 256.

    Example:
        >>> loader = PNGLoader(height_scale=256.0)
        >>> tile = loader.load("height_map.png")
    """

    def __init__(self, height_scale: float = 256.0):
        """Initialize the PNG loader.

        Args:
            height_scale: Divisor to convert pixel values to meters.
                         Default: 256.0 (uint16 format where height_m = pixel/256)
        """
        self.height_scale = height_scale

    def load(self, path: Path) -> GeoTile:
        """Load PNG height map (no geospatial metadata).

        Args:
            path: Path to PNG file.

        Returns:
            GeoTile with height map (without geospatial coordinates).
        """
        from PIL import Image

        path = Path(path)
        img = Image.open(str(path))

        # Convert to numpy array
        data = np.array(img).astype(np.float32)

        # Handle different bit depths
        if data.max() > 255:  # uint16
            height_map = data / self.height_scale
        else:  # uint8
            height_map = data / (self.height_scale / 256)

        # Create all-ones footprint (assume all pixels valid)
        footprint = np.ones_like(height_map)

        # Create basic metadata without geospatial info
        metadata = GeoMetadata(
            crs=None,
            transform=None,
            bounds=None,
            resolution=None,
            source_path=str(path),
        )

        tile = GeoTile.from_numpy(
            height_map=height_map,
            footprint=footprint,
            metadata=metadata,
            normalize=True,
        )

        return tile

    def load_tiled(
        self,
        path: Path,
        tile_size: Tuple[int, int] = (128, 128),
        overlap: int = 0,
    ) -> Iterator[GeoTile]:
        """Load and tile PNG image.

        Args:
            path: Path to PNG file.
            tile_size: Size of each tile (height, width).
            overlap: Overlap in pixels between adjacent tiles.

        Yields:
            GeoTile objects for each tile.
        """
        from PIL import Image

        path = Path(path)
        img = Image.open(str(path))
        data = np.array(img).astype(np.float32)

        if data.max() > 255:
            height_map = data / self.height_scale
        else:
            height_map = data / (self.height_scale / 256)

        tile_h, tile_w = tile_size
        step_h = tile_h - overlap
        step_w = tile_w - overlap

        img_h, img_w = height_map.shape[:2]
        n_rows = max(1, int(np.ceil((img_h - overlap) / step_h)))
        n_cols = max(1, int(np.ceil((img_w - overlap) / step_w)))

        for row in range(n_rows):
            for col in range(n_cols):
                row_start = row * step_h
                col_start = col * step_w
                row_end = min(row_start + tile_h, img_h)
                col_end = min(col_start + tile_w, img_w)

                tile_data = height_map[row_start:row_end, col_start:col_end]

                # Pad if needed
                if tile_data.shape != tile_size:
                    padded = np.zeros(tile_size, dtype=np.float32)
                    padded[:tile_data.shape[0], :tile_data.shape[1]] = tile_data
                    tile_data = padded

                footprint = np.ones_like(tile_data)

                metadata = GeoMetadata(
                    source_path=str(path),
                )

                tile = GeoTile.from_numpy(
                    height_map=tile_data,
                    footprint=footprint,
                    metadata=metadata,
                    normalize=True,
                )
                tile.tile_index = (row, col)

                yield tile

    def supports_format(self, path: Path) -> bool:
        """Check if this loader supports PNG/JPEG files."""
        return Path(path).suffix.lower() in [".png", ".jpg", ".jpeg"]


def create_loader(path: Path, **kwargs) -> BaseGeoLoader:
    """Factory function to create appropriate loader based on file extension.

    Args:
        path: Path to the file to load.
        **kwargs: Additional arguments passed to the loader constructor.

    Returns:
        Appropriate loader instance for the file type.

    Raises:
        ValueError: If no loader supports the file format.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in [".las", ".laz"]:
        return LASLoader(**kwargs)
    elif suffix in [".tif", ".tiff", ".geotiff"]:
        return GeoTIFFLoader(**kwargs)
    elif suffix in [".png", ".jpg", ".jpeg"]:
        return PNGLoader(**kwargs)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
