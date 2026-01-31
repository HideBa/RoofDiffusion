# RoofDiffusion Geospatial I/O Extension - Design Document

## 1. Overview

This document describes the architectural design for extending RoofDiffusion to support:
- **Input formats**: LAS/LAZ point clouds, GeoTIFF DSM/DTM rasters
- **Output formats**: Densified LAS/LAZ point clouds, GeoTIFF rasters

The goal is to enable real-world LiDAR processing workflows while preserving geospatial metadata (CRS, transforms, point attributes).

---

## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RoofDiffusion Pipeline                         │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│   LAS/LAZ     │           │   GeoTIFF     │           │   PNG/TIFF    │
│  Point Cloud  │           │   DSM/DTM     │           │  Height Map   │
└───────────────┘           └───────────────┘           └───────────────┘
        │                             │                             │
        ▼                             ▼                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GeoDataLoader (NEW)                               │
│  - Unified interface for all input formats                                  │
│  - Preserves geospatial metadata (CRS, transform, bounds)                   │
│  - Handles tiling for large datasets                                        │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GeoRasterizer (NEW)                                 │
│  - Converts point clouds to raster grids                                    │
│  - Configurable resolution and interpolation                                │
│  - Generates footprint masks from point density                             │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RoofDataset (EXISTING - Extended)                        │
│  - Accepts GeoTile objects instead of raw paths                             │
│  - Corruption synthesis unchanged                                           │
│  - Passes through geospatial metadata                                       │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      Palette Model (EXISTING)                               │
│  - Diffusion-based roof completion                                          │
│  - No changes to core model                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        GeoExporter (NEW)                                    │
│  - Exports to GeoTIFF with preserved CRS/transform                          │
│  - Exports to LAS/LAZ with densified points                                 │
│  - Handles coordinate system transformations                                │
└─────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│   LAS/LAZ     │           │   GeoTIFF     │           │   PNG/TIFF    │
│  (Densified)  │           │   (Output)    │           │  (Legacy)     │
└───────────────┘           └───────────────┘           └───────────────┘
```

---

## 3. Core Data Structures

### 3.1 GeoMetadata

Stores geospatial reference information that must be preserved through the pipeline.

```python
# File: data/geo/metadata.py

from dataclasses import dataclass, field
from typing import Optional, Tuple, List
import numpy as np

@dataclass
class GeoMetadata:
    """Geospatial metadata container.

    Preserves coordinate reference system and transform information
    through the processing pipeline.
    """
    # Coordinate Reference System (EPSG code or WKT)
    crs: Optional[str] = None

    # Affine transform: (x_origin, x_res, 0, y_origin, 0, -y_res)
    transform: Optional[Tuple[float, ...]] = None

    # Bounding box: (min_x, min_y, max_x, max_y)
    bounds: Optional[Tuple[float, float, float, float]] = None

    # Original resolution in world units (meters)
    resolution: Optional[float] = None

    # Height offset for normalization recovery
    height_offset: float = 0.0
    height_scale: float = 1.0

    # Original LAS/LAZ attributes to preserve
    point_attributes: Optional[dict] = None

    # Source file path for traceability
    source_path: Optional[str] = None

    def to_dict(self) -> dict:
        """Serialize metadata to dictionary for JSON storage."""
        ...

    @classmethod
    def from_dict(cls, data: dict) -> 'GeoMetadata':
        """Deserialize metadata from dictionary."""
        ...
```

### 3.2 GeoTile

Represents a single processing unit with its data and metadata.

```python
# File: data/geo/tile.py

from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch

@dataclass
class GeoTile:
    """A single tile with height data and geospatial metadata.

    This is the primary data container passed through the pipeline.
    """
    # Height map as tensor (C, H, W) - normalized to [-1, 1]
    height_map: torch.Tensor

    # Footprint/validity mask (1, H, W) - 1=valid, 0=invalid
    footprint: torch.Tensor

    # Geospatial metadata
    metadata: GeoMetadata

    # Original point cloud (if input was LAS/LAZ)
    # Stored for point-wise attribute preservation during export
    original_points: Optional[np.ndarray] = None

    # Tile index for reassembly of large datasets
    tile_index: Optional[Tuple[int, int]] = None

    @property
    def shape(self) -> Tuple[int, int]:
        """Return (height, width) of the tile."""
        return self.height_map.shape[-2:]
```

---

## 4. Input Pipeline Components

### 4.1 GeoDataLoader

Abstract base class and concrete implementations for different input formats.

```python
# File: data/geo/loaders.py

from abc import ABC, abstractmethod
from typing import Iterator, Optional, Tuple, List
from pathlib import Path
import numpy as np

class BaseGeoLoader(ABC):
    """Abstract base class for geospatial data loaders."""

    @abstractmethod
    def load(self, path: Path) -> GeoTile:
        """Load a single file and return as GeoTile."""
        ...

    @abstractmethod
    def load_tiled(
        self,
        path: Path,
        tile_size: Tuple[int, int] = (128, 128),
        overlap: int = 0
    ) -> Iterator[GeoTile]:
        """Load a large file as iterator of tiles."""
        ...

    @abstractmethod
    def supports_format(self, path: Path) -> bool:
        """Check if this loader supports the given file format."""
        ...


class LASLoader(BaseGeoLoader):
    """Loader for LAS/LAZ point cloud files.

    Uses laspy for LAS/LAZ I/O.
    """

    def __init__(
        self,
        resolution: float = 0.5,
        classification_filter: Optional[List[int]] = None,
        height_attribute: str = 'z',
        interpolation: str = 'nearest'
    ):
        """
        Args:
            resolution: Output raster resolution in world units (meters).
            classification_filter: LAS classification codes to include.
                                   None = all points. [6] = buildings only.
            height_attribute: Point attribute to use as height ('z' or custom).
            interpolation: Method for rasterization ('nearest', 'linear', 'cubic').
        """
        ...

    def load(self, path: Path) -> GeoTile:
        """Load LAS/LAZ file and rasterize to height map."""
        ...

    def load_tiled(
        self,
        path: Path,
        tile_size: Tuple[int, int] = (128, 128),
        overlap: int = 0
    ) -> Iterator[GeoTile]:
        """Load large point cloud as tiles using spatial indexing."""
        ...

    def supports_format(self, path: Path) -> bool:
        return path.suffix.lower() in ['.las', '.laz']


class GeoTIFFLoader(BaseGeoLoader):
    """Loader for GeoTIFF DSM/DTM raster files.

    Uses rasterio for GeoTIFF I/O.
    """

    def __init__(
        self,
        target_resolution: Optional[float] = None,
        nodata_value: Optional[float] = None,
        band: int = 1
    ):
        """
        Args:
            target_resolution: Resample to this resolution. None = native.
            nodata_value: Value to treat as no-data. None = from file metadata.
            band: Which band to read (1-indexed).
        """
        ...

    def load(self, path: Path) -> GeoTile:
        """Load GeoTIFF and convert to GeoTile."""
        ...

    def load_tiled(
        self,
        path: Path,
        tile_size: Tuple[int, int] = (128, 128),
        overlap: int = 0
    ) -> Iterator[GeoTile]:
        """Load large raster using windowed reading."""
        ...

    def supports_format(self, path: Path) -> bool:
        return path.suffix.lower() in ['.tif', '.tiff', '.geotiff']


class PNGLoader(BaseGeoLoader):
    """Loader for legacy PNG height maps (backward compatibility).

    Wraps existing roof_pil_loader functionality.
    """

    def __init__(self, height_scale: float = 256.0):
        """
        Args:
            height_scale: Divisor to convert pixel values to meters.
        """
        ...

    def load(self, path: Path) -> GeoTile:
        """Load PNG height map (no geospatial metadata)."""
        ...

    def load_tiled(
        self,
        path: Path,
        tile_size: Tuple[int, int] = (128, 128),
        overlap: int = 0
    ) -> Iterator[GeoTile]:
        """Load and tile PNG image."""
        ...

    def supports_format(self, path: Path) -> bool:
        return path.suffix.lower() in ['.png', '.jpg', '.jpeg']


def create_loader(path: Path, **kwargs) -> BaseGeoLoader:
    """Factory function to create appropriate loader based on file extension."""
    ...
```

### 4.2 GeoRasterizer

Converts point clouds to raster grids with configurable options.

```python
# File: data/geo/rasterizer.py

from typing import Tuple, Optional, Literal
import numpy as np

class GeoRasterizer:
    """Converts point clouds to raster height maps.

    Supports multiple rasterization strategies and handles
    sparse point clouds gracefully.
    """

    def __init__(
        self,
        resolution: float = 0.5,
        method: Literal['max', 'mean', 'median', 'min', 'count'] = 'max',
        fill_method: Literal['none', 'nearest', 'linear', 'cubic'] = 'none',
        fill_radius: Optional[float] = None
    ):
        """
        Args:
            resolution: Grid cell size in world units.
            method: Aggregation method when multiple points fall in same cell.
                   'max' is typical for DSM, 'mean' for DTM.
            fill_method: Interpolation to fill gaps in sparse data.
            fill_radius: Maximum distance for gap filling (in cells).
        """
        ...

    def rasterize(
        self,
        points: np.ndarray,
        bounds: Optional[Tuple[float, float, float, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray, GeoMetadata]:
        """Convert point cloud to raster.

        Args:
            points: Nx3+ array of (x, y, z, ...) coordinates.
            bounds: Optional (min_x, min_y, max_x, max_y) bounds.
                   If None, derived from point extent.

        Returns:
            height_map: 2D array of heights.
            footprint: 2D binary mask of valid cells.
            metadata: GeoMetadata with transform info.
        """
        ...

    def compute_footprint(
        self,
        points: np.ndarray,
        resolution: float,
        density_threshold: int = 1
    ) -> np.ndarray:
        """Generate footprint mask from point density.

        Args:
            points: Nx2+ array of (x, y, ...) coordinates.
            resolution: Grid cell size.
            density_threshold: Minimum points per cell to be considered valid.

        Returns:
            Binary mask where 1 = sufficient points, 0 = sparse/empty.
        """
        ...
```

---

## 5. Output Pipeline Components

### 5.1 GeoExporter

Exports processed results to various geospatial formats.

```python
# File: data/geo/exporters.py

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Literal
import numpy as np
import torch

class BaseGeoExporter(ABC):
    """Abstract base class for geospatial data exporters."""

    @abstractmethod
    def export(
        self,
        height_map: torch.Tensor,
        metadata: GeoMetadata,
        output_path: Path
    ) -> None:
        """Export height map to file."""
        ...

    @abstractmethod
    def get_extension(self) -> str:
        """Return the file extension for this format."""
        ...


class GeoTIFFExporter(BaseGeoExporter):
    """Export height maps as GeoTIFF raster files.

    Preserves CRS, transform, and supports various data types.
    """

    def __init__(
        self,
        dtype: Literal['float32', 'float64', 'uint16', 'int16'] = 'float32',
        nodata: Optional[float] = None,
        compress: Literal['none', 'lzw', 'deflate', 'zstd'] = 'lzw',
        tiled: bool = True,
        tile_size: int = 256
    ):
        """
        Args:
            dtype: Output data type.
            nodata: No-data value to use.
            compress: Compression algorithm.
            tiled: Use tiled storage (recommended for large files).
            tile_size: Tile size for tiled storage.
        """
        ...

    def export(
        self,
        height_map: torch.Tensor,
        metadata: GeoMetadata,
        output_path: Path
    ) -> None:
        """Export as GeoTIFF with full metadata."""
        ...

    def get_extension(self) -> str:
        return '.tif'


class LASExporter(BaseGeoExporter):
    """Export densified point clouds as LAS/LAZ files.

    Converts raster back to point cloud with optional densification.
    """

    def __init__(
        self,
        format: Literal['las', 'laz'] = 'laz',
        point_density: float = 1.0,
        densification_strategy: Literal['grid', 'jitter', 'regular'] = 'grid',
        preserve_original_points: bool = True,
        classification: int = 6  # Building
    ):
        """
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
        ...

    def export(
        self,
        height_map: torch.Tensor,
        metadata: GeoMetadata,
        output_path: Path,
        original_points: Optional[np.ndarray] = None
    ) -> None:
        """Export as LAS/LAZ point cloud.

        Args:
            height_map: Completed height map tensor.
            metadata: Geospatial metadata with transform.
            output_path: Output file path.
            original_points: Original input points to preserve.
        """
        ...

    def densify(
        self,
        height_map: np.ndarray,
        metadata: GeoMetadata,
        original_points: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Generate dense point cloud from height map.

        Args:
            height_map: 2D height array in world coordinates.
            metadata: Contains transform for coordinate conversion.
            original_points: Original sparse points to include.

        Returns:
            Nx3 array of (x, y, z) points.
        """
        ...

    def get_extension(self) -> str:
        return '.laz' if self.format == 'laz' else '.las'


class PNGExporter(BaseGeoExporter):
    """Export as legacy PNG format (backward compatibility)."""

    def __init__(self, height_scale: float = 256.0):
        ...

    def export(
        self,
        height_map: torch.Tensor,
        metadata: GeoMetadata,
        output_path: Path
    ) -> None:
        """Export as uint16 PNG (height * 256)."""
        ...

    def get_extension(self) -> str:
        return '.png'


class MultiFormatExporter:
    """Convenience class to export to multiple formats at once."""

    def __init__(
        self,
        exporters: list[BaseGeoExporter],
        output_dir: Path
    ):
        """
        Args:
            exporters: List of exporters to use.
            output_dir: Base directory for output files.
        """
        ...

    def export_all(
        self,
        height_map: torch.Tensor,
        metadata: GeoMetadata,
        base_name: str
    ) -> dict[str, Path]:
        """Export to all configured formats.

        Returns:
            Dictionary mapping format name to output path.
        """
        ...
```

---

## 6. Extended Dataset Classes

### 6.1 GeoRoofDataset

Extended dataset class that works with GeoTile objects.

```python
# File: data/geo_dataset.py

from typing import List, Optional, Union, Iterator
from pathlib import Path
import torch.utils.data as data

from data.geo.tile import GeoTile
from data.geo.loaders import BaseGeoLoader, create_loader
from data.geo.metadata import GeoMetadata

class GeoRoofDataset(data.Dataset):
    """Extended RoofDataset supporting geospatial formats.

    Wraps existing RoofDataset functionality while adding support
    for LAS/LAZ and GeoTIFF inputs with metadata preservation.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        footprint_root: Optional[Union[str, Path]] = None,
        loader_config: Optional[dict] = None,
        tile_size: tuple[int, int] = (128, 128),
        tile_overlap: int = 0,
        # Existing RoofDataset parameters
        noise_config: dict = {},
        mask_config: dict = {},
        data_aug: dict = {},
        data_len: int = -1,
        use_footprint: bool = True,
        no_height: bool = False,
    ):
        """
        Args:
            data_root: Path to data files or .flist.
            footprint_root: Path to footprint files (optional for LAS/GeoTIFF).
            loader_config: Configuration for geospatial loaders.
            tile_size: Size of tiles for large datasets.
            tile_overlap: Overlap between adjacent tiles.
            ... (existing RoofDataset parameters)
        """
        ...

    def _detect_format(self, path: Path) -> str:
        """Detect input format from file extension."""
        ...

    def _load_tile(self, index: int) -> GeoTile:
        """Load a single tile with format detection."""
        ...

    def __getitem__(self, index: int) -> dict:
        """Get item with geospatial metadata preserved.

        Returns dict with standard keys plus:
            'geo_metadata': Serialized GeoMetadata
            'original_points': Optional sparse input points
            'tile_index': (row, col) for tile reassembly
        """
        ...

    def __len__(self) -> int:
        ...


class GeoInferenceDataset(data.Dataset):
    """Inference-only dataset for processing real-world data.

    Simplified dataset for inference without corruption synthesis.
    Handles large files through tiling with overlap for seamless stitching.
    """

    def __init__(
        self,
        input_path: Union[str, Path],
        footprint_path: Optional[Union[str, Path]] = None,
        tile_size: tuple[int, int] = (128, 128),
        tile_overlap: int = 16,
        loader_config: Optional[dict] = None
    ):
        """
        Args:
            input_path: Path to input file (LAS/LAZ/GeoTIFF).
            footprint_path: Optional path to building footprints.
            tile_size: Processing tile size.
            tile_overlap: Overlap for seamless stitching.
            loader_config: Loader-specific configuration.
        """
        ...

    def __getitem__(self, index: int) -> dict:
        """Get tile for inference."""
        ...

    def __len__(self) -> int:
        """Total number of tiles."""
        ...

    def get_tile_grid(self) -> tuple[int, int]:
        """Return (n_rows, n_cols) of tile grid."""
        ...
```

---

## 7. Result Assembly and Stitching

### 7.1 TileAssembler

Reassembles tiled outputs into complete results.

```python
# File: data/geo/assembler.py

from typing import List, Tuple, Optional
from pathlib import Path
import numpy as np
import torch

from data.geo.tile import GeoTile
from data.geo.metadata import GeoMetadata
from data.geo.exporters import BaseGeoExporter

class TileAssembler:
    """Reassembles tiled inference results into complete outputs.

    Handles overlap blending for seamless stitching.
    """

    def __init__(
        self,
        tile_size: Tuple[int, int],
        overlap: int,
        blend_mode: str = 'linear'  # 'linear', 'cosine', 'none'
    ):
        """
        Args:
            tile_size: Size of each tile (H, W).
            overlap: Overlap in pixels between adjacent tiles.
            blend_mode: Blending strategy for overlapping regions.
        """
        ...

    def add_tile(
        self,
        tile: GeoTile,
        tile_index: Tuple[int, int],
        result: torch.Tensor
    ) -> None:
        """Add a processed tile to the assembly buffer."""
        ...

    def assemble(self) -> Tuple[np.ndarray, GeoMetadata]:
        """Assemble all tiles into final output.

        Returns:
            height_map: Complete assembled height map.
            metadata: Merged metadata for full extent.
        """
        ...

    def export(
        self,
        exporter: BaseGeoExporter,
        output_path: Path
    ) -> None:
        """Assemble and export in one step."""
        ...


class StreamingAssembler:
    """Memory-efficient assembler for very large datasets.

    Writes tiles directly to disk, avoiding full dataset in memory.
    """

    def __init__(
        self,
        output_path: Path,
        full_shape: Tuple[int, int],
        tile_size: Tuple[int, int],
        overlap: int,
        metadata: GeoMetadata,
        dtype: np.dtype = np.float32
    ):
        """
        Args:
            output_path: Path to output file.
            full_shape: Total (H, W) of assembled output.
            tile_size: Size of each tile.
            overlap: Overlap between tiles.
            metadata: Metadata for full output.
            dtype: Data type for output array.
        """
        ...

    def write_tile(
        self,
        tile_result: np.ndarray,
        tile_index: Tuple[int, int]
    ) -> None:
        """Write a single tile to the output file."""
        ...

    def finalize(self) -> None:
        """Complete the assembly and close file handles."""
        ...
```

---

## 8. Extended Palette Model

### 8.1 GeoAwarePalette

Extended model class that preserves geospatial metadata.

```python
# File: models/geo_model.py

from typing import Optional, Dict, Any
from pathlib import Path

from models.model import Palette
from data.geo.tile import GeoTile
from data.geo.metadata import GeoMetadata
from data.geo.exporters import BaseGeoExporter, GeoTIFFExporter, LASExporter
from data.geo.assembler import TileAssembler

class GeoAwarePalette(Palette):
    """Extended Palette model with geospatial export capabilities.

    Inherits all training/inference functionality from Palette,
    adds geospatial metadata handling and multi-format export.
    """

    def __init__(
        self,
        *args,
        export_formats: list[str] = ['geotiff'],
        export_config: Optional[dict] = None,
        **kwargs
    ):
        """
        Args:
            export_formats: List of output formats ['geotiff', 'las', 'laz', 'png'].
            export_config: Configuration for each exporter.
            ... (existing Palette parameters)
        """
        super().__init__(*args, **kwargs)
        self._setup_exporters(export_formats, export_config)

    def _setup_exporters(
        self,
        formats: list[str],
        config: Optional[dict]
    ) -> None:
        """Initialize exporters for requested formats."""
        ...

    def set_input(self, data: dict) -> None:
        """Extended set_input that preserves geo metadata."""
        super().set_input(data)
        self.geo_metadata = [
            GeoMetadata.from_dict(m) if isinstance(m, dict) else m
            for m in data.get('geo_metadata', [None] * self.batch_size)
        ]
        self.original_points = data.get('original_points', [None] * self.batch_size)
        self.tile_indices = data.get('tile_index', [None] * self.batch_size)

    def save_current_results(self) -> dict:
        """Save results with geospatial metadata."""
        results = super().save_current_results()
        # Add geo-specific exports
        for idx in range(self.batch_size):
            if self.geo_metadata[idx] is not None:
                self._export_geo_results(idx)
        return results

    def _export_geo_results(self, idx: int) -> dict[str, Path]:
        """Export results in geospatial formats."""
        ...

    def export_assembled(
        self,
        assembler: TileAssembler,
        output_dir: Path,
        base_name: str
    ) -> dict[str, Path]:
        """Export assembled results from tiled inference."""
        ...
```

---

## 9. CLI Interface Extensions

### 9.1 New Command-Line Arguments

```python
# File: run.py (extensions)

def add_geo_arguments(parser):
    """Add geospatial-specific CLI arguments."""

    geo_group = parser.add_argument_group('Geospatial I/O')

    # Input format options
    geo_group.add_argument(
        '--input_format',
        type=str,
        choices=['auto', 'las', 'laz', 'geotiff', 'png'],
        default='auto',
        help='Input data format (auto-detected if not specified)'
    )

    geo_group.add_argument(
        '--input_resolution',
        type=float,
        default=0.5,
        help='Resolution for rasterizing point clouds (meters)'
    )

    # Output format options
    geo_group.add_argument(
        '--output_formats',
        type=str,
        nargs='+',
        default=['geotiff'],
        choices=['geotiff', 'las', 'laz', 'png'],
        help='Output format(s) for results'
    )

    geo_group.add_argument(
        '--output_resolution',
        type=float,
        default=None,
        help='Output resolution (default: same as input)'
    )

    # Densification options
    geo_group.add_argument(
        '--densify_points',
        type=float,
        default=1.0,
        help='Point density for LAS/LAZ output (points per m^2)'
    )

    geo_group.add_argument(
        '--preserve_input_points',
        action='store_true',
        help='Include original input points in densified output'
    )

    # Tiling options
    geo_group.add_argument(
        '--tile_size',
        type=int,
        nargs=2,
        default=[128, 128],
        help='Tile size for large dataset processing'
    )

    geo_group.add_argument(
        '--tile_overlap',
        type=int,
        default=16,
        help='Overlap between tiles for seamless stitching'
    )

    return parser
```

---

## 10. Configuration Schema Extensions

### 10.1 JSON Configuration

```jsonc
// File: config/roof_completion_geo.json (example)
{
    "name": "roof_completion_geo",
    "gpu_ids": [0],

    // Existing dataset configuration
    "datasets": {
        "train": {
            "name": "GeoRoofDataset",
            "args": {
                "data_root": "./dataset/train",
                "footprint_root": "./dataset/train_footprint",

                // NEW: Geospatial loader configuration
                "loader_config": {
                    "las": {
                        "resolution": 0.5,
                        "classification_filter": [6],
                        "interpolation": "nearest"
                    },
                    "geotiff": {
                        "nodata_value": -9999
                    }
                },

                "tile_size": [128, 128],
                "tile_overlap": 0,

                // Existing corruption synthesis
                "mask_config": { ... },
                "noise_config": { ... }
            }
        }
    },

    // Existing model configuration
    "model": {
        "name": "GeoAwarePalette",
        "args": {
            // Existing Palette args...

            // NEW: Export configuration
            "export_formats": ["geotiff", "laz"],
            "export_config": {
                "geotiff": {
                    "dtype": "float32",
                    "compress": "lzw"
                },
                "laz": {
                    "point_density": 2.0,
                    "preserve_original_points": true,
                    "classification": 6
                }
            }
        }
    }
}
```

---

## 11. Directory Structure

New files and modules to be added:

```
RoofDiffusion/
├── data/
│   ├── geo/                        # NEW: Geospatial module
│   │   ├── __init__.py
│   │   ├── metadata.py             # GeoMetadata dataclass
│   │   ├── tile.py                 # GeoTile dataclass
│   │   ├── loaders.py              # LASLoader, GeoTIFFLoader, PNGLoader
│   │   ├── rasterizer.py           # GeoRasterizer
│   │   ├── exporters.py            # GeoTIFFExporter, LASExporter
│   │   └── assembler.py            # TileAssembler, StreamingAssembler
│   ├── geo_dataset.py              # NEW: GeoRoofDataset, GeoInferenceDataset
│   └── dataset.py                  # EXISTING (unchanged)
├── models/
│   ├── geo_model.py                # NEW: GeoAwarePalette
│   └── model.py                    # EXISTING (unchanged)
├── config/
│   └── roof_completion_geo.json    # NEW: Geo-enabled config template
├── scripts/
│   └── process_lidar.py            # NEW: Convenience script for LiDAR
└── docs/
    └── design/
        └── geo_io_extension.md     # THIS DOCUMENT
```

---

## 12. Dependencies

New Python dependencies required:

```
# requirements_geo.txt
laspy>=2.0.0          # LAS/LAZ I/O
lazrs>=0.5.0          # LAZ compression support for laspy
rasterio>=1.3.0       # GeoTIFF I/O
pyproj>=3.0.0         # CRS transformations
shapely>=2.0.0        # Geometry operations (optional, for footprints)
scipy>=1.9.0          # Interpolation for rasterization
```

---

## 13. Implementation Phases

### Phase 1: Core Infrastructure
1. Implement `GeoMetadata` and `GeoTile` dataclasses
2. Implement `GeoRasterizer` for point cloud to raster conversion
3. Implement `LASLoader` and `GeoTIFFLoader`
4. Unit tests for loaders and rasterizer

### Phase 2: Export Pipeline
1. Implement `GeoTIFFExporter`
2. Implement `LASExporter` with densification
3. Implement `TileAssembler` for large datasets
4. Integration tests for round-trip (LAS→process→LAS)

### Phase 3: Dataset Integration
1. Implement `GeoRoofDataset` extending `RoofDataset`
2. Implement `GeoInferenceDataset` for inference
3. Extend configuration schema
4. Update `run.py` with new CLI arguments

### Phase 4: Model Integration
1. Implement `GeoAwarePalette` extending `Palette`
2. Add metadata pass-through in training/inference loop
3. Multi-format export in result saving
4. End-to-end integration tests

### Phase 5: Documentation and Scripts
1. Update CLAUDE.md with new usage examples
2. Create convenience scripts for common workflows
3. Add example configurations
4. Performance optimization and profiling

---

## 14. Usage Examples

### Processing a LAS/LAZ Point Cloud

```bash
# Inference with LAS input, output both GeoTIFF and densified LAZ
python run.py -p test \
    -c config/roof_completion_geo.json \
    --resume ./pretrained/w_footprint/260 \
    --data_root ./input/sparse_roof.laz \
    --input_format las \
    --input_resolution 0.5 \
    --output_formats geotiff laz \
    --densify_points 4.0 \
    --preserve_input_points
```

### Processing a GeoTIFF DSM

```bash
# Inference with GeoTIFF input
python run.py -p test \
    -c config/roof_completion_geo.json \
    --resume ./pretrained/w_footprint/260 \
    --data_root ./input/sparse_dsm.tif \
    --input_format geotiff \
    --output_formats geotiff \
    --tile_size 128 128 \
    --tile_overlap 16
```

### Python API Usage

```python
from data.geo.loaders import LASLoader
from data.geo.exporters import GeoTIFFExporter, LASExporter
from models.geo_model import GeoAwarePalette

# Load sparse point cloud
loader = LASLoader(resolution=0.5, classification_filter=[6])
tile = loader.load("sparse_roof.laz")

# Run inference (model setup omitted for brevity)
model = GeoAwarePalette.load_from_checkpoint("checkpoint.pth")
result = model.inference(tile)

# Export to multiple formats
GeoTIFFExporter().export(result, tile.metadata, "output.tif")
LASExporter(point_density=4.0).export(result, tile.metadata, "output.laz")
```

---

## 15. Open Questions / Future Considerations

1. **Point attribute preservation**: Should we preserve intensity, RGB, return number, etc. from input LAS?

2. **Multi-band support**: Extend to handle multi-channel outputs (e.g., height + confidence)?

3. **Streaming inference**: For very large datasets that don't fit in GPU memory, implement streaming tile processing?

4. **CRS transformations**: Should we support on-the-fly reprojection between CRS?

5. **Building segmentation**: Auto-detect building footprints from point classification or density patterns?

6. **Quality masks**: Output confidence/uncertainty maps alongside height predictions?
