"""Geospatial I/O extension for RoofDiffusion.

This module provides support for:
- Input formats: LAS/LAZ point clouds, GeoTIFF DSM/DTM rasters
- Output formats: Densified LAS/LAZ point clouds, GeoTIFF rasters

Key components:
- GeoMetadata: Stores geospatial reference information
- GeoTile: Primary data container with height data and metadata
- Loaders: LASLoader, GeoTIFFLoader, PNGLoader
- Exporters: GeoTIFFExporter, LASExporter, PNGExporter
- GeoRasterizer: Converts point clouds to raster grids
- TileAssembler: Reassembles tiled outputs
"""

from data.geo.metadata import GeoMetadata
from data.geo.tile import GeoTile
from data.geo.loaders import (
    BaseGeoLoader,
    LASLoader,
    GeoTIFFLoader,
    PNGLoader,
    create_loader,
)
from data.geo.rasterizer import GeoRasterizer
from data.geo.exporters import (
    BaseGeoExporter,
    GeoTIFFExporter,
    LASExporter,
    PNGExporter,
    MultiFormatExporter,
)
from data.geo.assembler import TileAssembler, StreamingAssembler

__all__ = [
    "GeoMetadata",
    "GeoTile",
    "BaseGeoLoader",
    "LASLoader",
    "GeoTIFFLoader",
    "PNGLoader",
    "create_loader",
    "GeoRasterizer",
    "BaseGeoExporter",
    "GeoTIFFExporter",
    "LASExporter",
    "PNGExporter",
    "MultiFormatExporter",
    "TileAssembler",
    "StreamingAssembler",
]
