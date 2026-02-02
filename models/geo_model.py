"""Geospatial-aware Palette model for RoofDiffusion.

This module extends the Palette model with geospatial metadata handling
and multi-format export capabilities.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import numpy as np

from models.model import Palette
from data.geo.metadata import GeoMetadata
from data.geo.exporters import (
    BaseGeoExporter,
    GeoTIFFExporter,
    LASExporter,
    PNGExporter,
    create_exporter,
)
from data.geo.assembler import TileAssembler


class GeoAwarePalette(Palette):
    """Extended Palette model with geospatial export capabilities.

    Inherits all training/inference functionality from Palette,
    adds geospatial metadata handling and multi-format export.

    Example:
        >>> model = GeoAwarePalette(
        ...     networks=[network],
        ...     losses=[loss],
        ...     export_formats=['geotiff', 'laz'],
        ...     export_config={'geotiff': {'compress': 'lzw'}},
        ...     **kwargs
        ... )
        >>> model.test()  # Results exported in configured formats
    """

    def __init__(
        self,
        networks,
        losses,
        sample_num,
        task,
        optimizers,
        lr_schedulers=None,
        ema_scheduler=None,
        cond_on_mask=False,
        export_formats: List[str] = ["geotiff"],
        export_config: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize GeoAwarePalette.

        Args:
            networks: List of networks (diffusion network).
            losses: List of loss functions.
            sample_num: Number of intermediate samples to save.
            task: Task type ('inpainting', 'uncropping').
            optimizers: List of optimizer configurations.
            lr_schedulers: Learning rate scheduler configurations.
            ema_scheduler: EMA scheduler configuration.
            cond_on_mask: Whether to condition on mask.
            export_formats: List of output formats ['geotiff', 'las', 'laz', 'png'].
            export_config: Configuration for each exporter format.
            **kwargs: Additional arguments passed to Palette.
        """
        super(GeoAwarePalette, self).__init__(
            networks=networks,
            losses=losses,
            sample_num=sample_num,
            task=task,
            optimizers=optimizers,
            lr_schedulers=lr_schedulers,
            ema_scheduler=ema_scheduler,
            cond_on_mask=cond_on_mask,
            **kwargs
        )

        self.export_formats = export_formats
        self.export_config = export_config or {}
        self._setup_exporters()

        # Geospatial data storage
        self.geo_metadata: List[Optional[GeoMetadata]] = []
        self.original_points: List[Optional[np.ndarray]] = []
        self.tile_indices: List[Optional[tuple]] = []

    def _setup_exporters(self) -> None:
        """Initialize exporters for requested formats."""
        self.exporters: Dict[str, BaseGeoExporter] = {}

        for fmt in self.export_formats:
            config = self.export_config.get(fmt, {})
            try:
                self.exporters[fmt] = create_exporter(fmt, **config)
            except ValueError as e:
                self.logger.warning(f"Failed to create exporter for {fmt}: {e}")

    def set_input(self, data: Dict[str, Any]) -> None:
        """Extended set_input that preserves geo metadata.

        Args:
            data: Dictionary containing batch data including 'geo_metadata'.
        """
        # Call parent set_input
        super().set_input(data)

        # Extract geo-specific data
        geo_metadata_list = data.get("geo_metadata", [None] * self.batch_size)
        self.geo_metadata = []
        for m in geo_metadata_list:
            if isinstance(m, dict):
                self.geo_metadata.append(GeoMetadata.from_dict(m))
            elif isinstance(m, GeoMetadata):
                self.geo_metadata.append(m)
            else:
                self.geo_metadata.append(None)

        self.original_points = data.get("original_points", [None] * self.batch_size)
        if self.original_points is None:
            self.original_points = [None] * self.batch_size

        self.tile_indices = data.get("tile_index", [None] * self.batch_size)
        if self.tile_indices is None:
            self.tile_indices = [None] * self.batch_size

    def save_current_results(self) -> Dict[str, Any]:
        """Save results with geospatial metadata.

        Extends parent method to also export in geospatial formats.

        Returns:
            Dictionary with result paths and data.
        """
        # Call parent to save standard results
        results = super().save_current_results()

        # Export geo-specific results for each sample in batch
        for idx in range(self.batch_size):
            if self.geo_metadata[idx] is not None:
                self._export_geo_results(idx)

        return results

    def _export_geo_results(self, idx: int) -> Dict[str, Path]:
        """Export results in geospatial formats.

        Args:
            idx: Index in the current batch.

        Returns:
            Dictionary mapping format name to output path.
        """
        if self.geo_metadata[idx] is None:
            return {}

        metadata = self.geo_metadata[idx]
        output_tensor = self.output[idx:idx+1]  # Keep batch dimension
        base_name = Path(self.path[idx]).stem

        # Determine output directory
        results_dir = Path(self.opt["path"]["results"])
        geo_dir = results_dir / "geo"
        geo_dir.mkdir(parents=True, exist_ok=True)

        output_paths = {}

        for fmt, exporter in self.exporters.items():
            ext = exporter.get_extension()
            output_path = geo_dir / f"{base_name}{ext}"

            try:
                # Denormalize the output for export
                output_denorm = self._denormalize_output(output_tensor, idx)

                if isinstance(exporter, LASExporter):
                    original_pts = self.original_points[idx] if idx < len(self.original_points) else None
                    exporter.export(
                        output_denorm,
                        metadata,
                        output_path,
                        original_points=original_pts
                    )
                else:
                    exporter.export(output_denorm, metadata, output_path)

                output_paths[fmt] = output_path
                self.logger.info(f"Exported {fmt}: {output_path}")

            except Exception as e:
                self.logger.warning(f"Failed to export {fmt} for {base_name}: {e}")

        return output_paths

    def _denormalize_output(self, output: torch.Tensor, idx: int) -> torch.Tensor:
        """Denormalize output tensor using height range info.

        Args:
            output: Normalized output tensor in [-1, 1].
            idx: Batch index for height range lookup.

        Returns:
            Denormalized output tensor.
        """
        # Get height normalization parameters
        if self.height_range is not None and self.mid_height is not None:
            height_range = self.height_range[idx]
            mid_height = self.mid_height[idx]

            # Denormalize: from [-1, 1] to original height scale
            output_denorm = output.clone()
            output_denorm = output_denorm * 0.5 * height_range + mid_height
            return output_denorm

        # Also check geo_metadata for height info
        if self.geo_metadata[idx] is not None:
            metadata = self.geo_metadata[idx]
            if metadata.height_min is not None and metadata.height_max is not None:
                output_denorm = (output + 1) / 2  # [-1, 1] -> [0, 1]
                output_denorm = output_denorm * (metadata.height_max - metadata.height_min) + metadata.height_min
                return output_denorm

        return output

    def export_assembled(
        self,
        assembler: TileAssembler,
        output_dir: Path,
        base_name: str,
    ) -> Dict[str, Path]:
        """Export assembled results from tiled inference.

        Args:
            assembler: TileAssembler containing all processed tiles.
            output_dir: Directory for output files.
            base_name: Base name for output files.

        Returns:
            Dictionary mapping format name to output path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Assemble tiles
        height_map, metadata = assembler.assemble()
        height_tensor = torch.from_numpy(height_map).unsqueeze(0)

        # Get original points if available
        original_points = assembler.get_original_points()

        output_paths = {}

        for fmt, exporter in self.exporters.items():
            ext = exporter.get_extension()
            output_path = output_dir / f"{base_name}{ext}"

            try:
                if isinstance(exporter, LASExporter) and original_points is not None:
                    exporter.export(height_tensor, metadata, output_path, original_points)
                else:
                    exporter.export(height_tensor, metadata, output_path)

                output_paths[fmt] = output_path
                self.logger.info(f"Exported assembled {fmt}: {output_path}")

            except Exception as e:
                self.logger.warning(f"Failed to export assembled {fmt}: {e}")

        return output_paths

    def inference_tile(
        self,
        cond_image: torch.Tensor,
        footprint: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run inference on a single tile.

        Convenience method for tiled inference workflows.

        Args:
            cond_image: Conditional input image tensor (B, C, H, W).
            footprint: Optional footprint mask.
            mask: Optional loss mask.

        Returns:
            Output tensor (B, C, H, W).
        """
        self.netG.eval()

        with torch.no_grad():
            cond_image = self.set_device(cond_image)

            if self.cond_on_mask and mask is not None:
                mask = self.set_device(mask)
                mask_channel = mask.clone()
                mask_channel[mask_channel == 0] = -1
                input_image = torch.cat((cond_image, mask_channel), dim=1)
            else:
                input_image = cond_image

            if self.opt.get("distributed", False):
                output, _ = self.netG.module.restoration(
                    input_image, y_t=None, y_0=None, mask=mask, sample_num=0
                )
            else:
                output, _ = self.netG.restoration(
                    input_image, y_t=None, y_0=None, mask=mask, sample_num=0
                )

        return output

    def run_tiled_inference(
        self,
        dataset,
        output_dir: Path,
        base_name: str = "output",
    ) -> Dict[str, Path]:
        """Run inference on a tiled dataset and export assembled results.

        Args:
            dataset: GeoInferenceDataset or similar tiled dataset.
            output_dir: Directory for output files.
            base_name: Base name for output files.

        Returns:
            Dictionary mapping format name to output path.
        """
        from torch.utils.data import DataLoader

        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        grid_shape = dataset.get_tile_grid()

        # Create assembler
        tile_size = dataset.tile_size
        overlap = dataset.tile_overlap
        full_metadata = dataset.get_full_metadata()

        assembler = TileAssembler(
            tile_size=tile_size,
            overlap=overlap,
            blend_mode="linear",
        )

        # Process all tiles
        self.netG.eval()

        with torch.no_grad():
            for batch in loader:
                cond_image = self.set_device(batch["cond_image"])
                footprint = batch.get("footprint")
                tile_idx = batch["tile_index"]

                if footprint is not None:
                    footprint = self.set_device(footprint)

                # Run inference
                output = self.inference_tile(cond_image, footprint)

                # Create GeoTile for assembler
                from data.geo.tile import GeoTile

                metadata = GeoMetadata.from_dict(batch["geo_metadata"][0])
                tile = GeoTile(
                    height_map=cond_image.cpu().squeeze(0),
                    footprint=footprint.cpu().squeeze(0) if footprint is not None else torch.ones_like(cond_image.cpu().squeeze(0)),
                    metadata=metadata,
                    original_points=batch.get("original_points", [None])[0],
                    tile_index=(tile_idx[0].item(), tile_idx[1].item()),
                )

                # Add to assembler
                assembler.add_tile(
                    tile,
                    (tile_idx[0].item(), tile_idx[1].item()),
                    output.cpu().squeeze(0),
                )

        # Export assembled results
        return self.export_assembled(assembler, output_dir, base_name)

    def get_geo_results(self) -> List[Dict[str, Any]]:
        """Get geospatial results for the current batch.

        Returns:
            List of dictionaries with output tensor and metadata for each sample.
        """
        results = []

        for idx in range(self.batch_size):
            result = {
                "output": self.output[idx].cpu(),
                "metadata": self.geo_metadata[idx],
                "original_points": self.original_points[idx] if idx < len(self.original_points) else None,
                "tile_index": self.tile_indices[idx] if idx < len(self.tile_indices) else None,
                "path": self.path[idx],
            }
            results.append(result)

        return results
