"""Geospatial-aware datasets for RoofDiffusion.

This module provides extended dataset classes that support geospatial data formats
(LAS/LAZ, GeoTIFF) while maintaining compatibility with the original RoofDataset.
"""

from typing import List, Optional, Union, Dict, Any
from pathlib import Path
import os
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.transforms import functional as F

from data.dataset import RoofDataset, make_dataset, roof_pil_loader
from data.util.mask import get_multi_gauss_mask, get_down_res_mask
from data.util.noise import add_gauss_noise, add_outlier_noise
from data.util.roof_augment import add_tree_noise, random_height_scaling
from data.util.roof_transform import HeightNormalize
from data.geo.metadata import GeoMetadata
from data.geo.tile import GeoTile
from data.geo.loaders import create_loader, BaseGeoLoader


def is_geo_file(path: str) -> bool:
    """Check if a file is a geospatial format."""
    suffix = Path(path).suffix.lower()
    return suffix in [".las", ".laz", ".tif", ".tiff", ".geotiff"]


class GeoRoofDataset(data.Dataset):
    """Extended RoofDataset supporting geospatial formats.

    Wraps existing RoofDataset functionality while adding support
    for LAS/LAZ and GeoTIFF inputs with metadata preservation.

    The dataset automatically detects input format and uses appropriate loaders.
    Geospatial metadata is preserved and passed through to the output dictionary.

    Example:
        >>> dataset = GeoRoofDataset(
        ...     data_root='./dataset/train.flist',
        ...     footprint_root='./dataset/train_footprint.flist',
        ...     loader_config={'las': {'resolution': 0.5, 'classification_filter': [6]}},
        ... )
        >>> sample = dataset[0]
        >>> print(sample['geo_metadata'])  # Contains CRS, transform, etc.
    """

    def __init__(
        self,
        data_root: Union[str, Path],
        footprint_root: Optional[Union[str, Path]] = None,
        loader_config: Optional[Dict[str, Any]] = None,
        tile_size: tuple = (128, 128),
        tile_overlap: int = 0,
        # Existing RoofDataset parameters
        noise_config: Dict = {},
        mask_config: Dict = {},
        data_aug: Dict = {},
        data_len: int = -1,
        use_footprint: bool = True,
        footprint_as_mask: bool = False,
        recover_real_height: bool = False,
        no_height: bool = False,
        image_size: List[int] = [128, 128],
    ):
        """Initialize the GeoRoofDataset.

        Args:
            data_root: Path to data files or .flist file.
            footprint_root: Path to footprint files (optional for LAS/GeoTIFF
                           where footprints can be derived from point density).
            loader_config: Configuration for geospatial loaders.
                          Format: {'las': {...}, 'geotiff': {...}, 'png': {...}}
            tile_size: Size of tiles for large datasets.
            tile_overlap: Overlap between adjacent tiles.
            noise_config: Parameters for synthesizing noise.
            mask_config: Parameters for generating incompleteness mask.
            data_aug: Data augmentation settings.
            data_len: Number of data points to use (-1 for all).
            use_footprint: Whether to use footprint masking.
            footprint_as_mask: Use footprint as the training mask.
            recover_real_height: Return real-world height values.
            no_height: Replace height map with zeros (debug mode).
            image_size: Target image dimensions.
        """
        self.data_root = Path(data_root)
        self.footprint_root = Path(footprint_root) if footprint_root else None
        self.loader_config = loader_config or {}
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap

        self.noise_config = noise_config
        self.mask_config = mask_config
        self.use_footprint = use_footprint
        self.footprint_as_mask = footprint_as_mask
        self.recover_real_height = recover_real_height
        self.no_height = no_height
        self.image_size = image_size

        # Load file lists
        self.imgs = make_dataset(str(data_root))
        if footprint_root:
            self.footprints = make_dataset(str(footprint_root))
        else:
            self.footprints = []

        # Limit data if specified
        n_data = data_len if data_len > 0 else len(self.imgs)
        self.imgs = self.imgs[:n_data]
        self.footprints = self.footprints[:n_data] if self.footprints else []

        # Detect format
        self._detect_format()

        # Mask configuration
        self.down_res = mask_config.get("down_res_pct", [0])
        self.local_remove = mask_config.get("local_remove", [[0, 0, 0]])
        self.local_remove_percentage = mask_config.get("local_remove_percentage", -1)

        # Noise configuration
        self.min_gauss_noise_sigma = noise_config.get("min_gauss_noise_sigma", 0)
        self.max_gauss_noise_sigma = noise_config.get("max_gauss_noise_sigma", 0)
        self.outlier_noise_perc = noise_config.get("outlier_noise_percentage", 0)

        # Data augmentation
        self.repeat = data_aug.get("repeat", 1)
        self.rotate_deg = data_aug.get("rotate", 360)
        self.rotate_deg = 360 if self.rotate_deg == 0 else self.rotate_deg
        self.n_aug_deg = int(360 // self.rotate_deg)
        self.height_scale_prob = data_aug.get("height_scale_probability", 0)

        # Tree augmentation
        self.tree_aug = data_aug.get("tree", None)
        if self.tree_aug is not None:
            self.trees = make_dataset(self.tree_aug["flist_path"])
        else:
            self.trees = []

        # Image transforms
        self.height_normalize = HeightNormalize()
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            self.height_normalize,
        ])
        self.resize = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
        ])

        # Initialize loaders
        self._init_loaders()

    def _detect_format(self) -> None:
        """Detect the input format from file extensions."""
        if not self.imgs:
            self.format = "png"
            return

        first_file = self.imgs[0]
        suffix = Path(first_file).suffix.lower()

        if suffix in [".las", ".laz"]:
            self.format = "las"
        elif suffix in [".tif", ".tiff", ".geotiff"]:
            self.format = "geotiff"
        else:
            self.format = "png"

    def _init_loaders(self) -> None:
        """Initialize appropriate loaders based on format."""
        self.geo_loader: Optional[BaseGeoLoader] = None

        if self.format in ["las", "geotiff"]:
            config = self.loader_config.get(self.format, {})
            self.geo_loader = create_loader(
                Path(self.imgs[0]) if self.imgs else Path("."),
                **config
            )

    def _load_geo_tile(self, path: str) -> GeoTile:
        """Load a file using the appropriate geo loader.

        Args:
            path: Path to the file.

        Returns:
            GeoTile containing height map and metadata.
        """
        if self.geo_loader is None:
            raise RuntimeError("Geo loader not initialized")

        return self.geo_loader.load(Path(path))

    def _load_png(self, path: str) -> tuple:
        """Load a PNG file using the legacy loader.

        Args:
            path: Path to the PNG file.

        Returns:
            Tuple of (height_map_tensor, metadata).
        """
        height_tensor = roof_pil_loader(path)
        metadata = GeoMetadata(source_path=path)
        return height_tensor, metadata

    def _load_footprint(self, index: int) -> torch.Tensor:
        """Load footprint for the given index.

        Args:
            index: Data index.

        Returns:
            Footprint tensor.
        """
        if index < len(self.footprints):
            footprint_path = self.footprints[index]
            footprint = roof_pil_loader(footprint_path)
            footprint = self.resize(footprint)
            footprint = torch.gt(footprint, 0).float()
        else:
            # Generate footprint from height map validity
            footprint = torch.ones(1, self.image_size[0], self.image_size[1])

        return footprint

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a data sample with geospatial metadata.

        Args:
            index: Index into the augmented dataset.

        Returns:
            Dictionary containing:
                - cond_image: Corrupted input image
                - gt_image: Ground truth image
                - mask: Training mask
                - footprint: Building footprint
                - path: Source file path
                - geo_metadata: Serialized GeoMetadata dict
                - original_points: Original point cloud (if LAS input)
                - tile_index: Tile position (if tiled loading)
        """
        ret = {}

        # Compute raw index (before augmentation expansion)
        raw_idx = int(
            index // (len(self.down_res) * len(self.local_remove) * self.n_aug_deg * self.repeat)
        )

        img_path = self.imgs[raw_idx]
        original_points = None
        geo_metadata = GeoMetadata(source_path=img_path)

        # Load height map based on format
        if self.format in ["las", "geotiff"] and self.geo_loader is not None:
            try:
                geo_tile = self._load_geo_tile(img_path)
                # Resize to target size
                height_map = geo_tile.height_map
                if height_map.shape[-2:] != tuple(self.image_size):
                    height_map = F.resize(height_map, self.image_size)

                # Use tile's footprint if no separate footprint provided
                if raw_idx >= len(self.footprints):
                    footprint = geo_tile.footprint
                    if footprint.shape[-2:] != tuple(self.image_size):
                        footprint = F.resize(footprint, self.image_size)
                    footprint = torch.gt(footprint, 0).float()
                else:
                    footprint = self._load_footprint(raw_idx)

                geo_metadata = geo_tile.metadata
                original_points = geo_tile.original_points

                # Apply height normalization
                img = self.height_normalize(height_map)

            except Exception as e:
                # Fallback to PNG loader
                print(f"Warning: Failed to load {img_path} as geo file: {e}")
                img = roof_pil_loader(img_path)
                img = self.tfs(img)
                footprint = self._load_footprint(raw_idx)
        else:
            # Legacy PNG loading
            if self.no_height:
                img = torch.zeros(1, self.image_size[0], self.image_size[1])
            else:
                img = roof_pil_loader(img_path)
                img = self.tfs(img)
            footprint = self._load_footprint(raw_idx)

        # Rotation augmentation
        rot_idx = int(index % self.n_aug_deg)
        rot_deg = rot_idx * self.rotate_deg

        img = F.rotate(img, rot_deg)
        footprint = F.rotate(footprint, rot_deg)
        footprint = torch.clamp(footprint, 0, 1)

        # Generate mask if not using pre-computed masks
        fp = footprint if self.use_footprint else torch.ones_like(footprint)
        mask = self._get_mask(index, fp.numpy())

        gt_img = img.clone()

        # Random height scaling
        gt_img = random_height_scaling(gt_img, footprint, self.height_scale_prob)
        img = gt_img.clone()

        # Tree noise augmentation
        if self.tree_aug is not None and self.trees:
            img = add_tree_noise(img, footprint.numpy(), self.tree_aug, self.trees)

        # Apply noise
        if not self.use_footprint:
            footprint = torch.ones_like(footprint)

        img = add_gauss_noise(img, self.min_gauss_noise_sigma, self.max_gauss_noise_sigma)
        img = add_outlier_noise(img, self.outlier_noise_perc)

        # Create conditional image with missing pixels
        cond_img = img * (1.0 - mask) - mask
        cond_img = cond_img * footprint - (1 - footprint)
        mask_img = (img * (1.0 - mask) + mask) * footprint - (1 - footprint)

        # Populate return dictionary
        ret["cond_image"] = cond_img
        ret["gt_image"] = gt_img
        ret["mask"] = footprint if self.footprint_as_mask else mask
        ret["mask_image"] = mask_img
        ret["footprint"] = footprint
        ret["path"] = Path(img_path).name

        # Height normalization info
        ret["height_range"] = self.height_normalize.height_range
        ret["mid_height"] = self.height_normalize.mid_height

        # Geospatial metadata
        ret["geo_metadata"] = geo_metadata.to_dict()
        ret["original_points"] = original_points

        return ret

    def _get_mask(self, index: int, footprint: np.ndarray) -> torch.Tensor:
        """Generate mask for removing pixels.

        Args:
            index: Data index.
            footprint: Footprint array.

        Returns:
            Mask tensor where 1 = removed, 0 = kept.
        """
        dr_idx = index % len(self.down_res)
        lr_idx = index % len(self.local_remove)

        n_footprint_pixels = np.sum(footprint > 0)

        for _ in range(50):
            # Global downsampling mask
            dr_mask = get_down_res_mask(footprint, self.down_res[dr_idx])

            # Local removal mask
            min_sigma_ratio, max_sigma_ratio, n_gaussian = self.local_remove[lr_idx]
            gauss_mask = get_multi_gauss_mask(
                footprint,
                min_sigma_ratio,
                max_sigma_ratio,
                n_gauss_mask=n_gaussian,
                remove_percentage=self.local_remove_percentage,
            )
            lr_mask = gauss_mask * footprint

            # Combine masks
            mask = np.logical_or(dr_mask, lr_mask)

            if np.sum(mask == 1) < n_footprint_pixels:
                break

        return torch.from_numpy(mask).float()

    def __len__(self) -> int:
        """Return the length of the augmented dataset."""
        return (
            len(self.imgs)
            * len(self.down_res)
            * len(self.local_remove)
            * self.n_aug_deg
            * self.repeat
        )


class GeoInferenceDataset(data.Dataset):
    """Inference-only dataset for processing real-world geospatial data.

    Simplified dataset for inference without corruption synthesis.
    Handles large files through tiling with overlap for seamless stitching.

    Example:
        >>> dataset = GeoInferenceDataset(
        ...     input_path='building.laz',
        ...     tile_size=(128, 128),
        ...     tile_overlap=16,
        ... )
        >>> for sample in DataLoader(dataset, batch_size=1):
        ...     result = model.inference(sample)
    """

    def __init__(
        self,
        input_path: Union[str, Path],
        footprint_path: Optional[Union[str, Path]] = None,
        tile_size: tuple = (128, 128),
        tile_overlap: int = 16,
        loader_config: Optional[Dict[str, Any]] = None,
        image_size: Optional[List[int]] = None,
    ):
        """Initialize the inference dataset.

        Args:
            input_path: Path to input file (LAS/LAZ/GeoTIFF/PNG).
            footprint_path: Optional path to building footprints.
            tile_size: Processing tile size.
            tile_overlap: Overlap for seamless stitching.
            loader_config: Loader-specific configuration.
            image_size: If specified, resize tiles to this size.
        """
        self.input_path = Path(input_path)
        self.footprint_path = Path(footprint_path) if footprint_path else None
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.loader_config = loader_config or {}
        self.image_size = image_size

        # Create loader
        self.loader = create_loader(self.input_path, **self.loader_config)

        # Load all tiles
        self.tiles: List[GeoTile] = []
        for tile in self.loader.load_tiled(
            self.input_path, tile_size=tile_size, overlap=tile_overlap
        ):
            self.tiles.append(tile)

        # Load footprint if provided
        self.footprint_tiles: List[Optional[torch.Tensor]] = []
        if self.footprint_path and self.footprint_path.exists():
            footprint_loader = create_loader(self.footprint_path)
            for fp_tile in footprint_loader.load_tiled(
                self.footprint_path, tile_size=tile_size, overlap=tile_overlap
            ):
                self.footprint_tiles.append(fp_tile.footprint)

        # Compute grid shape
        if self.tiles:
            max_row = max(t.tile_index[0] for t in self.tiles if t.tile_index)
            max_col = max(t.tile_index[1] for t in self.tiles if t.tile_index)
            self._grid_shape = (max_row + 1, max_col + 1)
        else:
            self._grid_shape = (0, 0)

        # Height normalization
        self.height_normalize = HeightNormalize()

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get a tile for inference.

        Args:
            index: Tile index.

        Returns:
            Dictionary containing:
                - cond_image: Input height map (normalized)
                - footprint: Validity mask
                - geo_metadata: Geospatial metadata dict
                - original_points: Original point cloud (if LAS)
                - tile_index: (row, col) position
        """
        tile = self.tiles[index]

        # Get height map
        height_map = tile.height_map

        # Resize if needed
        if self.image_size and height_map.shape[-2:] != tuple(self.image_size):
            height_map = F.resize(height_map, self.image_size)

        # Get footprint
        if index < len(self.footprint_tiles) and self.footprint_tiles[index] is not None:
            footprint = self.footprint_tiles[index]
        else:
            footprint = tile.footprint

        if self.image_size and footprint.shape[-2:] != tuple(self.image_size):
            footprint = F.resize(footprint, self.image_size)

        # Prepare conditional image (set invalid regions to -1)
        cond_image = height_map * footprint - (1 - footprint)

        return {
            "cond_image": cond_image,
            "footprint": footprint,
            "geo_metadata": tile.metadata.to_dict(),
            "original_points": tile.original_points,
            "tile_index": tile.tile_index,
            "path": tile.metadata.source_path or "",
        }

    def __len__(self) -> int:
        """Return total number of tiles."""
        return len(self.tiles)

    def get_tile_grid(self) -> tuple:
        """Return (n_rows, n_cols) of tile grid."""
        return self._grid_shape

    def get_full_metadata(self) -> GeoMetadata:
        """Get metadata for the full (non-tiled) extent.

        Returns:
            GeoMetadata with bounds covering all tiles.
        """
        if not self.tiles:
            return GeoMetadata()

        # Merge bounds from all tiles
        all_bounds = [
            t.metadata.bounds
            for t in self.tiles
            if t.metadata.bounds is not None
        ]

        if not all_bounds:
            return self.tiles[0].metadata.copy()

        min_x = min(b[0] for b in all_bounds)
        min_y = min(b[1] for b in all_bounds)
        max_x = max(b[2] for b in all_bounds)
        max_y = max(b[3] for b in all_bounds)

        first = self.tiles[0].metadata
        return GeoMetadata(
            crs=first.crs,
            bounds=(min_x, min_y, max_x, max_y),
            resolution=first.resolution,
            height_min=first.height_min,
            height_max=first.height_max,
        )
