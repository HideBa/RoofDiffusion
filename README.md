<!-- # RoofDiffusion
Welcome to the official implementation of paper "RoofDiffusion: Constructing Roofs from Severely Corrupted Point Data via Diffusion" -->

<h1 align="left">RoofDiffusion: Constructing Roofs from Severely Corrupted Point Data via Diffusion
 <a href="https://arxiv.org/abs/2404.09290"><img  src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg" ></a> </h1> 

Update: We are excited to share that our work has been accepted for ECCV 2024! 🎉

[5min Video](https://youtu.be/34nyxi3qHJQ?si=z3u9Vx3lkTIywduf)

![Demo](demo.png)

## Setup
```console
conda env create -f environment.yml
conda activate RoofDiffusion
```

## Download Dataset \& Pretrained Model
Download [here](https://drive.google.com/drive/folders/1o_I4Z-9xRT7PqBgXQQgVUlcDOwOTT9Qj?usp=drive_link) .

Unzip and place dataset under the `RoofDiffusion/dataset` of repo e.g. `RoofDiffusion/dataset/PoznanRD`

Place RoofDiffusion pretrained model at `RoofDiffusion/pretrained/w_footprint/260_Network.pth`

Or place No-NF RoofDiffusion pretrained model at `RoofDiffusion/pretrained/wo_footprint/140_Network.pth`

> The height maps are in uint16 format, where the actual roof height (meter) = pixel value / 256. (same as [KITTI dataset](https://www.cvlibs.net/datasets/kitti/eval_depth_all.php))

## Training
**RoofDiffusion**

Use `roof_completion.json` for training the RoofDiffusion model with the footprint version, where in each footprint image, a pixel value of 1 denotes the building footprint and 0 denotes areas outside the footprint. 
```console
python run.py -p train -c config/roof_completion.json
```

**No-FP RoofDiffusion**

Use `roof_completion_no_footprint.json` for training with footprint images where all pixels are set to 1, indicating no distinction between inside and outside footprint areas.
```console
python run.py -p train -c config/roof_completion_no_footprint.json
```

See training progress
```console
tensorboard --logdir experiments/train_roof_completion_XXXXXX_XXXXXX
```

## Inference
**RoofDiffusion**
```console
python run.py -p test -c config/roof_completion.json \
    --resume ./pretrained/w_footprint/260 \
    --n_timestep 500 \
    --data_root ./dataset/PoznanRD/benchmark/w_footprint/s95_i30/img.flist \
    --footprint_root ./dataset/PoznanRD/benchmark/w_footprint/s95_i30/footprint.flist
```

**No-FP RoofDiffusion**
```console
python run.py -p test -c config/roof_completion_no_footprint.json \
    --resume ./pretrained/wo_footprint/140 \
    --n_timestep 500 \
    --data_root ./dataset/PoznanRD/benchmark/wo_footprint/s95_i30/img.flist \
    --footprint_root ./dataset/PoznanRD/benchmark/wo_footprint/s95_i30/footprint.flist
```

> Tested on NVIDIA RTX3090. Please adjust `batch_size` in JSON file if out of GPU memory.

## Geospatial I/O Extension

RoofDiffusion now supports direct processing of geospatial data formats (LAS/LAZ point clouds and GeoTIFF rasters) while preserving coordinate reference systems and spatial metadata.

### Installation
```console
pip install -r requirements_geo.txt
```

This installs:
- `laspy` + `lazrs` for LAS/LAZ point cloud I/O
- `rasterio` for GeoTIFF raster I/O
- `pyproj` for coordinate reference system handling

### Processing LiDAR Point Clouds

**Single LAS/LAZ file:**
```console
python scripts/process_lidar.py input.laz -o output.tif --checkpoint ./pretrained/w_footprint/260
```

**With densified point cloud output:**
```console
python scripts/process_lidar.py input.laz -o output.laz --densify --density 4.0 --preserve-original
```

**Batch process a directory:**
```console
python scripts/process_lidar.py ./input_dir -o ./output_dir --batch
```

### Processing GeoTIFF DSM/DTM

```console
python run.py -p test -c config/roof_completion_geo.json \
    --resume ./pretrained/w_footprint/260 \
    --data_root ./input/sparse_dsm.tif \
    --input_format geotiff \
    --output_formats geotiff \
    --tile_size 128 128 \
    --tile_overlap 16
```

### Key Options
| Option | Description | Default |
|--------|-------------|---------|
| `--input_resolution` | Resolution for rasterizing point clouds (m) | 0.5 |
| `--classification_filter` | LAS classification codes to include | 6 (buildings) |
| `--output_formats` | Output format(s) | geotiff |
| `--densify_points` | Point density for LAZ output (pts/m²) | 1.0 |
| `--tile_size` | Tile size for large files | 128 |
| `--tile_overlap` | Overlap between tiles | 16 |

### Python API
```python
from data.geo.loaders import LASLoader
from data.geo.exporters import GeoTIFFExporter, LASExporter

# Load sparse point cloud
loader = LASLoader(resolution=0.5, classification_filter=[6])
tile = loader.load("sparse_roof.laz")

# ... run model inference ...

# Export to multiple formats
GeoTIFFExporter().export(result, tile.metadata, "output.tif")
LASExporter(point_density=4.0).export(result, tile.metadata, "output.laz")
```

## Customize New Benchmark
Make costomized benchmark by adjusting parameters in `make_test_image.py` and run
```console
python gen_benchmark.py
```

## Customize Data Synthesis for Training
Modify JSON config file:
- `"down_res_pct"` controls sparsity.
- `"local_remove"` adjusts local incompleteness (Please refer to paper for details).
- `"noise_config"` injects senser/environmental noise.
- `"height_scale_probability"` randomly scales the distance between the min-max nonzero roof height.
- `"tree"` introduce tree noise into height maps


## Metric
Evaluate the predicted roof height map.
For example:
```console
python data/util/roof_metric.py \
    --gt_dir ./dataset/PoznanRD/benchmark/w_footprint/s95_i30/roof_gt \
    --footprint_dir ./dataset/PoznanRD/benchmark/w_footprint/s95_i30/roof_footprint \
    --pred_dir PRED_DIR \
    --img_name_prefix IMG_NAME_PREFIX
```
Set the path to the images predicted by model e.g. `PRED_DIR="experiments/test_roof_completion_XXXXXX_XXXXXX/results/test/0"` \
For PoznanRD dataset, `IMG_NAME_PREFIX="BID"`. \
For Cambridge and WayneCo dataset, `IMG_NAME_PREFIX=""`.

## Visualization
We view the uint16 height map using [ImageJ](https://imagej.net/ij/download.html)

## Scripts
Reproduce paper results by running shell scripts files from project's root folder. e.g. `./scripts/PoznanRD_w_FP_metric.sh`

## Acknowledge
This project is based on the following wonderful implementation of the paper [Palette: Image-to-Image Diffusion Models](https://arxiv.org/abs/2111.05826) \
https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models

## Citing
```
@article{lo2024roofdiffusion,
  title={RoofDiffusion: Constructing Roofs from Severely Corrupted Point Data via Diffusion},
  author={Lo, Kyle Shih-Huang and Peters, J{\"o}rg and Spellman, Eric},
  journal={arXiv preprint arXiv:2404.09290},
  year={2024}
}
```
