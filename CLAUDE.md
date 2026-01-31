# RoofDiffusion

**Paper**: "RoofDiffusion: Constructing Roofs from Severely Corrupted Point Data via Diffusion" (ECCV 2024)
**arXiv**: https://arxiv.org/abs/2404.09290

## Paper Methodology

### Problem Statement
Reconstructing complete 3D roof height maps from severely corrupted LiDAR point cloud data. The corruption includes:
- **Sparsity**: Large portions of data missing (up to 99% downsampling)
- **Local incompleteness**: Gaussian-shaped missing regions
- **Sensor noise**: Gaussian noise and outliers
- **Environmental noise**: Tree occlusions

### Approach
The method uses a **conditional diffusion model** based on the Palette framework to learn the distribution of complete roof structures and generate plausible completions from corrupted inputs.

#### Key Components

1. **Forward Diffusion Process**: Gradually adds Gaussian noise to clean height maps over T timesteps using a linear beta schedule
2. **Reverse Diffusion Process**: Learns to denoise using a time-conditioned UNet, iteratively recovering the clean image
3. **Conditioning**: The corrupted height map (with footprint) is concatenated with the noisy image as input to the denoiser
4. **Building Footprint Guidance**: Optional binary mask indicating building boundaries, used to focus loss computation and improve reconstruction quality

#### Training Objective
Masked L1 loss computed only within building footprint regions:
```
L = ||M * (x_0 - x_pred)||_1
```
where M is the footprint mask, x_0 is ground truth, and x_pred is the predicted clean image.

### Data Synthesis Pipeline
Training data is synthesized by applying controlled corruptions to clean height maps:
- Sparsity: Random downsampling by 25-99%
- Local removal: Multiple Gaussian masks covering 30-80% of pixels
- Gaussian noise: sigma in [0, 0.05]
- Outlier noise: 1% of pixels
- Tree overlay: 30% probability, 1-3 trees per sample

---

## Code Structure

```
RoofDiffusion/
├── config/                     # Configuration files (JSON with comments)
│   ├── roof_completion.json    # With footprint guidance
│   └── roof_completion_no_footprint.json
├── core/                       # Framework infrastructure
│   ├── base_dataset.py         # Abstract dataset class
│   ├── base_model.py           # Training/testing loop management
│   ├── base_network.py         # Weight initialization utilities
│   ├── logger.py               # Logging and TensorBoard
│   ├── praser.py               # Config parser with dynamic imports
│   └── util.py                 # General utilities
├── models/                     # Neural network components
│   ├── model.py                # Palette trainer (main training logic)
│   ├── network.py              # Diffusion network with noise scheduling
│   ├── loss.py                 # Masked L1/MSE losses
│   ├── metric.py               # MAE, inception score
│   ├── guided_diffusion_modules/
│   │   ├── unet.py             # UNet architecture (default)
│   │   └── nn.py               # Network primitives
│   └── sr3_modules/            # Alternative SR3 UNet
├── data/                       # Data handling
│   ├── dataset.py              # RoofDataset class
│   ├── __init__.py             # Dataloader creation
│   └── util/                   # Data processing utilities
│       ├── mask.py             # Sparsity and local removal masks
│       ├── noise.py            # Gaussian and outlier noise
│       ├── roof_augment.py     # Tree overlay, height scaling
│       ├── roof_transform.py   # Height normalization
│       └── roof_metric.py      # RMSE, MAE, IoU evaluation
├── dataset/                    # Data directory
│   └── PoznanRD/               # Poznan Roof Dataset
├── pretrained/                 # Pre-trained model checkpoints
├── experiments/                # Training/testing outputs
├── scripts/                    # Benchmark evaluation scripts
├── run.py                      # Main entry point
└── gen_benchmark.py            # Benchmark dataset generation
```

---

## Key Components

### Entry Point: `run.py`
- Parses config and command-line arguments
- Initializes distributed training (DDP)
- Creates dataloaders, networks, losses, metrics
- Runs training or testing loop

### Model: `models/model.py` (Palette)
- EMA (Exponential Moving Average) for stable training
- Adam optimizer with linear warmup scheduler
- Training step: noise GT, predict noise, compute masked loss
- Testing step: iterative denoising (500-2000 steps)

### Network: `models/network.py`
- Linear beta schedule for noise variance
- `q_sample()`: Forward diffusion (add noise)
- `p_sample()`: Reverse diffusion (denoise one step)
- UNet denoiser with time embeddings

### Dataset: `data/dataset.py` (RoofDataset)
- Loads uint16 height maps (height_m = pixel_value / 256)
- Applies corruption pipeline (sparsity, noise, trees)
- Returns: corrupted image, ground truth, footprint mask

---

## Usage

### Training
```bash
# With footprint guidance
python run.py -p train -c config/roof_completion.json

# Without footprint guidance
python run.py -p train -c config/roof_completion_no_footprint.json
```

### Inference
```bash
python run.py -p test -c config/roof_completion.json \
    --resume ./pretrained/w_footprint/260 \
    --n_timestep 500 \
    --data_root ./dataset/PoznanRD/benchmark/w_footprint/s95_i30/img.flist \
    --footprint_root ./dataset/PoznanRD/benchmark/w_footprint/s95_i30/footprint.flist
```

### Evaluation
```bash
python data/util/roof_metric.py \
    --gt_dir ./dataset/PoznanRD/benchmark/w_footprint/s95_i30/roof_gt \
    --footprint_dir ./dataset/PoznanRD/benchmark/w_footprint/s95_i30/roof_footprint \
    --pred_dir experiments/test_roof_completion_XXXXXX/results/test/0 \
    --img_name_prefix BID
```

---

## Configuration Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `down_res_pct` | Sparsity levels (% of data kept) | [99,98,90,80,50,25] |
| `local_remove` | [min_sigma, max_sigma, n_gaussian] | [[0,0.3,5],[0.15,0.3,5]] |
| `noise_config` | Gaussian/outlier noise parameters | sigma 0-0.05, 1% outliers |
| `tree` | Tree overlay augmentation | 30% prob, 1-3 trees |
| `n_timestep` | Diffusion steps (train/test) | 2000/1000 |
| `inner_channel` | UNet base channels | 64 |
| `channel_mults` | Channel multipliers per resolution | [1,2,4,8] |

---

## Data Format

- **Height maps**: uint16 PNG, height(m) = pixel_value / 256
- **Footprints**: Binary masks (1 = building, 0 = outside)
- **Resolution**: 128x128 pixels
- **Dataset**: Poznan Roof Dataset (PoznanRD)

---

## Dependencies

Based on [Palette-Image-to-Image-Diffusion-Models](https://github.com/Janspiry/Palette-Image-to-Image-Diffusion-Models)

- PyTorch
- TorchVision
- NumPy
- PIL/OpenCV
- TensorBoard
- SciPy
