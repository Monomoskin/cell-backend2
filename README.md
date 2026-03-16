

# High-Throughput Bamboo Callus Volume Estimation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

This repository implements a non-destructive, high-throughput pipeline for estimating bamboo callus volume in Petri dishes using computer vision and machine learning, as described in the paper:

> High-Throughput Volume Estimation of Bamboo Callus Using Computer Vision and Machine Learning  
> Zhou Ming Bing, Enzo Battaglia Mbula  
> Zhejiang A&F University, School of Bamboo  
> (2026)

**Core features**:

- Customized Mask R-CNN (Detectron2) for instance segmentation
- Dual-image strategy: top-view (XY area) + side-view (Z height)
- Dynamic calibration using real Petri dish dimensions (diameter 90 mm, height 12 mm)
- Feature extraction + calibrated OLS regression for accurate volume in mL
- Exploratory damage detection with percentage estimation
- Performance on held-out real test set: **MAE = 0.366 mL, RMSE = 0.674 mL, R² = 0.855**

## Prerequisites (macOS – CPU-only)

- macOS (Ventura, Sonoma, Sequoia or later)
- Python 3.10 or higher → [Download official installer](https://www.python.org/downloads/macos/)
- Git (usually pre-installed; if missing: run `xcode-select --install`)

**Note**: This project runs fully on CPU (no GPU/CUDA required). Detectron2 CPU mode on macOS is stable and easy to set up.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Monomoskin/cell-backend2.git
   cd cell-backend2
   ```


2. (Recommended) Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   If Detectron2 installation fails (rare on recent macOS), install CPU-only PyTorch first:

   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

   Installation typically takes 5–10 minutes.

## How the Pipeline Works

The script processes **paired images** (`*_TOP.jpg/png` and `*_SIDE.jpg/png`):

1. **Side view processing (Z height calibration)**
   - Detects container mask → computes mm/pixel factor from known flask height (12 mm)
   - Extracts callus profile mask → measures real height in mm

2. **Top view processing (XY area calibration)**
   - Detects container mask → computes mm²/pixel² factor from known flask diameter (90 mm)
   - Extracts callus mask → measures real base area in mm²

3. **Volume calculation**
   - Geometric volume = Area XY × Height Z (converted to mL)
   - Refined with OLS-calibrated regression using additional features (area, solidity, texture)

4. **Damage estimation (exploratory)**
   - Identifies defective regions via texture/color deviations
   - Computes % damaged area relative to callus

5. **Outputs**
   - Two versions per view: clean (masks only) + with text overlays (class, volume, defect %, quality)
   - Per-sample CSV (`{sample}_volumes.csv`)
   - Consolidated results (`all_volumes_summary.xlsx`) with predicted/real volumes, errors, precision %

## Running Inference

**Single sample**:

```bash
python predict.py \
  --top_image testImages_copy/sample1_TOP.jpg \
  --side_image testImages_copy/sample1_SIDE.jpg \
  --output_dir output_predict/
```

**Batch mode** (process folder of pairs):

```bash
python predict.py \
  --input_dir testImages_copy/ \
  --output_dir output_predict/
```

**Filename convention** (batch mode):

- `sample1_TOP.jpg` + `sample1_SIDE.jpg` (matched automatically by prefix)
- Supported extensions: .jpg, .jpeg, .png

**Expected runtime** (macOS CPU): ~5–15 seconds per pair

## Reproducing Paper Metrics

To reproduce the test-set results (MAE 0.366 mL, RMSE 0.674 mL, R² 0.855 on 29 held-out real samples):

1. Place test images in a folder (e.g., `examples/test_set/`) with matching TOP/SIDE pairs.
2. Run batch inference:
   ```bash
   python predict.py --input_dir examples/test_set/ --output_dir examples/results/
   ```
3. Open `examples/results/all_volumes_summary.xlsx`  
   → Compare predicted volumes, absolute errors, and precision percentages with the paper's tables and scatter plot.

Model weights (`output_train_attr/model_final.pth`) should be in place.

## License

MIT License – see the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this code for academic, research, or commercial purposes, provided the original copyright and license notice are included.

## Acknowledgments

- Built with [Detectron2](https://github.com/facebookresearch/detectron2)
- Thanks to the laboratory staff at Zhejiang A&F University for bamboo callus sample collection

Questions, issues, or collaborations? Open an issue or contact enzombul@gmail.com

Happy phenotyping! 🌱

```

```
