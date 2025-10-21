# Yeon-2025-pumping-analysis
Scripts to quantify pumping frequency in "An enteric neuron-expressed ionotropic receptor detects ingested salts to regulate salt stress resistance," Yeon, ..., Sengupta 2025.

This set of scripts analyzes pharyngeal pumping behavior in *C. elegans* from video recordings. The analysis pipeline generates kymographs (time-space plots) that visualize intensity changes along the worm's body over time, enabling quantitative measurement of pumping rates.

## Overview

The analysis consists of two main steps:

1. **Kymograph Generation** (`01_make_kymographs.py`): Processes video recordings to create kymographs showing pharyngeal pumping patterns
2. **Annotation Analysis** (`02_load_annotated_kymographs_and_export.py`): Extracts quantitative pumping rate data from manually annotated kymographs

## Requirements

- Python 3.8+
- See `requirements.txt` for all dependencies
- [SinRas/Tierpsy Tracker](https://github.com/SinRas/tierpsy-tracker) for C. elegans computer vision. My fork of [Tierpsy Tracker](https://github.com/Tierpsy/tierpsy-tracker) that enabled installation in your current python env.


## Installation

1. Download this set of scripts
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install Tierpsy Tracker:
   By following [SinRas/Tierpsy Tracker](https://github.com/SinRas/tierpsy-tracker) or try the original repo [Tierpsy Tracker](https://github.com/Tierpsy/tierpsy-tracker).

## Usage


### 1. Configure Data Path

Edit `config.py` and set `FP_READ_FOLDER` to point to your data directory:

```python
FP_READ_FOLDER = "/path/to/your/worm_behavior_recordings"
```

### 2. Data Structure

Organize your data as follows:
```
your_data_folder/
├── WORM1_behavior/
│   ├── 000.h5
│   ├── 001.h5
│   └── ...
├── WORM2_behavior/
│   ├── 000.h5
│   └── ...
└── metadata/
```

### 3. Generate Kymographs

```bash
python 01_make_kymographs.py
```

This creates:
- Kymograph images (PNG)
- Processed videos with skeleton overlays (MP4)
- Numerical data (NPZ, MAT files)

### 4. Manual Annotation

Manually annotate the generated kymograph images using image editing software, e.g. Paint:
- **Red pixels**: Mark pumping events
- **Green pixels**: Mark regions to exclude from analysis

Make sure to use "pen" tools, e.g. no brush or patterned coloring. And also add the RGB values the **red** and **green** colors used, in the `config.py` file. E.g.
- `KYMOGRAPH_ANNOTATION_COLOR_RED = np.array([237, 28, 36], dtype=np.float32)` corresponds to `rgb(237,28,36)`.

### 5. Extract Pumping Rates

```bash
python 02_load_annotated_kymographs_and_export.py
```

This generates CSV files with pumping rate measurements.

- `rates_per_15s_windows.csv`: Pumping rates for 15-second time windows
- `rates_per_dataset.csv`: Average pumping rates per dataset
- `notracking_indices.csv`: Summary of excluded regions

## Configuration

Key parameters in `config.py`:
- `FRAMES_PER_SECOND`: Video frame rate (default: 40 Hz)
- `UM_PER_PIXEL`: Micrometers per pixel calibration
- `SKELETON_N_SEGMENTS`: Number of skeleton points (default: 101)
- `WINDOW_SIZE`: Analysis window size in frames (default: 600 = 15 seconds)

## Citation

**Main Publication:**
```
Yeon J, Chen L, Krishnan N, Bates S, Porwal C, Sengupta P.
An enteric neuron-expressed variant ionotropic receptor detects ingested salts to regulate salt stress resistance.
bioRxiv [Preprint]. 2025 May 8:2025.04.11.648259. doi: 10.1101/2025.04.11.648259. PMID: 40391324; PMCID: PMC12087990.
```

**Software Dependencies:**
- [Tierpsy Tracker](https://github.com/Tierpsy/tierpsy-tracker) - Original C. elegans tracking software
- [SinRas/Tierpsy Tracker](https://github.com/SinRas/tierpsy-tracker) - Fork enabling installation in current Python environment


