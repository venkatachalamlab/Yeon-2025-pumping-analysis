# Yeon-2025-pumping-analysis

Scripts to quantify pharyngeal pumping frequency in *C. elegans* from video recordings, as described in **"An enteric neuron-expressed ionotropic receptor detects ingested salts to regulate salt stress resistance"**, *Yeon, ..., Sengupta 2025*.

The analysis pipeline generates kymographs (time-space plots) that visualize intensity changes along the worm's body over time, enabling quantitative measurement of pumping rates.

## Overview

The analysis consists of two main steps:

1. **Kymograph Generation** (`01_make_kymographs.py`): Processes video recordings to create kymographs showing pharyngeal pumping patterns
2. **Annotation Analysis** (`02_load_annotated_kymographs_and_export.py`): Extracts quantitative pumping rate data from manually annotated kymographs

Supporting files:
- `config.py`: All configurable parameters (file paths, thresholds, imaging parameters)
- `utils.py`: Utility functions for video I/O, data loading, and image processing classes

## System Requirements

### Software Dependencies

- Python 3.8
- conda (Anaconda or Miniconda) for environment management
- See `requirements.txt` for all Python package dependencies:
  - numpy 1.24.4
  - scipy 1.10.1
  - pandas 2.0.3
  - matplotlib 3.7.5
  - opencv-python 4.13.0.90
  - Pillow 10.4.0
  - h5py 3.11.0
  - tqdm 4.67.2
- [SinRas/Tierpsy Tracker](https://github.com/SinRas/tierpsy-tracker) for *C. elegans* computer vision -- a fork of the [original Tierpsy Tracker](https://github.com/Tierpsy/tierpsy-tracker) modified for Python module installation; cloned and included in this repository under `tierpsy-tracker/`

### Operating System

- **Tested on:** Ubuntu 24.04 LTS (Linux kernel 6.17), Windows 10/11
- **Expected to work on:** Linux, macOS, Windows (with conda)

### Versions Tested

- Python 3.8.20
- All dependency versions as listed in `requirements.txt`

### Non-standard Hardware

No non-standard hardware is required. The software runs on a standard desktop computer.

## Installation

> **Windows users:** Clone or extract the repository to a **short path** (e.g., `C:\pumping\`). The Tierpsy Tracker Cython extension build generates deeply nested paths that can exceed the Windows 260-character path limit (`MAX_PATH`), causing linker errors during installation. Additionally, [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) are required to compile the Cython extension.

1. Create and activate a conda environment:
   ```bash
   conda create -n pumping python=3.8
   conda activate pumping
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install Tierpsy Tracker. The `tierpsy-tracker/` folder included in this repository is a clone of [SinRas/Tierpsy Tracker](https://github.com/SinRas/tierpsy-tracker), a fork of the [original Tierpsy Tracker](https://github.com/Tierpsy/tierpsy-tracker) modified to allow installation as a Python module via `pip install -e .`:
   ```bash
   cd tierpsy-tracker
   pip install -e .
   cd ..
   ```

### Typical Install Time

Approximately 5 minutes on a standard desktop computer with a broadband internet connection (most of the time is spent downloading packages).

## Demo

### Sample Data

A small demo dataset containing two short recordings (400 frames each, ~10 seconds at 40 fps) is available as a ZIP file from the [GitHub Releases page](https://github.com/venkatachalamlab/Yeon-2025-pumping-analysis/releases/tag/v1.0).

**Automatic download:** Running `python 01_make_kymographs.py` will automatically download and extract the sample data if the `data/` directory does not exist.

**Manual download:** You can also download and extract the sample data manually:

```bash
# Download the sample data ZIP file
wget https://github.com/venkatachalamlab/Yeon-2025-pumping-analysis/releases/download/v1.0/data.zip

# Create the data directory and extract
mkdir -p data
unzip data.zip -d data
rm data.zip
```

After extraction, the `data/` directory should contain:

- `data/worm1_50mM_N2_behavior/000_010.h5` -- Wild-type (N2) worm in 50 mM NaCl
- `data/worm2_50mM_glr9_behavior/000_010.h5` -- *glr-9* mutant worm in 50 mM NaCl
- `data/pumping_analysis/` -- Pre-annotated kymograph images for the demo

### Running the Demo

**Step 1: Generate kymographs**

```bash
conda activate pumping
python 01_make_kymographs.py
```

This produces output in `data/pumping_extracts/`:
- Kymograph images (PNG): intensity along the skeleton and along normals, both raw and rescaled
- Quality control videos (MP4): foreground masks, worm masks, and skeleton overlays
- Numerical data (NPZ, MAT): timestamps, intensities, and distance arrays

Aligned kymograph images for annotation are saved in `data/pumping_analysis/`.

**Step 2: Manually annotate kymographs**

Annotate the generated kymograph PNG images using image editing software (e.g., Microsoft Paint):
- **Red pixels** (`rgb(237, 28, 36)`): Mark pumping events
- **Green pixels** (`rgb(34, 177, 76)`): Mark regions to exclude from analysis

Use solid "pen" tools (not brushes or patterned fills). Save the annotated images with `_annotated` appended to the filename (e.g., `*_kymograph_all_annotated.png`).

Pre-annotated kymograph images for the demo data are included in `data/pumping_analysis/`.

**Step 3: Extract pumping rates**

```bash
python 02_load_annotated_kymographs_and_export.py
```

### Expected Output

Three CSV files are generated in `data/pumping_analysis/`:

| File | Description |
|------|-------------|
| `rates_per_15s_windows.csv` | Pumping rates for each 15-second time window |
| `rates_per_dataset.csv` | Average pumping rates per worm/dataset |
| `notracking_indices.csv` | Summary of excluded/non-tracking regions |

Expected pumping rates for the demo data:
- **N2 (wild-type):** ~4.0 Hz (44 pumps in ~10.9 seconds)
- **glr-9 mutant:** ~4.9 Hz (43 pumps in ~8.7 seconds)

### Expected Run Time

- **Step 1** (kymograph generation): less than 5 minutes for the sample data on a standard desktop computer
- **Step 3** (annotation analysis): a few seconds

## Instructions for Use

### Running on Your Own Data

1. **Organize your data** in the following folder structure:
   ```
   your_data_folder/
   ├── WORMID_CONDITION_STRAIN_behavior/
   │   ├── 000.h5
   │   ├── 001.h5
   │   └── ...
   ├── WORMID_CONDITION_STRAIN_behavior/
   │   ├── 000.h5
   │   └── ...
   └── metadata/          (created automatically)
   ```

   Each `.h5` file must contain:
   - `times`: 1D numpy array (`shape=(n_frames,)`) of unix timestamps for each frame
   - `data`: 3D numpy array (`shape=(n_frames, height, width)`) of grayscale video frames

2. **Configure the data path** in `config.py`:
   ```python
   FP_READ_FOLDER = "/path/to/your_data_folder"
   ```

3. **Adjust imaging parameters** in `config.py` if your microscope setup differs:
   - `PIXELS_PER_MM`: Microscope calibration (pixels per millimeter)
   - `FRAMES_PER_SECOND`: Video frame rate (Hz)
   - `MASK_THRESHOLD_FOREGROUND`: Intensity threshold for worm segmentation

4. **Run the pipeline** as described in the Demo section above (Steps 1--3).

5. **Head-tail correction** (if needed): In `01_make_kymographs.py`, the `indices_skeleton_reverse` dictionary allows manual correction of frames where the skeleton head-tail orientation is incorrect. Add frame indices where the orientation flips:
   ```python
   indices_skeleton_reverse[(start_frame, end_frame)] = {frame1, frame2, ...}
   ```

### Configuration

Key parameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FRAMES_PER_SECOND` | 40.0 | Video frame rate (Hz) |
| `PIXELS_PER_MM` | 820 | Microscope calibration |
| `UM_PER_PIXEL` | 1.22 | Micrometers per pixel (derived) |
| `SKELETON_N_SEGMENTS` | 101 | Number of skeleton points |
| `WINDOW_SIZE` | 600 | Analysis window (15 s at 40 fps) |
| `MASK_THRESHOLD_FOREGROUND` | 110 | Worm segmentation threshold |

## Code Description

A complete description of the analysis algorithms is provided in the **Methods** section of the manuscript. In brief:

1. **Worm segmentation:** Foreground masking via intensity thresholding and connected component analysis, followed by morphological refinement.
2. **Skeleton extraction:** Centerline tracing using the Tierpsy Tracker computer vision library, with temporal smoothing and spline interpolation.
3. **Kymograph generation:** Intensity sampling along the skeleton and perpendicular normals, producing time-space plots.
4. **Pumping quantification:** Manual annotation of pumping events on kymographs, followed by automated extraction of pumping rates in configurable time windows.

## License

This software is released under the [MIT License](LICENSE), approved by the [Open Source Initiative](https://opensource.org/licenses/MIT).

## Repository

Source code is available at: https://github.com/venkatachalamlab/Yeon-2025-pumping-analysis

## Citation

**Main Publication:**
```
Yeon J, Chen L, Krishnan N, Bates S, Porwal C, Sengupta P.
An enteric neuron-expressed variant ionotropic receptor detects ingested salts
to regulate salt stress resistance.
bioRxiv [Preprint]. 2025 May 8:2025.04.11.648259.
doi: 10.1101/2025.04.11.648259. PMID: 40391324; PMCID: PMC12087990.
```

**Software Dependencies:**
- [Tierpsy Tracker](https://github.com/Tierpsy/tierpsy-tracker) -- Original *C. elegans* tracking software
- [SinRas/Tierpsy Tracker](https://github.com/SinRas/tierpsy-tracker) -- Fork enabling installation in current Python environment
