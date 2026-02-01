#!/usr/bin/env python3
#######################################################################
# Kymograph Annotation Analysis and Export
# 
# This script processes manually annotated kymographs to extract quantitative
# pumping rate data.
# 
# Key functions:
# 1. Loads annotated kymograph images with color-coded pumping events
# 2. Aligns kymographs to compensate for worm movement
# 3. Extracts pumping events from manual annotations
# 4. Calculates pumping rates in different time windows
# 5. Exports results to CSV files for statistical analysis
#######################################################################

# Modules
import os  # File and folder path handling
from glob import glob  # File pattern matching
import numpy as np  # Numerical computing and array operations
import pandas as pd  # Data manipulation and analysis
import matplotlib.pyplot as plt  # Plotting and image generation
from PIL import Image  # Image processing and loading

# Tierpsy Tracker - C. elegans computer vision library
# https://github.com/SinRas/tierpsy-tracker
import tierpsy
from tierpsy.analysis.ske_create.getSkeletonsTables import getWormMask, getSkeleton

# Custom utilities for data processing
from utils import *
# Import configuration parameters
from config import *


#######################################################################
# Kymograph Processing Functions
# 
# These functions handle the loading, alignment, and analysis of
# manually annotated kymographs to extract quantitative pumping data.
#######################################################################

def find_annotation_pixels(arr_rgb, color, mask_nonconsecutive=True):
    """
    Find pixels in an image that match a specific annotation color.
    
    This function identifies manually annotated regions in kymographs by
    detecting pixels that match the expected annotation colors (red for
    pumping events, green for excluded regions).
    
    Args:
        arr_rgb: RGB image array (height, width, 3)
        color: Target RGB color to find (numpy array)
        mask_nonconsecutive: If True, only return first pixel of consecutive regions
        
    Returns:
        Array of pixel indices where the color was found
    """
    # Calculate color distance for each pixel
    arr_rgb_coloredness = np.linalg.norm(
        arr_rgb.astype(np.float32) - color[np.newaxis, np.newaxis, :],
        axis=2
    )
    
    # Find pixels within color tolerance
    indices_color = np.where(np.nanmin(arr_rgb_coloredness, axis=0) < 10)[0]
    indices_color = np.sort(indices_color)
    
    # Remove consecutive pixels if requested (keep only first pixel of each region)
    if mask_nonconsecutive:
        _mask_non_consecutive = np.ones_like(indices_color, dtype=np.bool_)
        _mask_non_consecutive[1:] = np.diff(indices_color) > 1
        indices_color = indices_color[_mask_non_consecutive]
    # Return
    return indices_color

def extract_annotations_from_kymograph(fp_read, T_lower=0, T_max=40*120):
    """
    Extract pumping events and excluded regions from annotated kymograph images.
    
    This function processes manually annotated kymographs to identify:
    - Red pixels: Mark pumping events
    - Green pixels: Mark regions to exclude from analysis
    
    Args:
        fp_read: Path to annotated kymograph image
        T_lower: Starting time frame for analysis
        T_max: Ending time frame for analysis
        
    Returns:
        Tuple of (image_shape, pumping_indices, excluded_indices)
    """
    # Load annotated image and crop to analysis time window
    food_entry_kymograph_annotations = np.array(
        Image.open(fp_read)
    )[:, T_lower:T_max, :3]
    
    # Extract pumping events (red pixels)
    indices_pumps = find_annotation_pixels(food_entry_kymograph_annotations, KYMOGRAPH_ANNOTATION_COLOR_RED)
    
    # Extract excluded regions (green pixels)
    indices_exclude = find_annotation_pixels(food_entry_kymograph_annotations, KYMOGRAPH_ANNOTATION_COLOR_GREEN, mask_nonconsecutive=False)
    # Return
    return food_entry_kymograph_annotations.shape, indices_pumps, indices_exclude

#######################################################################
# Main Analysis Pipeline
# 
# The following sections execute the complete analysis pipeline:
# 1. Generate aligned kymograph images for manual annotation
# 2. Process annotated kymographs to extract pumping data
# 3. Calculate pumping rates and export results
#######################################################################

if __name__ == "__main__":
    # Check if data path is configured
    if FP_READ_FOLDER == "/path/to/your/worm_behavior_recordings":
        print("ERROR: Please update FP_READ_FOLDER in config.py to point to your data directory.")
        print("Current value is a placeholder. Exiting...")
        exit(1)
    
    # Check if data directory exists
    if not os.path.exists(FP_READ_FOLDER):
        print(f"ERROR: Data directory does not exist: {FP_READ_FOLDER}")
        print("Please check the path in config.py. Exiting...")
        exit(1)
    
    print(f"Starting annotation analysis for data in: {FP_READ_FOLDER}")
    
    ###################################
    # Setup
    # All parameters are imported from config.py
    fps_cases = sorted(glob(
        os.path.join(FP_PUMPING_EXTRACTS, "*.npz")
    ))
    print(f"Found {len(fps_cases)} cases")
    if len(fps_cases) == 0:
        raise ValueError("No NPZ files found in {FP_WRITE_FOLDER}")

    ###################################
    # Process annotated kymographs and extract pumping rate data
    _data_points_missing = []  # Track datasets with missing annotations
    _data_points = []  # Store pumping rate measurements

    for label_section, T_lower, T_upper in TIME_INTERVALS.items():
        for fp_npz in tqdm(fps_cases):
            # File paths
            # condition, strain, _, filename = fp_npz.split(os.sep)[2:]
            filename = fp_npz.split(os.sep)[-1]
            condition, strain, _ = filename.split('_', 2)
            fp_annotated = os.path.join(
                FP_PUMPING_ANALYSIS,
                f"{condition}_{strain}_{filename[:-4]}_annotated.png"
            )
            # If exists
            if not os.path.exists(fp_annotated):
                continue
            # Load
            with np.load(fp_npz) as in_file:
                times = in_file['timestamps']
            ## Annotations
            img_shape, indices_pumps, indices_exclude = extract_annotations_from_kymograph( fp_annotated, T_lower, T_upper )
            # Store missing indices
            _data_points_missing.append((
                label_section,
                condition, strain,
                img_shape[1], len(indices_exclude),
                fp_annotated, fp_npz
            ))
            ###################################
            # Process annotations
            array_pumps = np.zeros(img_shape[1], dtype=np.bool_)
            array_pumps[indices_pumps] = True
            # Find intervals
            intervals = []
            if len(indices_exclude) == 0:
                intervals = [ (0, img_shape[1]) ]
            else:
                _indices = np.array([0] + list(indices_exclude))
                # First to one-before-last
                for il, ir in zip( _indices[:-1], _indices[1:] ):
                    # Paddings
                    ir -= PADDING_FOR_GREEN
                    il = il if il == 0 else il + PADDING_FOR_GREEN
                    if ir-il < (WINDOW_SIZE_MIN):
                        continue
                    intervals.append((il, ir))
                # Last
                il = _indices[-1] + PADDING_FOR_GREEN
                ir = img_shape[1]
                if ir-il > WINDOW_SIZE_MIN:
                    intervals.append((il, ir))
            ###################################
            # Calculate pumping rate for windows
            for il, ir in intervals:
                for i in range(il, ir, WINDOW_SIZE):
                    j = min( i + WINDOW_SIZE, ir )
                    _n = j - i
                    if _n < WINDOW_SIZE/2:
                        continue
                    # Store the interval
                    _duration = times[j] - times[i] if j < len(times) else times[-1] - times[i]
                    _n_pumps = array_pumps[i:j].sum()
                    _data_points.append((
                        label_section,
                        condition, strain,
                        _duration, _n_pumps, _n_pumps/_duration,
                        i, j, _n,
                        fp_annotated, fp_npz
                    ))
    if len(_data_points) == 0:
        raise ValueError("No pumping rate data found! -> make sure there are annotated kymographs in {FP_ANALYSIS_FOLDER}.")

    #######################################################################
    # Data Export and Summary Statistics
    # 
    # Convert extracted pumping data to structured formats suitable for
    # further analysis.
    #######################################################################

    # Convert to DataFrames for analysis
    df_nontracking = pd.DataFrame(
        _data_points_missing,
        columns=[
            'label_section',
            'condition', 'strain',
            'n_frames', 'n_frames_nontracking',
            'fp_annotation', 'fp_data'
        ]
    )
    df_nontracking['duration'] = df_nontracking['n_frames']/40.0
    df_nontracking['time_fraction_non_tracking'] = df_nontracking['n_frames_nontracking'] / df_nontracking['n_frames']
    df_nontracking['duration_nontracking'] = df_nontracking['duration'] * df_nontracking['time_fraction_non_tracking']
    ###################################
    df_pumping_rates = pd.DataFrame(
        _data_points,
        columns=[
            'label_section',
            'condition', 'strain',
            'window_duration_seconds', 'window_pumps',
            'window_pumping_rate_Hz',
            'idx_start', 'idx_end', 'n_indices',
            'fp_annotation', 'fp_data'
        ]
    )
    df_pumping_rates['weight'] = df_pumping_rates['window_duration_seconds'] * df_pumping_rates['window_pumps']
    ###################################
    df_pumping_rates_per_dataset = df_pumping_rates.groupby(['label_section', 'condition', 'strain', 'fp_data']).agg({
        'window_pumps': 'sum',
        'window_duration_seconds': 'sum'
    }).reset_index()
    df_pumping_rates_per_dataset['pumping_rate_avg_Hz'] = df_pumping_rates_per_dataset['window_pumps'] / df_pumping_rates_per_dataset['window_duration_seconds']
    ###################################
    df_pumping_rates_stats = df_pumping_rates.groupby(['label_section', 'condition', 'strain']).agg({
        'fp_annotation': [ 'nunique', 'count' ],
        'window_duration_seconds': ['sum']
    })
    df_pumping_rates_stats.columns = [
        'n_datasets_annotated', 'n_15second_windows', 'total_duration_annotated'
    ]
    df_pumping_rates_stats.reset_index(inplace=True)
    # Merge with the general dataframe
    df_pumping_rates_per_dataset = pd.merge(
        df_pumping_rates_per_dataset,
        df_pumping_rates_stats,
        on=['label_section', 'condition', 'strain']
    )

    # Export results to CSV files for further analysis
    df_nontracking.to_csv(
        os.path.join(FP_PUMPING_ANALYSIS, "notracking_indices.csv"),
        index=False
    )
    df_pumping_rates.to_csv(
        os.path.join(FP_PUMPING_ANALYSIS, "rates_per_15s_windows.csv"),
        index=False
    )
    df_pumping_rates_per_dataset.to_csv(
        os.path.join(FP_PUMPING_ANALYSIS, "rates_per_dataset.csv"),
        index=False
    )

