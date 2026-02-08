#!/usr/bin/env python3
#######################################################################
# Kymograph Generation for C. elegans Pharyngeal Pumping Analysis
# 
# This script processes video recordings of C. elegans to generate kymographs
# that visualize pharyngeal pumping behavior over time. The analysis pipeline:
# 
# 1. Loads behavior recordings from HDF5 files
# 2. Segments worms from background using image processing
# 3. Extracts skeleton centerlines using Tierpsy tracker
# 4. Creates kymographs showing intensity changes along the body
# 5. Generates quality control videos with skeleton overlays
# 
# Key outputs:
# - Kymograph images showing pumping patterns
# - Processed videos for quality control
# - Numerical data for further analysis
#######################################################################

# Modules
import os  # File and folder path handling
import gc  # Garbage collection for memory management
import datetime  # Timestamp generation
import numpy as np  # Numerical computing and array operations
import matplotlib.pyplot as plt  # Plotting and image generation
from scipy.io import savemat  # MATLAB file export

# Tierpsy Tracker - C. elegans computer vision library
# https://github.com/SinRas/tierpsy-tracker
import tierpsy
from tierpsy.analysis.ske_create.getSkeletonsTables import getWormMask, getSkeleton

# Custom utilities for data processing
from utils import *
# Import configuration parameters
from config import *


#######################################################################
# Computer Vision Functions
# 
# These functions handle the image processing pipeline for worm detection
# and skeleton extraction. They work together to convert raw video frames
# into structured data suitable for behavioral analysis.
#######################################################################

def create_foreground_mask(img_beh, threshold_foreground=130, threshold_area_min_object=100, threshold_area_max_object=500_000):
    """
    Create a binary mask identifying worm pixels in the image.
    
    This function segments the worm from the background by:
    1. Applying intensity thresholding to find dark objects
    2. Filtering by object size to remove debris and artifacts
    
    Args:
        img_beh: Input grayscale image (numpy array)
        threshold_foreground: Intensity threshold for worm detection (0-255)
        threshold_area_min_object: Minimum object size in pixels
        threshold_area_max_object: Maximum object size in pixels
        
    Returns:
        Binary mask where True indicates worm pixels
    """
    # Apply multi-scale thresholding to handle varying image quality
    img_beh_masked = (cv.medianBlur(img_beh, 11) < threshold_foreground) | \
                     (cv.medianBlur(img_beh, 3) < threshold_foreground) | \
                     (img_beh < threshold_foreground)
    
    # Remove objects that are too small or too large (debris, artifacts)
    n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(img_beh_masked.astype(np.uint8))
    img_beh_masked &= False  # Reset mask
    
    # Keep only objects within size constraints
    for label, area in enumerate(stats[:, -1]):
        if label == 0 or area < threshold_area_min_object or area > threshold_area_max_object:
            continue
        img_beh_masked |= labels == label
    # Return
    return img_beh_masked
def extract_worm_mask(img_mask, coords_center=None, size_close=11):
    """
    Extract the worm mask by selecting the largest connected component and filling gaps.
    
    This function refines the foreground mask by:
    1. Finding the object closest to the expected worm position
    2. Filling gaps in the worm outline using morphological operations
    3. Ensuring the mask represents a single, connected worm
    
    Args:
        img_mask: Binary foreground mask from create_foreground_mask()
        coords_center: Expected worm position (None = use image center)
        size_close: Size of morphological closing kernel
        
    Returns:
        Refined binary mask representing the worm
    """
    if coords_center is None:
        coords_center = np.array(img_mask.shape, dtype=np.float32) / 2
    
    # Find all connected components in the mask
    n_labels, labels, stats, centroids = cv.connectedComponentsWithStats(img_mask.astype(np.uint8))
    
    # Handle case where no objects are found
    if n_labels <= 1:
        return np.zeros_like(img_mask, dtype=np.bool_)
    
    # Select the object closest to the expected center position
    dists = np.linalg.norm(centroids - coords_center[np.newaxis, :], axis=1)
    label_closest_center = 1 + np.argmin(dists[1:])
    worm_mask = labels == label_closest_center
    
    # Fill gaps in the worm outline using morphological closing
    worm_mask = cv.morphologyEx(
        worm_mask.astype(np.uint8),
        cv.MORPH_CLOSE,
        cv.getStructuringElement(cv.MORPH_ELLIPSE, (size_close, size_close))
    ) > 0
    
    # Fill interior regions by inverting and finding background
    n_labels, labels, stats, centroids = cv.connectedComponentsWithStats((~worm_mask).astype(np.uint8))
    label_all_background = 1 + np.argmax(stats[1:, -1])
    worm_mask = labels != label_all_background
    # Return
    return worm_mask
def convert_mask_for_tierpsy(worm_mask):
    """
    Convert binary mask to format expected by Tierpsy skeleton extraction.
    
    Tierpsy expects specific intensity values:
    - Background: 255 (white)
    - Threshold: 110 (splits background from worm)
    - Worm: 55 (dark gray, or any value below 110)
    
    Args:
        worm_mask: Binary mask where True = worm pixels
        
    Returns:
        Grayscale image with proper intensity values for Tierpsy
    """
    img_tierpsy = (worm_mask == 0).astype(np.uint8) * 200  # Background = 200
    img_tierpsy += 55  # Worm = 55, Background = 255
    return img_tierpsy
def extract_skeleton(worm_mask, skeleton_prev=np.zeros(0), n_skeleton_segments=101):
    """
    Extract worm skeleton using Tierpsy computer vision library.
    
    This function uses Tierpsy's algorithms to find the centerline of the worm,
    which is essential for creating kymographs.
    
    Args:
        worm_mask: Binary mask representing the worm
        skeleton_prev: Previous skeleton for temporal consistency (optional)
        n_skeleton_segments: Number of points along the skeleton
        
    Returns:
        Array of skeleton points (x, y coordinates)
    """
    # Convert mask to Tierpsy format
    worm_tierpsy = convert_mask_for_tierpsy(worm_mask)
    
    # Extract worm contour using Tierpsy
    _, worm_cnt, _ = getWormMask(
        worm_img=worm_tierpsy,
        threshold=110,  # Don't change it, this is paired with `img_mask_to_tierpsy`
        strel_size=11,  # Hard coded in Tierpsy
        min_blob_area=TIERPSY_MIN_BLOB_AREA,  # Change if you need to find skeleton for a very tiny worm, which should not be the case!
        is_light_background=TIERPSY_IS_LIGHT_BACKGROUND
    )
    
    # Extract skeleton from contour
    output = getSkeleton(
        worm_cnt=worm_cnt,
        prev_skeleton=skeleton_prev,
        resampling_N=n_skeleton_segments
    )
    skeleton, ske_len, cnt_side1, cnt_side2, cnt_widths, cnt_area = output
    # Return
    return skeleton

#######################################################################
# Time Series Processing Functions
# 
# These functions handle temporal smoothing of skeleton data to reduce
# noise and improve the quality of kymograph generation. They implement
# different types of moving averages for different analysis needs.
#######################################################################

def moving_average_causal(arr, k, dtype=np.float32):
    """
    Apply causal (past-only) moving average to time series data.
    
    This function smooths data using only past values, making it suitable
    for real-time analysis where future data is not available.
    
    Args:
        arr: Input time series array
        k: Window size for averaging
        dtype: Output data type
        
    Returns:
        Smoothed time series array
    """
    if k <= 1:
        return arr.copy()
    if np.isnan(arr[0]) or np.isnan(arr[-1]):
        il = 0
        while il < len(arr) and np.isnan(arr[il]):
            il += 1
        ir = len(arr)-1
        while ir >= 0 and np.isnan(arr[ir]):
            ir -= 1
        ir += 1
        assert il < ir, "Please don't test me on all edge cases :)) give me a better array! :grin: (array is all NaNs)"
        result = np.zeros_like(arr) * np.nan
        result[il:ir] = moving_average_causal( arr[il:ir], k )
        return result
    # Rolling counts
    w = k+1
    arr_cumsum = np.nancumsum( arr, axis=0 )
    ns_cumsum = np.cumsum( ~np.isnan(arr), axis=0 )
    result = np.zeros_like(arr_cumsum, dtype=dtype) * np.nan
    # Beginning
    result[:w] = arr_cumsum[:w] / ns_cumsum[:w]
    # Rest
    result[w:] = (arr_cumsum[w:] - arr_cumsum[:-w]) / np.maximum(ns_cumsum[w:] - ns_cumsum[:-w], 1)
    # NaNs
    _indices = w + np.where( ns_cumsum[w:] == ns_cumsum[:-w] )[0]
    result[_indices] = np.nan
    # Return
    return result
def moving_average_symmetric(arr, k, dtype=np.float32):
    """
    Apply symmetric (past and future) moving average to time series data.
    
    This function smooths data using both past and future values, providing
    better smoothing but requiring the entire time series to be available.
    
    Args:
        arr: Input time series array
        k: Window size for averaging
        dtype: Output data type
        
    Returns:
        Smoothed time series array
    """
    if k <= 1:
        return arr.astype(dtype)
    return (moving_average_causal(arr, k, dtype=dtype) * (k+1) + 
            moving_average_causal(arr[::-1], k-1, dtype=dtype)[::-1] * k) / (2*k+1)

#######################################################################
# Skeleton Processing and Kymograph Generation
# 
# These functions handle the geometric processing of skeleton data and
# the creation of kymographs that visualize intensity changes along the
# worm's body over time.
#######################################################################

from scipy.interpolate import make_interp_spline

def smooth_skeleton(arr2d, stride=5, k=3):
    """
    Smooth skeleton coordinates using cubic spline interpolation.
    
    This function reduces noise in skeleton tracking by fitting smooth
    curves through the skeleton points.
    
    Args:
        arr2d: 2D array of skeleton coordinates (time, points, x/y)
        stride: Sampling interval for spline fitting
        k: Spline order (3 = cubic)
        
    Returns:
        Smoothed skeleton coordinates
    """
    ts = np.arange(len(arr2d))
    spl = make_interp_spline(ts[::stride], arr2d[::stride], k=k)
    return spl(ts)
def interpolate_skeleton(coords, dists_to_interpolate, k=3):
    """
    Interpolate skeleton points at specific distances from the head.
    
    This function creates a standardized skeleton representation by
    sampling points at regular intervals along the worm's length.
    
    Args:
        coords: Skeleton coordinates (points, x/y)
        dists_to_interpolate: Target distances from head (in pixels)
        k: Spline order for interpolation
        
    Returns:
        Interpolated skeleton coordinates at specified distances
    """
    # Calculate cumulative distances along skeleton
    dists = np.zeros(len(coords))
    dists[1:] = np.cumsum(np.linalg.norm(np.diff(coords, axis=0), axis=1))
    
    # Interpolate at target distances
    spl = make_interp_spline(dists, coords, k=k)
    return spl(dists_to_interpolate)
def compute_skeleton_normals(skeleton, normal_factors=np.arange(-5.0, 6.0, 1.0)):
    """
    Create perpendicular lines (normals) extending from skeleton points.
    
    These normals are used to measure intensity profiles
    across the body for kymograph generation.
    
    Args:
        skeleton: Skeleton coordinates (points, x/y)
        normal_factors: Distances along normals (negative = left, positive = right)
        
    Returns:
        3D array of normal line coordinates (midpoints, x/y, normal_points)
    """
    # Calculate midpoints between consecutive skeleton points
    skeleton_mids = (skeleton[:-1] + skeleton[1:]) / 2
    
    # Calculate normal vectors (perpendicular to skeleton direction)
    diffs = np.diff(skeleton, axis=0)
    dists = np.linalg.norm(diffs, axis=1)
    normals = diffs[:, ::-1] / dists[:, np.newaxis]  # Rotate 90 degrees
    normals[:, 0] *= -1  # Flip x-component for correct orientation
    
    # Create normal lines extending from midpoints
    skeleton_mids_normals = skeleton_mids[:, :, np.newaxis] + \
                           (normals[:, :, np.newaxis] * normal_factors[np.newaxis, np.newaxis, :])
    
    return skeleton_mids_normals

#######################################################################
# Main Kymograph Processing Functions
# 
# These functions process a segment of video data to generate kymographs
# and quality control videos. They handle the complete pipeline from
# raw video frames to kymograph images.
#######################################################################

def load_and_align_kymograph(fp_npz, scale_roll=1.0, il=130, ir=200):
    """
    Load kymograph data and align it to compensate for worm movement.
    
    This function addresses the challenge that worms can move during recording,
    causing the pharynx to shift position in the kymograph. Alignment is done
    by rolling the kymograph along the normal lines.
    
    Args:
        fp_npz: Path to NPZ file containing kymograph data
        scale_roll: Scaling factor for alignment correction
        il: Left boundary of analysis region (pixels from head)
        ir: Right boundary of analysis region (pixels from head)
        
    Returns:
        Tuple of (original_kymograph, aligned_kymograph)
    """
    # Load
    with np.load(fp_npz) as in_file:
        data = {
            key: in_file[key] for key in in_file.keys()
        }
    # Load or average
    intensities_skeleton_normals_avg = np.nanmean(
        data['intensities_skeleton_normals_full'][:, :, 15:46],
        axis=-1
    ) if 'intensities_skeleton_normals_full' in data else data['kymograph_skeleton_normals'].copy()
    # Find the gut boundary shifts
    weights = intensities_skeleton_normals_avg[:,il:ir]
    center_of_mass = np.nansum(
        weights * np.arange(ir-il)[np.newaxis,:], axis=1
    ) / np.nansum( weights, axis=1 )
    dindex_center_of_mass = np.round(
        (center_of_mass - np.nanmedian(center_of_mass))*scale_roll,
        0
    ).astype(np.int64)
    # Rolled
    intensities_skeleton_normals_avg_rolled = intensities_skeleton_normals_avg.copy()
    for idx, k in enumerate(dindex_center_of_mass):
        if np.isnan(k) or np.isinf(k) or k == 0:
            continue
        intensities_skeleton_normals_avg_rolled[idx] = np.roll(intensities_skeleton_normals_avg_rolled[idx], k)
    # Return
    return intensities_skeleton_normals_avg, intensities_skeleton_normals_avg_rolled

def process_kymograph_segment(idx_start, idx_end, indices_skeleton_reverse=set(), worm_id=None, condition=None, strain=None):
    """
    Process a video segment to generate kymographs and quality control videos.
    
    This function performs the complete analysis pipeline:
    1. Creates foreground mask videos
    2. Extracts worm masks and skeleton overlays
    3. Generates kymographs showing intensity changes over time
    4. Saves all data in multiple formats (NPZ, MAT, PNG)
    
    Args:
        idx_start: Starting frame index for processing
        idx_end: Ending frame index for processing
        indices_skeleton_reverse: Set of frame indices where skeleton orientation should be reversed
    """
    # Recording and writing parameters
    INDEX_START = idx_start
    INDICES_TO_CONSIDER = np.arange(idx_start, idx_end, 1)
    NT = len(INDICES_TO_CONSIDER)
    _PREFIX = f"{worm_id}_{condition}_{strain}_pumping_{str(idx_start).zfill(8)}_{str(idx_end).zfill(8)}_"
    FP_WRITE_VIDEO_FOREGROUND = os.path.join( FP_PUMPING_EXTRACTS, f"{_PREFIX}movie_foreground.mp4" )  # Path to the video file to write masked foreground
    FP_WRITE_VIDEO_WORMMASK = os.path.join( FP_PUMPING_EXTRACTS, f"{_PREFIX}movie_worm_mask.mp4" )  # Path to the video file to write worm mask video
    FP_WRITE_VIDEO_WORMSKELETON = os.path.join( FP_PUMPING_EXTRACTS, f"{_PREFIX}movie_worm_skeleton.mp4" )  # Path to the video file to write worm mask video
    FP_WRITE_IMAGE_KYMOGRAPH = os.path.join( FP_PUMPING_EXTRACTS, f"{_PREFIX}kymograph_only_center.png" )  # Path to the video file to write worm mask video
    FP_WRITE_IMAGE_KYMOGRAPH_SCALED = os.path.join( FP_PUMPING_EXTRACTS, f"{_PREFIX}kymograph_only_center_scaled.png" )  # Path to the video file to write worm mask video
    FP_WRITE_IMAGE_KYMOGRAPH_NORMALS = os.path.join( FP_PUMPING_EXTRACTS, f"{_PREFIX}kymograph_normals.png" )  # Path to the video file to write worm mask video
    FP_WRITE_IMAGE_KYMOGRAPH_NORMALS_SCALED = os.path.join( FP_PUMPING_EXTRACTS, f"{_PREFIX}kymograph_normals_scaled.png" )  # Path to the video file to write worm mask video
    FP_WRITE_NPZ_KYMOGRAPH_ALL = os.path.join( FP_PUMPING_EXTRACTS, f"{_PREFIX}kymograph_all.npz" )  # Path to all the data for the kymographs, e.g. timestamps and distances in `um`
    FP_WRITE_MAT_KYMOGRAPH_ALL = os.path.join( FP_PUMPING_EXTRACTS, f"{_PREFIX}kymograph_all.mat" )  # Path to all the data for the kymographs, e.g. timestamps and distances in `um`
    # 1) Make foreground mask video
    # Add frame index and actual FPS
    # series_beh = SerializeDatas([imgs.copy()])
    ## Mask images (lazy calculation -> nothing is calculated until writing it to file)
    def do_mask(img):
        img_mask = create_foreground_mask(img, threshold_foreground=MASK_THRESHOLD_FOREGROUND, threshold_area_min_object=MASK_AREA_MIN_OBJECT, threshold_area_max_object=MASK_AREA_MAX_OBJECT)
        return img_mask.astype(np.uint8)
    series_to_write = ImgToProcess(series_beh, fn_process=do_mask, rescale=True)
    ## White color
    series_to_write = AddFrameIndicesTexts(
        series_to_write,
        texts=texts,
        textOrigin=TEXT_ORIGIN_WHITE, color=TEXT_COLOR_WHITE
    )
    ## Black color -> for very bright images
    series_to_write = AddFrameIndicesTexts(
        series_to_write,
        texts=texts,
        textOrigin=TEXT_ORIGIN_BLACK, color=TEXT_COLOR_BLACK
    )
    ## Slice specific range
    series_to_write = ReIndexData(
        series_to_write,
        indices_new=INDICES_TO_CONSIDER
    )
    write_video(
        series_to_write,
        fp=FP_WRITE_VIDEO_FOREGROUND,
        fps=FRAMES_PER_SECOND,
        verbose=True
    )
    # 2) Find worm mask
    # Add frame index and actual FPS
    # series_beh = SerializeDatas([imgs.copy()])
    ## Find worm mask
    def do_mask(img):
        img_mask = create_foreground_mask(img, threshold_foreground=MASK_THRESHOLD_FOREGROUND, threshold_area_min_object=MASK_AREA_MIN_OBJECT, threshold_area_max_object=MASK_AREA_MAX_OBJECT)
        worm_mask = extract_worm_mask(img_mask, coords_center=WORM_CENTER_ESTIMATE, size_close=WORM_SIZE_CLOSE)
        return worm_mask.astype(np.uint8)
    series_to_write = ImgToProcess(series_beh, fn_process=do_mask, rescale=True)
    ## White color
    series_to_write = AddFrameIndicesTexts(
        series_to_write,
        texts=texts,
        textOrigin=TEXT_ORIGIN_WHITE, color=TEXT_COLOR_WHITE
    )
    ## Black color -> for very bright images
    series_to_write = AddFrameIndicesTexts(
        series_to_write,
        texts=texts,
        textOrigin=TEXT_ORIGIN_BLACK, color=TEXT_COLOR_BLACK
    )
    ## Slice specific range
    series_to_write = ReIndexData(
        series_to_write,
        indices_new=INDICES_TO_CONSIDER
    )
    write_video(
        series_to_write,
        fp=FP_WRITE_VIDEO_WORMMASK,
        fps=FRAMES_PER_SECOND,
        verbose=True
    )
    # 3) Find skeleton and video from it
    # Add frame index and actual FPS
    # series_beh = SerializeDatas([imgs.copy()])
    ## Find worm mask
    def do_mask(img, idx_frame):
        # These non-local variables are used: idx_start, infos
        # Get skeleton
        idx_skeleton = idx_frame - idx_start if idx_frame >= idx_start else 0  # Technical due to definition of class `ImgToProcessIndexed`
        skeleton = infos['skeleton_smooth'][idx_skeleton]
        # Overlay image
        img_overlayed = cv.polylines( img.astype(np.uint8), [ skeleton.astype(np.int32)[:,np.newaxis,:] ], False, SKELETON_LINE_COLOR, SKELETON_LINE_WIDTH )  # Aesthetics: width and color of the skeleton
        # Head part
        img_overlayed = cv.polylines( img_overlayed, [ skeleton.astype(np.int32)[:SKELETON_HEAD_POINTS,np.newaxis,:] ], False, SKELETON_HEAD_LINE_COLOR, SKELETON_HEAD_LINE_WIDTH )  # Aesthetics: width and color of the skeleton
        return img_overlayed
    # Find skeleton
    infos = {
        'indices': INDICES_TO_CONSIDER,
        'skeleton': np.zeros((NT, SKELETON_N_SEGMENTS, 2), dtype=np.float64)*np.nan,
    }
    for idx, idx_frame in enumerate(tqdm(INDICES_TO_CONSIDER)):
        # Load
        img = series_beh[idx_frame]
        # Mask
        img_mask = create_foreground_mask(img, threshold_foreground=MASK_THRESHOLD_FOREGROUND, threshold_area_min_object=MASK_AREA_MIN_OBJECT, threshold_area_max_object=MASK_AREA_MAX_OBJECT)
        worm_mask = extract_worm_mask(img_mask, coords_center=WORM_CENTER_ESTIMATE, size_close=WORM_SIZE_CLOSE)
        # Skeleton
        skeleton_prev = np.zeros(0) if idx == 0 or np.any(np.isnan(infos['skeleton'][idx-1])) else infos['skeleton'][idx-1]  # Use last skeleton if it was found, to re-orient and keep orientation consistency
        skeleton = extract_skeleton(
            worm_mask,
            skeleton_prev=skeleton_prev,
            n_skeleton_segments=SKELETON_N_SEGMENTS
        )
        ## Reverse initial skeleton
        if idx_frame in indices_skeleton_reverse:
            print(f"*** skeleton reversed at {idx_frame}")
            skeleton = skeleton[::-1]
        if len(skeleton) == 0:
            continue
        # Intensities
        infos['skeleton'][idx] = skeleton
    # Smooth Skeletons temporaly
    infos['skeleton_smooth'] = infos['skeleton'].copy()
    infos['skeleton_smooth'][1:] += infos['skeleton'][:-1]
    infos['skeleton_smooth'][:-1] += infos['skeleton'][1:]
    infos['skeleton_smooth'][:1] /= 2
    infos['skeleton_smooth'][1:-1] /= 3
    infos['skeleton_smooth'][-1:] /= 2
    # Write video
    series_to_write = ImgToProcessIndexed(series_beh, fn_process=do_mask, rescale=True)
    ## White color
    series_to_write = AddFrameIndicesTexts(
        series_to_write,
        texts=texts,
        textOrigin=TEXT_ORIGIN_WHITE, color=TEXT_COLOR_WHITE
    )
    ## Black color -> for very bright images
    series_to_write = AddFrameIndicesTexts(
        series_to_write,
        texts=texts,
        textOrigin=TEXT_ORIGIN_BLACK, color=TEXT_COLOR_BLACK
    )
    ## Slice specific range
    series_to_write = ReIndexData(
        series_to_write,
        indices_new=INDICES_TO_CONSIDER
    )
    write_video(
        series_to_write,
        fp=FP_WRITE_VIDEO_WORMSKELETON,
        fps=FRAMES_PER_SECOND,
        verbose=True
    )
    # 4) Create the kymograph
    # Interpolated
    dists_to_interpolate_px = SKELETON_INTENSITY_STEPS_UM / UM_PER_PIXEL
    _n = len(dists_to_interpolate_px)
    infos['skeleton_smooth_interp'] = infos['skeleton_smooth'].copy()
    infos['skeleton_smooth_interp_um'] = np.zeros((NT, _n, 2))*np.nan
    infos['skeleton_smooth_interp_um_normals'] = np.zeros((NT, _n, 2))*np.nan
    infos['intensities_skeleton'] = np.zeros((NT, _n))*np.nan
    infos['intensities_skeleton_normals'] = np.zeros((NT, _n-1))*np.nan
    infos['intensities_skeleton_normals_full'] = np.zeros((NT, _n-1, N_SKELETON_NORMALS_HALF_LENGTH_PIXELS))*np.nan
    infos['widths_skeleton_normals'] = np.zeros((NT, _n-1))*np.nan
    # Load intensities along lines
    for idx, idx_frame in enumerate(tqdm(infos['indices'])):
        # Skeleton Smoothed
        skeleton = infos['skeleton_smooth'][idx]
        if np.any(np.isnan(skeleton)):
            continue
        # Smooth spatially
        skeleton_interp = smooth_skeleton( skeleton, stride=5, k=3 )
        skeleton_interp_um = interpolate_skeleton(skeleton_interp, dists_to_interpolate_px, k=3 )
        skeleton_interp_um_normals = compute_skeleton_normals(
            skeleton_interp_um,
            normal_factors=np.arange(-SKELETON_NORMALS_HALF_LENGTH_PIXELS, SKELETON_NORMALS_HALF_LENGTH_PIXELS+1).astype(np.float32)
        )
        skeleton_interp_um_normals = np.minimum(
                np.maximum(
                skeleton_interp_um_normals, 0
            ),
            NX-1
        ).astype(np.int64)
        infos['skeleton_smooth_interp'][idx] = skeleton_interp
        infos['skeleton_smooth_interp_um'][idx] = skeleton_interp_um
        skeleton_int = np.minimum(
            np.maximum(
                skeleton_interp_um, 0
            ),
            NX-1
        ).astype(np.int64)
        # Load
        img = series_beh[idx_frame]
        # Mask
        img_mask = create_foreground_mask(img, threshold_foreground=MASK_THRESHOLD_FOREGROUND, threshold_area_min_object=MASK_AREA_MIN_OBJECT, threshold_area_max_object=MASK_AREA_MAX_OBJECT)
        worm_mask = extract_worm_mask(img_mask, coords_center=WORM_CENTER_ESTIMATE, size_close=WORM_SIZE_CLOSE)
        # NaNed
        img_naned = img.astype(np.float32)
        img_naned[~worm_mask] = np.nan
        # Intensities
        infos['intensities_skeleton'][idx] = img_naned[skeleton_int[:,1], skeleton_int[:,0]]
        infos['intensities_skeleton_normals'][idx] = np.nanmean(
            img_naned[skeleton_interp_um_normals[:,1], skeleton_interp_um_normals[:,0]],
            axis=-1
        )
        infos['widths_skeleton_normals'][idx] = np.sum(
            ~np.isnan(img_naned[skeleton_interp_um_normals[:,1], skeleton_interp_um_normals[:,0]]),
            axis=-1
        )
        infos['intensities_skeleton_normals_full'][idx] = img_naned[skeleton_interp_um_normals[:,1], skeleton_interp_um_normals[:,0]]
    # Store
    ## Numpy
    np.savez_compressed(
        FP_WRITE_NPZ_KYMOGRAPH_ALL,
        timestamps = times_beh[INDICES_TO_CONSIDER],
        kymograph_skeleton = infos['intensities_skeleton'],
        kymograph_skeleton_normals = infos['intensities_skeleton_normals'],
        intensities_skeleton_normals_full = infos['intensities_skeleton_normals_full'],
        distance_from_nose = SKELETON_INTENSITY_STEPS_UM,
        normal_scales_pixel = np.arange(-SKELETON_NORMALS_HALF_LENGTH_PIXELS, SKELETON_NORMALS_HALF_LENGTH_PIXELS+1).astype(np.float32),
    )
    ## Matlab
    savemat(
        FP_WRITE_MAT_KYMOGRAPH_ALL,
        dict(
            timestamps = times_beh[INDICES_TO_CONSIDER],
            kymograph_skeleton = infos['intensities_skeleton'],
            kymograph_skeleton_normals = infos['intensities_skeleton_normals'],
            distance_from_nose = SKELETON_INTENSITY_STEPS_UM,
            normal_scales_pixel = np.arange(-SKELETON_NORMALS_HALF_LENGTH_PIXELS, SKELETON_NORMALS_HALF_LENGTH_PIXELS+1).astype(np.float32),
        )
    )
    #
    _ = gc.collect()
    def plotting_function(kymograph, fp_write, title = None):
        plt.ioff()
        plt.figure(figsize=FIGURE_SIZE_LARGE, dpi=FIGURE_DPI)
        plt.title(title)
        plt.imshow( kymograph, cmap='gray')
        # X ticks
        _ticks = np.arange(NT)
        _labels = (1000*(times_beh[INDICES_TO_CONSIDER] - times_beh[INDICES_TO_CONSIDER[0]]).round(5) ).astype(np.int64)
        plt.xticks(ticks=_ticks[::TICK_INTERVAL_TIME], labels=_labels[::TICK_INTERVAL_TIME], rotation=90)
        plt.xlabel("Milliseconds")
        # Y ticks
        _n = kymograph.shape[0]
        print(_n)
        _ticks = np.arange(len(dists_to_interpolate_px))[:_n]
        _labels = ( dists_to_interpolate_px * UM_PER_PIXEL ).round(1)[-_n:]
        plt.yticks(ticks=_ticks[::TICK_INTERVAL_DISTANCE], labels=_labels[::TICK_INTERVAL_DISTANCE])
        plt.ylabel("Distance from nose (um)")
        plt.savefig(fp_write, bbox_inches='tight')
        plt.clf()
        plt.cla()
        plt.close('all')
        _ = gc.collect()
        return
    ############################################
    plotting_function(
        kymograph = infos['intensities_skeleton'].T[:,:],
        fp_write = FP_WRITE_IMAGE_KYMOGRAPH,
        title = "Kymograph of intensity along the body"
    )
    plotting_function(
        kymograph = infos['intensities_skeleton_normals'].T[:,:],
        fp_write = FP_WRITE_IMAGE_KYMOGRAPH_NORMALS,
        title = "Kymograph of intensity along normals along the body"
    )
    ############################################
    # Normalize the kymograph?!
    _tmp = infos['intensities_skeleton'].T.copy()
    _tmp = _tmp[:, :]
    _tmp = np.clip(
        128*(_tmp / np.nanmean(_tmp, axis=0, keepdims=True))**1.5,
        0, 255
    )
    
    plt.ioff()
    plt.figure(figsize=FIGURE_SIZE_XLARGE, dpi=FIGURE_DPI)
    plt.title("Kymograph of intensity along the body - Rescaled")
    plt.imshow( _tmp, cmap='gray')
    # X ticks
    _ticks = np.arange(NT)
    _labels = (1000*(times_beh[INDICES_TO_CONSIDER] - times_beh[INDICES_TO_CONSIDER[0]]).round(5) ).astype(np.int64)
    plt.xticks(ticks=_ticks[::TICK_INTERVAL_TIME], labels=INDICES_TO_CONSIDER[::TICK_INTERVAL_TIME], rotation=90)
    plt.xlabel("Milliseconds")
    # Y ticks
    _ticks = np.arange(len(dists_to_interpolate_px))[:100]
    _labels = ( dists_to_interpolate_px * UM_PER_PIXEL ).round(1)[:100]
    plt.yticks(ticks=_ticks[::TICK_INTERVAL_DISTANCE], labels=_labels[::TICK_INTERVAL_DISTANCE])
    plt.ylabel("Distance from nose (um)")
    plt.savefig(FP_WRITE_IMAGE_KYMOGRAPH_SCALED, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close('all')
    ############################################
    # Normalize the kymograph?!
    _tmp = infos['intensities_skeleton_normals'].T.copy()
    _tmp = _tmp[:, :]
    _tmp = np.clip(
        128*(_tmp / np.nanmean(_tmp, axis=0, keepdims=True))**1.5,
        0, 255
    )

    plt.ioff()
    plt.figure(figsize=FIGURE_SIZE_XLARGE, dpi=FIGURE_DPI)
    plt.title("Kymograph of intensity along the body - Rescaled")
    plt.imshow( _tmp, cmap='gray')
    # X ticks
    _ticks = np.arange(NT)
    _labels = (1000*(times_beh[INDICES_TO_CONSIDER] - times_beh[INDICES_TO_CONSIDER[0]]).round(5) ).astype(np.int64)
    plt.xticks(ticks=_ticks[::TICK_INTERVAL_TIME], labels=INDICES_TO_CONSIDER[::TICK_INTERVAL_TIME], rotation=90)
    plt.xlabel("Milliseconds")
    # Y ticks
    _ticks = np.arange(len(dists_to_interpolate_px))[:100]
    _labels = ( dists_to_interpolate_px * UM_PER_PIXEL ).round(1)[:100]
    plt.yticks(ticks=_ticks[::TICK_INTERVAL_DISTANCE], labels=_labels[::TICK_INTERVAL_DISTANCE])
    plt.ylabel("Distance from nose (um)")
    plt.savefig(FP_WRITE_IMAGE_KYMOGRAPH_NORMALS_SCALED, bbox_inches='tight')
    plt.clf()
    plt.cla()
    plt.close('all')
    _ = gc.collect()


#######################################################################
# Main Function
# 
# This function is the entry point for the script. It loads the files,
# processes the kymographs, and saves the results.
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
    
    print(f"Starting kymograph generation for data in: {FP_READ_FOLDER}")
    
    # Expected folder and files structure
    # FP_READ_FOLDER
    # |-> WORM1_behavior
    # |   |-> 000.h5
    # |   |-> 001.h5
    # |-> WORM2_behavior
    # |   |-> 000.h5
    # ...
    # Run
    files_beh, _, times_beh, series_beh = load_files_data_times_persistent(
        fp_folder=os.path.join( FP_READ_FOLDER, "*_behavior" ),
        fp_folder_persistence=os.path.join( FP_READ_FOLDER, "metadata" )
    )
    _, NX, NY = series_beh.shape
    # Annotation texts
    _n = len(times_beh)
    texts = [ "" ]
    for idx in range(1,_n):
        _T = times_beh[idx]-times_beh[0]
        _fps = 1/(times_beh[idx]-times_beh[idx-1])
        texts.append(
            ", {:>6.2f}s [fps: {:>4.1f}]".format(_T, _fps)
        )
    print(f"Behavior Series Loaded: records={len(series_beh)}")


    # Figure out files frame counts
    ns_per_recording = dict()
    recording_states = list()
    for file, n in zip(files_beh, series_beh.ns):
        name_recording = file.filename.split(os.sep)[-2]
        worm_id, condition, strain = name_recording.split("_")[:3]
        if name_recording not in ns_per_recording:
            ns_per_recording[name_recording] = n
            recording_states.append((worm_id, condition, strain))
        else:
            ns_per_recording[name_recording] += n
    indices_per_recording = np.array([0] + [
        ns_per_recording[k] for k in sorted(ns_per_recording)
    ])
    indices_per_recording = np.cumsum(indices_per_recording)
    print(list(zip( indices_per_recording[:-1], indices_per_recording[1:] )))

    # Manual Head-Tail confusion fixes
    # Possible good candidate that might have pumping as well
    from collections import defaultdict

    ## Intervals to extrct pumping from
    # e.g. this is by default the whole recording
    indices_inbetween_pairs = list(zip( indices_per_recording[:-1], indices_per_recording[1:] ))
    print(indices_inbetween_pairs)

    ## Indices where skeleton head-tail is confused
    ## Make sure you only add indices once, and only do it once for each interval
    # e.g. the head-tail remains consistent until the worm bends or the skeleton is not extractable by tierpsy
    # after the skeleton is again calculable, it might get head and tail confused.
    indices_skeleton_reverse = defaultdict(set)
    indices_skeleton_reverse[(0, 400)] = {0,}
    indices_skeleton_reverse[(400, 800)] = {400,}
    # indices_skeleton_reverse[(0, len(times_beh))] = set()  # Add frame indices from the annotated video where the confusion starts and it will be flipped for the following frames as well
    # E.g. indices_skeleton_reverse[(0, 5956)] = { 1772, 2567, 4428, }
    # This means that the skeleton is reversed at frames 1772, 2567, 4428.
    # MANUAL PART! -> you need to add the indices manually if needed.

    # Make videos, kymographs and store data for all
    for (idx_start, idx_end), (worm_id, condition, strain) in zip(indices_inbetween_pairs,recording_states):
        print(f"##### {datetime.datetime.now()}  Interval processing: {str(idx_start).zfill(8)}_{str(idx_end).zfill(8)}")
        # Extract all
        key = (idx_start, idx_end)
        ## Create Paths
        fp_read_pumping_kymograph = os.path.join(
            FP_PUMPING_EXTRACTS,
            f"{worm_id}_{condition}_{strain}_pumping_{str(idx_start).zfill(8)}_{str(idx_end).zfill(8)}_kymograph_all.npz"
        )
        fp_write_pumping_kymograph_png = os.path.join(
            FP_PUMPING_EXTRACTS,
            f"{worm_id}_{condition}_{strain}_pumping_{str(idx_start).zfill(8)}_{str(idx_end).zfill(8)}_kymograph_normals_true.png"
        )
        # Skip if exists
        if os.path.exists(fp_write_pumping_kymograph_png):
            continue
        # Report
        print("####"*20)
        print(f"####  {datetime.datetime.now()} Procesing interval: {str(idx_start).zfill(8)}-{str(idx_end).zfill(8)}")
        print("####"*20)
        # Do the thing
        process_kymograph_segment(idx_start, idx_end, indices_skeleton_reverse=indices_skeleton_reverse[key], worm_id=worm_id, condition=condition, strain=strain )
        print("####"*20)
        print(f"####  {datetime.datetime.now()} Finished!")
        print("####"*20)
        # Load Kymograph convert and write
        with np.load(fp_read_pumping_kymograph) as file_kymograph:
            # Load & Convert
            food_entry_normals_kymograph_uint8 = file_kymograph['kymograph_skeleton_normals']
            food_entry_normals_kymograph_uint8[np.isnan(food_entry_normals_kymograph_uint8)] = 255.0
            food_entry_normals_kymograph_uint8 = np.clip(food_entry_normals_kymograph_uint8, 0.0, 255.0).astype(np.uint8)
            ## Resize
            nx, ny = food_entry_normals_kymograph_uint8.shape
            food_entry_normals_kymograph_uint8 = cv.resize( food_entry_normals_kymograph_uint8, (ny, nx*IMAGE_RESIZE_FACTOR) )
            # Write
            plt.imsave(
                fp_write_pumping_kymograph_png,
                food_entry_normals_kymograph_uint8.T, cmap='gray',
                vmin=np.floor(np.nanmin(food_entry_normals_kymograph_uint8)), vmax=np.ceil(np.nanmax(food_entry_normals_kymograph_uint8))
            )
    
    # Load all NPZ files and create rolled kymographs
    fps_cases = sorted(glob(
        os.path.join(FP_PUMPING_EXTRACTS, "*.npz")
    ))
    print(f"Found {len(fps_cases)} cases")
    if len(fps_cases) == 0:
        raise ValueError(f"No NPZ files found in {FP_PUMPING_EXTRACTS}")

    # Generate aligned kymograph images for manual annotation
    for fp_npz in tqdm(fps_cases):
        # File path
        filename = fp_npz.split(os.sep)[-1]
        fp_write_png = os.path.join(
            FP_PUMPING_ANALYSIS,
            f"{filename[:-4]}.png"
        )
        # Skip if exists
        if os.path.exists(fp_write_png):
            continue
        # Load
        intensities_skeleton_normals_avg, intensities_skeleton_normals_avg_rolled = load_and_align_kymograph(fp_npz)
        # Store
        plt.imsave(
            fp_write_png,
            intensities_skeleton_normals_avg_rolled.T,
            cmap='gray',
            vmin=np.nanquantile( intensities_skeleton_normals_avg_rolled, 0.01 ),
            vmax=np.nanquantile( intensities_skeleton_normals_avg_rolled, 0.99 )
        )

