#######################################################################
# Configuration file for C. elegans pumping rate analysis
# 
# This file contains all the parameters needed to analyze pharyngeal pumping
# behavior in C. elegans videos. The parameters are organized by function
# and includes explanations for biologists familiar with programming.
#
# Key concepts:
# - Kymographs: time-space plots showing intensity changes along the worm's body
# - Skeleton: a series of points tracing the worm's centerline from head to tail
# - Normals: perpendicular lines to the skeleton used to measure body width
# - Pumping: rhythmic contractions of the pharynx (feeding organ)
#######################################################################

import os
import numpy as np

#######################################################################
# Imaging Parameters
# 
# These parameters define the microscope setup and video recording properties.
# They are crucial for converting pixel measurements to real-world distances
# and for proper temporal analysis of pumping events.
#######################################################################

# Calibration parameters - convert pixels to micrometers
PIXELS_PER_MM = 410*2  # Microscope magnification: pixels per millimeter at the sample plane
UM_PER_PIXEL = 1000 / PIXELS_PER_MM  # Conversion factor: micrometers per pixel (used for skeleton analysis)

# Frame rates - temporal resolution of the recordings
FRAMES_PER_SECOND = 40.0  # Recording frame rate (Hz) - higher rates capture faster pumping events

#######################################################################
# File Paths
# 
# Directory structure for input and output files. The analysis expects
# specific folder organization with behavior recordings and metadata.
#######################################################################

# Default paths (can be overridden in scripts)
# MANUAL PART! -> you need to add the folder path manually.
# Set this to your data directory containing worm behavior recordings
FP_READ_FOLDER = "/path/to/your/worm_behavior_recordings"  # Root folder containing worm behavior recordings (*_behavior subfolders)
# Check that the path is set correctly
if FP_READ_FOLDER == "/path/to/your/worm_behavior_recordings":
    raise ValueError("Please update FP_READ_FOLDER in config.py to point to your actual data directory. Current value is a placeholder.")
# Create output folders
FP_PUMPING_EXTRACTS = os.path.join(FP_READ_FOLDER, "pumping_extracts")  # Output folder for kymographs and analysis results
os.makedirs(FP_PUMPING_EXTRACTS, exist_ok=True)

# Folder for annotated kymographs and pumping rate data
FP_PUMPING_ANALYSIS = os.path.join(FP_READ_FOLDER, "pumping_analysis")  # Folder for annotated kymographs and pumping rate data
os.makedirs(FP_PUMPING_ANALYSIS, exist_ok=True)

#######################################################################
# Foreground Masking Parameters
# 
# These parameters control the initial segmentation of the worm from the
# background. The algorithm identifies dark objects (worms) against bright
# backgrounds (agar surface).
#######################################################################

MASK_THRESHOLD_FOREGROUND = 110  # Pixel intensity threshold: values below this are considered worm (0-255 scale). Usually getting the worm perimeter is sufficient for things to work. As long as you can get an enclosed region at the end.
MASK_AREA_MIN_OBJECT = 20_000  # Minimum object size in pixels (filters out small debris and artifacts)
MASK_AREA_MAX_OBJECT = 500_000  # Maximum object size in pixels (filters out large artifacts or multiple worms). Usually this should not be needed if plate is clean and a single worm is visible.

#######################################################################
# Worm Masking Parameters
# 
# These parameters refine the worm segmentation by selecting the largest
# connected component (the worm) and filling gaps in the mask.
#######################################################################

WORM_SIZE_CLOSE = 11  # Morphological closing kernel size: fills gaps in worm outline (pixels)
WORM_CENTER_ESTIMATE = None  # Expected worm position (None = use image center). Set to np.array([x, y]) for off-center worms

#######################################################################
# Skeleton Parameters
# 
# The skeleton is a series of points tracing the worm's centerline from
# head to tail. These parameters control skeleton extraction and the
# creation of perpendicular lines (normals) for creating kymographs.
#######################################################################

SKELETON_N_SEGMENTS = 101  # Number of points along the skeleton (higher = more precise but slower)
SKELETON_INTENSITY_STEPS_UM = np.arange(0.0, 300.0, 1.0)  # Distance steps along skeleton in micrometers (0-300 μm from head)
SKELETON_NORMALS_HALF_LENGTH_PIXELS = 30  # Half-length of normal lines extending from skeleton (total width = 2*30+1 = 61 pixels)
N_SKELETON_NORMALS_HALF_LENGTH_PIXELS = 2*SKELETON_NORMALS_HALF_LENGTH_PIXELS+1  # Total number of points per normal line

#######################################################################
# Kymograph Annotation Parameters
# 
# These parameters control the manual annotation of pumping events in
# kymographs and the automated analysis of annotated data.
#######################################################################

# Annotation colors for manual marking in kymographs
KYMOGRAPH_ANNOTATION_COLOR_RED = np.array([237, 28, 36], dtype=np.float32)  # RGB color for marking pumping events
KYMOGRAPH_ANNOTATION_COLOR_GREEN = np.array([34, 177, 76], dtype=np.float32)  # RGB color for marking excluded regions

# Analysis window parameters for calculating pumping rates
PADDING_FOR_GREEN = 5  # Pixels to exclude around green annotations (frames to skip)
WINDOW_SIZE = 40*15  # Analysis window size: 15 seconds at 40 fps (600 frames)
WINDOW_SIZE_MIN = WINDOW_SIZE//2  # Minimum window size for valid analysis (7.5 seconds)

# Time intervals for analysis (in frames at 40 fps)
TIME_INTERVALS = {
    'first-60s': (0, 40*60),      # First 60 seconds of recording
    'second-60s': (40*60, 40*120), # Second 60 seconds of recording  
    'all-120s': (0, 40*120)        # Entire 120-second recording
}

#######################################################################
# Tierpsy Parameters
# 
# Tierpsy is a computer vision library for C. elegans analysis. These
# parameters control the skeleton extraction algorithm.
# You can follow the instructions on the github page to install Tierpsy: https://github.com/SinRas/tierpsy-tracker.
# And don't forget to cite the original repository: https://github.com/Tierpsy/tierpsy-tracker
#######################################################################

# Tierpsy-specific thresholds and parameters
TIERPSY_MIN_BLOB_AREA = 300  # Minimum blob size for skeleton extraction (pixels) - adjust for very small worms
TIERPSY_IS_LIGHT_BACKGROUND = True  # Background is brighter than the worm (standard brightfield microscopy)

#######################################################################
# Video Processing Parameters
# 
# These parameters control the visual overlay of skeleton information
# on videos for quality control and manual verification.
#######################################################################

# Video overlay parameters for skeleton visualization
SKELETON_LINE_WIDTH = 2  # Thickness of skeleton line overlay (pixels)
SKELETON_LINE_COLOR = 255  # Color of skeleton line (255 = white)
SKELETON_HEAD_LINE_WIDTH = 1  # Thickness of head region line (pixels)
SKELETON_HEAD_LINE_COLOR = 0  # Color of head region line (0 = black)
SKELETON_HEAD_POINTS = 20  # Number of skeleton points considered as "head" region

# Text overlay parameters for frame information
TEXT_ORIGIN_WHITE = (25, 25)  # Position for white text overlay (x, y pixels from top-left)
TEXT_COLOR_WHITE = 255  # White text color (for dark backgrounds)
TEXT_ORIGIN_BLACK = (25, 50)  # Position for black text overlay (x, y pixels from top-left)
TEXT_COLOR_BLACK = 0  # Black text color (for bright backgrounds)

#######################################################################
# Plotting Parameters
# 
# These parameters control the generation of kymograph images.
#######################################################################

# Figure parameters for kymograph generation
FIGURE_SIZE_LARGE = (64*2, 12)  # Size for detailed kymographs (width, height in inches)
FIGURE_SIZE_XLARGE = (128, 12)  # Size for high-resolution kymographs (width, height in inches)
FIGURE_DPI = 300  # Dots per inch for publication-quality images

# Tick parameters for axis labeling
TICK_INTERVAL_TIME = 20  # Show time labels every N frames
TICK_INTERVAL_DISTANCE = 20  # Show distance labels every N skeleton points

# Image processing parameters
IMAGE_RESIZE_FACTOR = 2  # Factor for resizing kymograph images (2 = double size)
