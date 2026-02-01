#######################################################################
# Utility functions for C. elegans behavior analysis
# 
# This module provides core functionality for:
# - Loading and processing video data from HDF5 files
# - Creating videos with overlays for quality control
# - Managing large datasets with persistent caching
# 
# Key classes:
# - SerializeDatas: Efficiently handles multiple video files as a single dataset
# - ImgToProcess: Applies image processing functions to video frames
# - AddFrameIndicesTexts: Adds frame information overlays to videos
#######################################################################
# Modules
import cv2 as cv
from glob import glob
from tqdm import tqdm
import numpy as np
from h5py import File as h5File
import os
import shelve

#######################################################################
# Video Processing Functions
#######################################################################

def write_video(data, fp='tmp.mp4', fps=20, verbose=False):
    """
    Write a sequence of images to an MP4 video file.
    
    Args:
        data: 3D numpy array (time, height, width) of grayscale images
        fp: Output file path for the video
        fps: Frames per second for the output video
        verbose: Show progress bar during video writing
    """
    video_writer = cv.VideoWriter(
        fp,
        cv.VideoWriter_fourcc(*'mp4v'),
        fps,
        data.shape[1:][::-1],
        False
    )
    iterable = range(len(data))
    if verbose:
        iterable = tqdm(iterable)
    for t in iterable:
        video_writer.write(data[t])
    video_writer.release()
    return

#######################################################################
# Data Loading Functions
#######################################################################

def load_files_data_times(fp_folder):
    """
    Load all HDF5 video files from a folder and extract data and timestamps.
    
    Args:
        fp_folder: Path to folder containing .h5 files
        
    Returns:
        files: List of open HDF5 file objects
        datas: List of data arrays (video frames)
        times: List of timestamp arrays
    """
    files = []
    for fp in sorted(glob(os.path.join(fp_folder, "*.h5"))):
        try:
            files.append(h5File(fp))
        except Exception as e:
            print(f"Error in loading file:\n--->{fp}\n###\n{str(e)}\n###")
    datas = [file['data'] for file in files]
    times = [file['times'] for file in files]
    return files, datas, times


def load_files_data_times_persistent(fp_folder, fp_folder_persistence, overwrite=False):
    """
    Load video data with persistent caching to avoid reloading large files.
    
    This function creates a cache of metadata (file shapes and timestamps) 
    to speed up subsequent runs. The actual video data is loaded fresh each time.
    
    Args:
        fp_folder: Path to folder containing .h5 files
        fp_folder_persistence: Path to folder for storing cache files
        overwrite: If True, rebuild the cache even if it exists
        
    Returns:
        files: List of open HDF5 file objects
        datas: List of data arrays (video frames)  
        times: Concatenated timestamp array from all files
        series_beh: SerializeDatas object for efficient data access
    """
    # Create cache folder if it doesn't exist
    if not os.path.exists(fp_folder_persistence):
        os.makedirs(fp_folder_persistence)
    
    # Connect to persistent cache (shelve database)
    fp_shelf_metadata = os.path.join(fp_folder_persistence, "series_metadata.shelf")
    shelf_metadata = shelve.open(fp_shelf_metadata, writeback=False)  # Read about shelve in the documentation: https://docs.python.org/3/library/shelve.html
    
    # Load all video files
    files, datas, times = load_files_data_times(fp_folder)
    
    # Cache metadata for each file
    for file, data, time in zip(files, datas, times):
        fp = file.filename
        # Skip if already cached and not overwriting
        if fp in shelf_metadata and not overwrite:
            continue
        # Store metadata in cache
        entry = {
            'data_shape': data.shape,
            'time_list': [float(t) for t in time[:]],
        }
        shelf_metadata[fp] = entry
    
    # Concatenate timestamps from all files
    times = np.concatenate([
        shelf_metadata[file.filename]['time_list'] for file in files
    ])
    ns = [
        shelf_metadata[file.filename]['data_shape'][0] for file in files
    ]
    
    # Clean up cache
    shelf_metadata.sync()
    shelf_metadata.close()
    # Return
    return files, datas, times, SerializeDatas(datas, ns=ns)

#######################################################################
# Data Processing Classes
#######################################################################

class ReIndexData:
    """
    Re-indexes a dataset to select specific frames by their indices.
    
    This is useful for processing only a subset of frames from a video,
    such as a specific time window for analysis.
    """
    def __init__(self, data, indices_new):
        self.data = data
        self.indices_new = indices_new
        self.shape = (len(self.indices_new), *self.data.shape[1:])
        return
    
    def __len__(self):
        return len(self.indices_new)
    
    def __getitem__(self, t):
        idx = self.indices_new[t]
        return self.data[idx]
class ImgToProcess:
    """
    Applies a processing function to each frame of a video dataset.
    
    This class allows you to apply image processing operations (like masking,
    filtering, or skeleton extraction) to entire video sequences efficiently.
    """
    def __init__(self, data, fn_process, rescale=False):
        self.data = data
        self.fn_process = fn_process
        self.rescale = rescale
        self.shape = self.data.shape
        # Determine output shape from first frame
        _shape_frame = self.fn_process(self.data[0]).shape
        self.shape = (self.shape[0], *_shape_frame)
        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, t):
        result = self.fn_process(self.data[t]).astype(np.uint8)
        # Optional intensity rescaling to 0-255 range
        if self.rescale and result.max() != 0:
            result *= (255//result.max())
        return result
class ImgToProcessIndexed:
    """
    Similar to ImgToProcess, but the processing function receives both the frame
    and its index. This is useful when processing depends on frame position.
    """
    def __init__(self, data, fn_process, rescale=False):
        self.data = data
        self.fn_process = fn_process
        self.rescale = rescale
        self.shape = self.data.shape
        # Determine output shape from first frame
        _shape_frame = self.fn_process(self.data[0], 0).shape
        self.shape = (self.shape[0], *_shape_frame)
        return
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, t):
        result = self.fn_process(self.data[t], t).astype(np.uint8)
        # Optional intensity rescaling to 0-255 range
        if self.rescale and result.max() != 0:
            result *= (255//result.max())
        return result
class SerializeDatas:
    """
    Efficiently handles multiple video files as a single continuous dataset.
    
    This class concatenates multiple HDF5 files into a virtual single dataset,
    allowing seamless access across file boundaries. This is essential for
    analyzing long recordings that span multiple files.
    """
    def __init__(self, data_list, ns=None):
        self.data_list = data_list
        self.shape = self.data_list[0].shape
        # Number of frames in each file
        self.ns = [len(x) for x in self.data_list] if ns is None else ns.copy()
        self.n = sum(self.ns)  # Total number of frames
        self.shape = (self.n, *self.shape[1:])  # Update shape to total frames
        return
    
    def __len__(self):
        return self.n
    
    def __getitem__(self, t):
        # Find which file contains frame t
        idx = 0
        while t >= self.ns[idx]:
            t -= self.ns[idx]
            idx += 1
        return self.data_list[idx][t]
class AddFrameIndicesTexts:
    """
    Adds text overlays to video frames showing frame numbers and timing information.
    
    This is useful for creating quality control videos where you need to verify
    frame timing and identify specific events in the recording.
    """
    def __init__(self, data, idx_offset=0, texts=None, indices=None, 
                 fontScale=0.5, textOrigin=(25, 25), color=(255, 0, 0)):
        self.data = data
        self.shape = self.data.shape
        self.idx_offset = idx_offset
        self.texts = texts
        self.indices = indices
        # Adjust shape if using specific frame indices
        if self.indices is not None:
            self.shape = (len(self.indices), self.shape[1], self.shape[2])
        self.fontScale = fontScale
        self.textOrigin = textOrigin
        self.color = color
        return
    
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, t):
        idx_frame = self.idx_offset + t
        res = self.data[t] if self.indices is None else self.data[self.indices[t]]
        
        # Add text overlay
        font = cv.FONT_HERSHEY_SIMPLEX
        thickness = 1
        
        # Create frame information text
        txt = "Frame IDX: {:>7}".format(idx_frame)
        if self.texts is not None:
            txt_extra = self.texts[t]
            txt = "{}{}".format(txt, txt_extra)
        
        # Draw text on image
        res = cv.putText(
            res, txt, self.textOrigin, font,
            self.fontScale, self.color, thickness, cv.LINE_AA
        )
        return res
