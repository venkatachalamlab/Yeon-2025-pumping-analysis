#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 25 17:17:38 2021

@author: lferiani
"""

import tqdm
import shutil
from pathlib import Path
from matplotlib import pyplot as plt

from tierpsy.analysis.split_fov.FOVMultiWellsSplitter import (
    FOVMultiWellsSplitter, process_image_from_name)

from tierpsy import DFLT_SPLITFOV_PARAMS_PATH, DFLT_SPLITFOV_PARAMS_FILES
from tierpsy.helper.params.tracker_param import SplitFOVParams
from tierpsy.analysis.compress.selectVideoReader import selectVideoReader


def test_from_raw():

    # where are things
    wd = Path('~/Hackathon/multiwell_tierpsy/12_FEAT_TIERPSY/').expanduser()
    raw_fname = (
        wd / 'RawVideos' / '20191205' /
        'syngenta_screen_run1_bluelight_20191205_151104.22956805' /
        'metadata.yaml'
        )
    masked_fname = Path(
        str(raw_fname)
        .replace('RawVideos',  'MaskedVideos_')
        .replace('.yaml',  '.hdf5')
        )

    masked_fname.parent.mkdir(parents=True, exist_ok=True)

    json_fname = Path(DFLT_SPLITFOV_PARAMS_PATH) / 'HYDRA_96WP_UPRIGHT.json'

    splitfov_params = SplitFOVParams(json_file=json_fname)
    shape, edge_frac, sz_mm = splitfov_params.get_common_params()
    uid, rig, ch, mwp_map = splitfov_params.get_params_from_filename(
        masked_fname)
    px2um = 12.4

    # read image
    vid = selectVideoReader(str(raw_fname))
    status, img = vid.read_frame(0)

    fovsplitter = FOVMultiWellsSplitter(
        img,
        microns_per_pixel=px2um,
        well_shape=shape,
        well_size_mm=sz_mm,
        well_masked_edge=edge_frac,
        camera_serial=uid,
        rig=rig,
        channel=ch,
        wells_map=mwp_map)
    fig = fovsplitter.plot_wells()

    with open(masked_fname, 'w') as fid:
        pass
    fovsplitter.write_fov_wells_to_file(masked_fname)
    shutil.rmtree(masked_fname.parent)

    return


def test_from_new_fov_wells():
    masked_fname = Path(
        '/Users/lferiani/Hackathon/multiwell_tierpsy/12_FEAT_TIERPSY/'
        'MaskedVideos/20191205/'
        'syngenta_screen_run1_bluelight_20191205_151104.22956805/metadata.hdf5'
        )

    fs_from_wells = FOVMultiWellsSplitter(masked_fname)
    fs_from_wells.plot_wells()
    return


def test_from_old_fov_wells():
    # test from masked video with old /fov_wells
    # when building from wells, no need for json
    masked_fname = Path(
        '/Users/lferiani/Hackathon/multiwell_tierpsy/12_FEAT_TIERPSY/'
        '_MaskedVideos/20191205/'
        'syngenta_screen_run1_bluelight_20191205_151104.22956805/metadata.hdf5'
        )
    fs_from_old_wells = FOVMultiWellsSplitter(masked_fname)
    fs_from_old_wells.plot_wells()


def test_from_imgs():
    json_fname = Path(DFLT_SPLITFOV_PARAMS_PATH) / 'HYDRA_96WP_UPRIGHT.json'
    wd = Path(
        '/Volumes/behavgenom$/Luigi/Data/'
        'LoopBio_calibrations/wells_mapping/20190710/')
    img_dir = wd
    fnames = list(img_dir.rglob('*.png'))
    fnames = [str(f) for f in fnames if '_wells' not in str(f)]

    plt.ioff()
    for fname in tqdm.tqdm(fnames):
        process_image_from_name(fname, json_fname)
    plt.ion()


if __name__ == '__main__':

    test_from_raw()
    test_from_new_fov_wells()
    test_from_old_fov_wells()
    # test_from_imgs()