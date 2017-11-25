from DataProcessor import DataProcessor


import gdal
import glob
import json
import math
import os
import pickle
import rasterio
import rasterio.features
import re
import subprocess
import sys
import warnings
from logging import getLogger, Formatter, StreamHandler, INFO

import fiona
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import shapely.geometry
import shapely.ops
import shapely.wkt
import shapely.wkt
import skimage.morphology
import skimage.transform
import skimage.transform
import tables as tb
# from dir_traversal_tfrecord import tfrecord_auto_traversal
import tensorflow as tf
import tqdm
from pathlib import Path
from sklearn.datasets.base import Bunch

from DataProviderGSI import GISDataProvider
from QueueLoader import QueueLoader
from unet import unet



plugin_config = "./config/config.json"
type_of_data="multi"

analysis_image_list = "./config/imageInfo.csv"

def prep_osmlayer_train(dataprocessor, image_id):
    area_id = dataprocessor.directory_name_to_area_id(dataprocessor.DATA_PATH)
    prefix = dataprocessor.area_id_to_prefix(area_id)
    dataprocessor.logger.info("prep_osmlayer for {}".format(prefix))

    layers = dataprocessor.extract_osmlayers(area_id)

    df = pd.read_csv(analysis_image_list, index_col='ImageId')

    df_sz = len(df)
    fn_tif = dataprocessor.get_train_image_path_from_imageid(image_id, dataprocessor.DATA_PATH, mul=False)
    with rasterio.open(fn_tif, 'r') as fr:
        values = fr.read(1).astype(np.float32)
        fig, ax1 = plt.subplots(1, 2, sharey=True, figsize=(5, 5))
        cou = 0
        for layer_geoms in layers:
            mask = rasterio.features.rasterize(
                layer_geoms,
                out_shape=values.shape,
                transform=rasterio.guard_transform(
                    fr.transform))
            imageInfo1 = np.array(mask)
            drawrgbonly1 = imageInfo1
            ax1[cou].imshow(drawrgbonly1, aspect="auto")
            ax1[cou].xaxis.set_visible(False)
            ax1[cou].yaxis.set_visible(False)
            cou = cou + 1


print("end  of creating layers")
print("end  of creating layers")


def get_resized_8chan_image_train(dataprocessor, image_id):
    im = []
    area_id = dataprocessor.directory_name_to_area_id(dataprocessor.DATA_PATH)
    bs_rgb = dataprocessor.load_rgb_bandstats(area_id)
    bs_mul = dataprocessor.load_mul_bandstats(area_id)
    fn = dataprocessor.get_train_image_path_from_imageid(image_id, dataprocessor.DATA_PATH, mul=False)
    with rasterio.open(fn, 'r') as f:
        values = f.read().astype(np.float32)
        original = values
        original = np.swapaxes(original, 0, 2)
        original = np.swapaxes(original, 0, 1)
        fig1, ax1 = plt.subplots(1, 1, sharey=True, figsize=(5, 5))
        imageInfo = np.array(original)
        drawrgbonly = imageInfo
        ax1.imshow(drawrgbonly, aspect="auto")

        for chan_i in range(3):
            min_val = bs_rgb[chan_i]['min']
            max_val = bs_rgb[chan_i]['max']
            values[chan_i] = np.clip(values[chan_i], min_val, max_val)
            values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
            if(chan_i==2):
                values = np.swapaxes(values, 0, 2)
                values = np.swapaxes(values, 0, 1)
                fig, ax = plt.subplots(1, 1, sharey=True, figsize=(5, 5))
                imageInfo = np.array(values)
                drawrgbonly = imageInfo
                ax.imshow(drawrgbonly, aspect="auto")
                
def get_slice_8chan_im(dataprocessor, image_id):
    area_id = dataprocessor.directory_name_to_area_id(dataprocessor.DATA_PATH)
    bandstats = dataprocessor.load_mul_bandstats(area_id)
    fn = dataprocessor.get_train_image_path_from_imageid(image_id, dataprocessor.DATA_PATH, mul=True)
    with rasterio.open(fn, 'r') as f:
        values = f.read().astype(np.float32)
        for chan_i in range(8):
            min_val = bandstats[chan_i]['min']
            max_val = bandstats[chan_i]['max']
            values[chan_i] = np.clip(values[chan_i], min_val, max_val)
            values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
    values = np.swapaxes(values, 0, 2)
    values = np.swapaxes(values, 0, 1)
    assert values.shape == (650, 650, 8)

    fig, ax = plt.subplots(1, 1, sharey=True, figsize=(5, 5))
    imageInfo = np.array(values)
    drawrgbonly= imageInfo[:,:,[4,2,1]]
    ax.imshow(drawrgbonly, aspect="auto")
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    #
    fig, ax1 = plt.subplots(3, 3, sharey=True, figsize=(6, 6))
    indexx=-1
    for slice_pos in range(9):
        pos_j = int(math.floor(slice_pos / 3.0))
        pos_i = int(slice_pos % 3)
        if(pos_i==0):
            indexx=indexx+1
        x0 = dataprocessor.STRIDE_SZ * pos_i
        y0 = dataprocessor.STRIDE_SZ * pos_j
        im = values[x0:x0 + dataprocessor.INPUT_SIZE, y0:y0 + dataprocessor.INPUT_SIZE]
        assert im.shape == (256, 256, 8)

        imageInfo1 = np.array(im)
        drawrgbonly1 = imageInfo1[:, :, [4,2,1]]
        ax1[indexx, pos_i].imshow(drawrgbonly1, aspect="auto")
        ax1[indexx, pos_i].xaxis.set_visible(False)
        ax1[indexx, pos_i].yaxis.set_visible(False)



def _get_valtrain_mul_data(dataprocessor):
    area_id = dataprocessor.directory_name_to_area_id(dataprocessor.DATA_PATH)
    prefix = dataprocessor.area_id_to_prefix(area_id)
    fn_train = analysis_image_list
    df_train = pd.read_csv(fn_train)
    fn_im = dataprocessor.FMT_VALTRAIN_MUL_STORE.format(prefix)
    fn_mask = dataprocessor.FMT_VALTRAIN_MASK_STORE.format(prefix)
    with tb.open_file(fn_im, 'r') as f:
        with tb.open_file(fn_mask, 'r') as af:
            for idx, image_id in tqdm.tqdm(enumerate(df_train.ImageId.tolist())):
                im = np.array(f.get_node('/' + image_id))
                fig, ax = plt.subplots(1, 1, sharey=True, figsize=(5, 5))
                rgb_image = im[:, :, 0:3]
                imageInfo = np.array(rgb_image)
                drawrgbonly= imageInfo[:,:,0:3]
                ax.imshow(drawrgbonly, aspect="auto")

                mask = np.array(af.get_node('/' + image_id))
                mask = (mask > 0.5).astype(np.uint8)

                fig3, ax4 = plt.subplots(1, 1, sharey=True, figsize=(3, 3))
                ax4.imshow(mask, aspect="auto")


                get_resized_8chan_image_train(dataprocessor, image_id)
                get_slice_8chan_im(dataprocessor, image_id)
                prep_osmlayer_train(dataprocessor, image_id)





dataprocessor = DataProcessor(plugin_config, base_model_name="v1", model_name="v1", image_dir="v1")
dataprocessor.execute()
_get_valtrain_mul_data(dataprocessor)


plt.show()



