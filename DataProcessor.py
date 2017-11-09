from __future__ import print_function

import gdal
import json
import os
import sys
from logging import getLogger, Formatter, StreamHandler, INFO
from pathlib import Path
import subprocess
import glob
import warnings
import scipy
import re
import tqdm
import tables as tb
import pandas as pd
import numpy as np
import skimage.transform
import rasterio
import shapely.wkt
import matplotlib.pyplot as plt

from PIL import Image
import skimage.io as io
import tensorflow as tf
import skimage.transform
import skimage.morphology
import rasterio.features
import shapely.wkt
import shapely.ops
import shapely.geometry
# from dir_traversal_tfrecord import tfrecord_auto_traversal
import tensorflow as tf
import math

from scipy.stats import pearsonr
from sklearn.datasets.base import Bunch
import fiona
import pickle

from tf_unetnew.tf_unet import unet

import os
from DataProviderGSI import GISDataProvider
from QueueLoader import QueueLoader
plt.rcParams['image.cmap'] = 'gist_earth'
from collections import namedtuple


# Logger
from image_object import image_object

warnings.simplefilter("ignore", UserWarning)



class DataProcessor():
    def __init__(self, plugin_config, base_model_name="v5", model_name="v5", image_dir="v5", is_final=False):
        spacenet_util_dir = os.getcwd() + '/spaceNetUtilities'
        sys.path.extend([spacenet_util_dir])
        self.plugin_config_dir = plugin_config
        with open(plugin_config) as plugin_config:
            self.plugin_config = json.load(plugin_config)
            self.MODEL_NAME = model_name
            self.ORIGINAL_SIZE = 650
            self.STRIDE_SZ = 197
            self.ROOT_DIR = str(self.plugin_config["root_dir"])
            self.INPUT_SIZE =  int(self.plugin_config["width_of_image"])
            self.input_image_width = int(self.plugin_config["width_of_image"])
            self.input_image_height = int(self.plugin_config["height_of_image"])
            self.BASE_TRAIN_DIR = str(self.plugin_config["base_train_dir"])
            self.DATA_PATH = str(self.plugin_config["datapath"])
            self.DATA_PATH_TEST = str(self.plugin_config["datapath_test"])
            self.WORKING_DIR = str(self.plugin_config["working_dir"])
            self.IMAGE_DIR = self.ROOT_DIR + "/working/images/{}".format(image_dir)
            self.LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'
            self.BASE_DIR = self.ROOT_DIR + "/train"
            self.MODEL_DIR = self.ROOT_DIR + "/working/models/{}".format(self.MODEL_NAME)
            self.FN_SOLUTION_CSV = self.ROOT_DIR + "/output/{}.csv".format(self.MODEL_NAME)
            self.FMT_TESTPOLY_PATH = self.MODEL_DIR + "/{}_poly.csv"
            self.FN_SOLUTION_CSV = "/data/{}.csv".format(self.MODEL_NAME)
            self.FMT_TRAIN_SUMMARY_PATH = str(
                Path(self.BASE_TRAIN_DIR) /
                Path("{prefix:s}_Train/") /
                Path("summaryData/{prefix:s}_Train_Building_Solutions.csv"))
            self.FMT_TRAIN_RGB_IMAGE_PATH = str(
                Path("{datapath:s}/") /
                Path("RGB-PanSharpen/RGB-PanSharpen_{image_id:s}.tif"))
            self.FMT_TEST_RGB_IMAGE_PATH = str(
                Path("{datapath:s}/") /
                Path("RGB-PanSharpen/RGB-PanSharpen_{image_id:s}.tif"))
            self.FMT_TRAIN_MSPEC_IMAGE_PATH = str(
                Path("{datapath:s}/") /
                Path("MUL-PanSharpen/MUL-PanSharpen_{image_id:s}.tif"))
            self.FMT_TEST_MSPEC_IMAGE_PATH = str(
                Path("{datapath:s}/") /
                Path("MUL-PanSharpen/MUL-PanSharpen_{image_id:s}.tif"))
            self.FMT_RGB_BANDCUT_TH_PATH = self.IMAGE_DIR + "/rgb_bandcut{}.csv"
            self.BASE_IMAGE_DIR = self.ROOT_DIR + "/working/images/{}".format(base_model_name)
            self.FMT_VALTRAIN_MASK_STORE = self.IMAGE_DIR + "/valtrain_{}_mask.h5"
            self.FMT_VALTRAIN_IM_STORE = self.IMAGE_DIR + "/valtrain_{}_im.h5"
            self.FMT_VALTRAIN_MUL_STORE = self.IMAGE_DIR + "/valtrain_{}_mul.h5"
            self.FMT_VALTEST_MASK_STORE = self.IMAGE_DIR + "/valtest_{}_mask.h5"
            self.FMT_VALTEST_IM_STORE = self.IMAGE_DIR + "/valtest_{}_im.h5"
            self.FMT_VALTEST_MUL_STORE = self.IMAGE_DIR + "/valtest_{}_mul.h5"
            self.FMT_VALTESTPOLY_PATH = self.MODEL_DIR + "/{}_eval_poly.csv"
            self.FMT_IMMEAN = self.IMAGE_DIR + "/{}_immean.h5"
            self.FMT_MULMEAN = self.IMAGE_DIR + "/{}_mulmean.h5"
            self.FMT_TEST_IM_STORE = self.IMAGE_DIR + "/test_{}_im.h5"
            self.FMT_TEST_MUL_STORE = self.IMAGE_DIR + "/test_{}_mul.h5"
            self.FMT_VALTESTTRUTH_PATH = self.MODEL_DIR + "/{}_eval_poly_truth.csv"
            self.FMT_VALMODEL_EVALHIST = self.MODEL_DIR + "/{}_val_evalhist.csv"
            self.FMT_VALMODEL_EVALTHHIST = self.MODEL_DIR + "/{}_val_evalhist_th.csv"
            self.FMT_VALTESTPRED_PATH = self.MODEL_DIR + "/{}_eval_pred.h5"
            self.FMT_VALTRAIN_OSM_STORE = self.IMAGE_DIR + "/valtrain_{}_osm.h5"
            self.FMT_VALTEST_OSM_STORE = self.IMAGE_DIR + "/valtest_{}_osm.h5"
            self.FMT_TRAIN_OSM_STORE = self.IMAGE_DIR + "/train_{}_osm.h5"
            self.FMT_TEST_OSM_STORE = self.IMAGE_DIR + "/test_{}_osm.h5"
            self.FMT_OSM_MEAN = self.IMAGE_DIR + "/{}_osmmean.h5"
            self.FMT_OSMSHAPEFILE = self.WORKING_DIR + "/osmdata/{name:}/{name:}_{layer:}.shp"
            self.FMT_SERIALIZED_OSMDATA = self.WORKING_DIR + "/osm_{}_subset.pkl"
            self.LAYER_NAMES = [
                'buildings',
                'landusages',
                'roads',
                'waterareas',
            ]
            if (base_model_name == model_name):
                self.FMT_VALTRAIN_IMAGELIST_PATH = self.IMAGE_DIR + "/{prefix:s}_valtrain_ImageId.csv"
                self.FMT_MUL_BANDCUT_TH_PATH = self.IMAGE_DIR + "/mul_bandcut{}.csv"
                self.FMT_VALTEST_IMAGELIST_PATH = self.IMAGE_DIR + "/{prefix:s}_valtest_ImageId.csv"
                self.FMT_TEST_IMAGELIST_PATH = self.IMAGE_DIR + "/{prefix:s}_test_ImageId.csv"
            else:
                self.FMT_VALTRAIN_IMAGELIST_PATH = self.BASE_IMAGE_DIR + "/{prefix:s}_valtrain_ImageId.csv"
                self.FMT_MUL_BANDCUT_TH_PATH = self.BASE_IMAGE_DIR + "/mul_bandcut{}.csv"
                self.FMT_VALTEST_IMAGELIST_PATH = self.BASE_IMAGE_DIR + "/{prefix:s}_valtest_ImageId.csv"
                self.FMT_TEST_IMAGELIST_PATH = self.BASE_IMAGE_DIR + "/{prefix:s}_test_ImageId.csv"


            if (is_final):
                self.FMT_TRAIN_IMAGELIST_PATH = self.BASE_IMAGE_DIR + "/{prefix:s}_train_ImageId.csv"
                # self.FMT_TEST_IMAGELIST_PATH = self.BASE_IMAGE_DIR + "/{prefix:s}_test_ImageId.csv"
                self.V12_IMAGE_DIR = self.ROOT_DIR + "/working/images/{}".format("v12")
                # Mask
                self.FMT_VALTRAIN_MASK_STORE = self.V12_IMAGE_DIR + "/valtrain_{}_mask.h5"
                self.FMT_VALTEST_MASK_STORE = self.V12_IMAGE_DIR + "/valtest_{}_mask.h5"
                self.FMT_TRAIN_MASK_STORE = self.V12_IMAGE_DIR + "/train_{}_mask.h5"

                # MUL
                self.FMT_VALTRAIN_MUL_STORE = self.V12_IMAGE_DIR + "/valtrain_{}_mul.h5"
                self.FMT_VALTEST_MUL_STORE = self.V12_IMAGE_DIR + "/valtest_{}_mul.h5"
                self.FMT_TRAIN_MUL_STORE = self.V12_IMAGE_DIR + "/train_{}_mul.h5"
                self.FMT_TEST_MUL_STORE = self.V12_IMAGE_DIR + "/test_{}_mul.h5"
                self.FMT_MULMEAN = self.V12_IMAGE_DIR + "/{}_mulmean.h5"

                self.FMT_RGB_BANDCUT_TH_PATH = self.V12_IMAGE_DIR + "/rgb_bandcut{}.csv"
                self.FMT_MUL_BANDCUT_TH_PATH = self.V12_IMAGE_DIR + "/mul_bandcut{}.csv"

            else:
                print("")

            self.handler = StreamHandler()
            self.handler.setLevel(INFO)
            self.handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))
            self.logger = getLogger('spacenet')
            self.logger.setLevel(INFO)
            np.random.seed(1145141919)


            # Parameters
            self.MIN_POLYGON_AREA = 30

            # Input files
            self.FMT_TRAIN_SUMMARY_PATH_V5 = str(
                Path(self.BASE_DIR) /
                Path("{prefix:s}_Train/") /
                Path("summaryData/{prefix:s}_Train_Building_Solutions.csv"))
            self.FMT_TRAIN_RGB_IMAGE_PATH_V5 = str(
                Path(self.BASE_DIR) /
                Path("{prefix:s}_Train/") /
                Path("RGB-PanSharpen/RGB-PanSharpen_{image_id:s}.tif"))
            self.FMT_TEST_RGB_IMAGE_PATH_V5 = str(
                Path(self.BASE_DIR) /
                Path("{prefix:s}_Test_public/") /
                Path("RGB-PanSharpen/RGB-PanSharpen_{image_id:s}.tif"))
            self.FMT_TRAIN_MSPEC_IMAGE_PATH_V5 = str(
                Path(self.BASE_DIR) /
                Path("{prefix:s}_Train/") /
                Path("MUL-PanSharpen/MUL-PanSharpen_{image_id:s}.tif"))
            self.FMT_TEST_MSPEC_IMAGE_PATH_V5 = str(
                Path(self.BASE_DIR) /
                Path("{prefix:s}_Test_public/") /
                Path("MUL-PanSharpen/MUL-PanSharpen_{image_id:s}.tif"))

            self.FMT_BANDCUT_TH_PATH = self.IMAGE_DIR + "/bandcut{}.csv"

            self.tfrecords_filename_rgb_train = self.IMAGE_DIR +"/"+ str(self.plugin_config["tfrecords_filename_rgb_train"])
            self.tfrecords_filename_rgb_test = self.IMAGE_DIR +"/"+ str(self.plugin_config["tfrecords_filename_rgb_test"])
            self.tfrecords_filename_multi_train = self.IMAGE_DIR +"/"+ str(self.plugin_config["tfrecords_filename_multi_train"])
            self.tfrecords_filename_multi_test = self.IMAGE_DIR +"/"+ str(self.plugin_config["tfrecords_filename_multi_test"])

    def createRFRecoad(self, img, annotation, image_id, writer):
        height = self.input_image_height
        width = self.input_image_width
        # print("image ---->", str(image_id), "---", "width", str(width))
        channels = img.shape[2]
        mask_height = self.input_image_height
        mask_width = self.input_image_width
        mask_channels = 1

        img_raw = img.tostring()
        annotation_raw = annotation.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'image_height': self._int64_feature(height),
            'image_width': self._int64_feature(width),
            'channels': self._int64_feature(channels),
            'mask_height': self._int64_feature(mask_height),
            'mask_width': self._int64_feature(mask_width),
            'mask_channels': self._int64_feature(mask_channels),
            'image_raw': self._bytes_feature(img_raw),
            'image_id': self._int64_feature(image_id),
            'mask_raw': self._bytes_feature(annotation_raw)}))
        serialized = example.SerializeToString()
        writer.write(serialized)


    def directory_name_to_area_id(self, datapath):
        dir_name = Path(datapath).name
        if dir_name.startswith('AOI_2_Vegas'):
            return 2
        elif dir_name.startswith('AOI_3_Paris'):
            return 3
        elif dir_name.startswith('AOI_4_Shanghai'):
            return 4
        elif dir_name.startswith('AOI_5_Khartoum'):
            return 5
        else:
            raise RuntimeError("Unsupported city id is given.")

    def _bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def calc_rgb_multiband_cut_threshold(self, area_id, datapath):
        rows = []
        band_cut_th = self.__calc_rgb_multiband_cut_threshold(area_id, datapath)
        prefix = self.area_id_to_prefix(area_id)
        row = dict(prefix=self.area_id_to_prefix(area_id))
        row['area_id'] = area_id
        for chan_i in band_cut_th.keys():
            row['chan{}_max'.format(chan_i)] = band_cut_th[chan_i]['max']
            row['chan{}_min'.format(chan_i)] = band_cut_th[chan_i]['min']
        rows.append(row)
        pd.DataFrame(rows).to_csv(
            self.FMT_RGB_BANDCUT_TH_PATH.format(prefix), index=False)

    def __calc_rgb_multiband_cut_threshold(self, area_id, datapath):
        prefix = self.area_id_to_prefix(area_id)
        band_values = {k: [] for k in range(3)}
        band_cut_th = {k: dict(max=0, min=0) for k in range(3)}

        image_id_list = pd.read_csv(self.FMT_VALTRAIN_IMAGELIST_PATH.format(
            prefix=prefix)).ImageId.tolist()
        for image_id in tqdm.tqdm(image_id_list[:500]):
            image_fn = self.get_train_image_path_from_imageid(image_id, datapath)
            with rasterio.open(image_fn, 'r') as f:
                values = f.read().astype(np.float32)
                for i_chan in range(3):
                    values_ = values[i_chan].ravel().tolist()
                    values_ = np.array(
                        [v for v in values_ if v != 0]
                    )  # Remove sensored mask
                    band_values[i_chan].append(values_)

        image_id_list = pd.read_csv(self.FMT_VALTEST_IMAGELIST_PATH.format(
            prefix=prefix)).ImageId.tolist()
        for image_id in tqdm.tqdm(image_id_list[:500]):
            image_fn = self.get_train_image_path_from_imageid(image_id, datapath)
            with rasterio.open(image_fn, 'r') as f:
                values = f.read().astype(np.float32)
                for i_chan in range(3):
                    values_ = values[i_chan].ravel().tolist()
                    values_ = np.array(
                        [v for v in values_ if v != 0]
                    )  # Remove sensored mask
                    band_values[i_chan].append(values_)

        self.logger.info("Calc percentile point ...")
        for i_chan in range(3):
            band_values[i_chan] = np.concatenate(
                band_values[i_chan]).ravel()
            band_cut_th[i_chan]['max'] = scipy.percentile(
                band_values[i_chan], 98)
            band_cut_th[i_chan]['min'] = scipy.percentile(
                band_values[i_chan], 2)
        return band_cut_th

    def calc_mul_multiband_cut_threshold(self, area_id, datapath):
        rows = []
        band_cut_th = self.__calc_mul_multiband_cut_threshold(area_id, datapath)
        prefix = self.area_id_to_prefix(area_id)
        row = dict(prefix=self.area_id_to_prefix(area_id))
        row['area_id'] = area_id
        for chan_i in band_cut_th.keys():
            row['chan{}_max'.format(chan_i)] = band_cut_th[chan_i]['max']
            row['chan{}_min'.format(chan_i)] = band_cut_th[chan_i]['min']
        rows.append(row)
        pd.DataFrame(rows).to_csv(
            self.FMT_MUL_BANDCUT_TH_PATH.format(prefix),
            index=False)

    def __calc_mul_multiband_cut_threshold(self, area_id, datapath):
        prefix = self.area_id_to_prefix(area_id)
        band_values = {k: [] for k in range(8)}
        band_cut_th = {k: dict(max=0, min=0) for k in range(8)}

        image_id_list = pd.read_csv(self.FMT_VALTRAIN_IMAGELIST_PATH.format(
            prefix=prefix)).ImageId.tolist()
        for image_id in tqdm.tqdm(image_id_list[:500]):
            image_fn = self.get_train_image_path_from_imageid(
                image_id, datapath, mul=True)
            with rasterio.open(image_fn, 'r') as f:
                values = f.read().astype(np.float32)
                for i_chan in range(8):
                    values_ = values[i_chan].ravel().tolist()
                    values_ = np.array(
                        [v for v in values_ if v != 0]
                    )  # Remove sensored mask
                    band_values[i_chan].append(values_)

        image_id_list = pd.read_csv(self.FMT_VALTEST_IMAGELIST_PATH.format(
            prefix=prefix)).ImageId.tolist()
        for image_id in tqdm.tqdm(image_id_list[:500]):
            image_fn = self.get_train_image_path_from_imageid(
                image_id, datapath, mul=True)
            with rasterio.open(image_fn, 'r') as f:
                values = f.read().astype(np.float32)
                for i_chan in range(8):
                    values_ = values[i_chan].ravel().tolist()
                    values_ = np.array(
                        [v for v in values_ if v != 0]
                    )  # Remove sensored mask
                    band_values[i_chan].append(values_)

        self.logger.info("Calc percentile point ...")
        for i_chan in range(8):
            band_values[i_chan] = np.concatenate(
                band_values[i_chan]).ravel()
            band_cut_th[i_chan]['max'] = scipy.percentile(
                band_values[i_chan], 98)
            band_cut_th[i_chan]['min'] = scipy.percentile(
                band_values[i_chan], 2)
        return band_cut_th

    def image_mask_resized_from_summary(self, df, image_id):
        im_mask = np.zeros((650, 650))

        if len(df[df.ImageId == image_id]) == 0:
            raise RuntimeError("ImageId not found on summaryData: {}".format(
                image_id))

        for idx, row in df[df.ImageId == image_id].iterrows():
            shape_obj = shapely.wkt.loads(row.PolygonWKT_Pix)
            if shape_obj.exterior is not None:
                coords = list(shape_obj.exterior.coords)
                x = [round(float(pp[0])) for pp in coords]
                y = [round(float(pp[1])) for pp in coords]
                yy, xx = skimage.draw.polygon(y, x, (650, 650))
                im_mask[yy, xx] = 1

                interiors = shape_obj.interiors
                for interior in interiors:
                    coords = list(interior.coords)
                    x = [round(float(pp[0])) for pp in coords]
                    y = [round(float(pp[1])) for pp in coords]
                    yy, xx = skimage.draw.polygon(y, x, (650, 650))
                    im_mask[yy, xx] = 0
        im_mask = skimage.transform.resize(im_mask, (self.INPUT_SIZE, self.INPUT_SIZE))
        im_mask = (im_mask > 0.5).astype(np.uint8)
        return im_mask

    def prep_image_mask(self, area_id, is_valtrain=True):
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("prep_image_mask for {}".format(prefix))
        if is_valtrain:
            fn_list = self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
            fn_mask = self.FMT_VALTRAIN_MASK_STORE.format(prefix)
        else:
            fn_list = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
            fn_mask = self.FMT_VALTEST_MASK_STORE.format(prefix)

        df = pd.read_csv(fn_list, index_col='ImageId')
        df_summary = self._load_train_summary_data(area_id)
        self.logger.info("Prepare image container: {}".format(fn_mask))
        with tb.open_file(fn_mask, 'w') as f:
            for image_id in tqdm.tqdm(df.index, total=len(df)):
                im_mask = self.image_mask_resized_from_summary(df_summary, image_id)
                atom = tb.Atom.from_dtype(im_mask.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im_mask.shape,
                                     filters=filters)
                ds[:] = im_mask

    def get_slice_mask_im(self, df, image_id):
        im_mask = np.zeros((650, 650))

        if len(df[df.ImageId == image_id]) == 0:
            raise RuntimeError("ImageId not found on summaryData: {}".format(
                image_id))

        for idx, row in df[df.ImageId == image_id].iterrows():
            shape_obj = shapely.wkt.loads(row.PolygonWKT_Pix)
            if shape_obj.exterior is not None:
                coords = list(shape_obj.exterior.coords)
                x = [round(float(pp[0])) for pp in coords]
                y = [round(float(pp[1])) for pp in coords]
                yy, xx = skimage.draw.polygon(y, x, (650, 650))
                im_mask[yy, xx] = 1

                interiors = shape_obj.interiors
                for interior in interiors:
                    coords = list(interior.coords)
                    x = [round(float(pp[0])) for pp in coords]
                    y = [round(float(pp[1])) for pp in coords]
                    yy, xx = skimage.draw.polygon(y, x, (650, 650))
                    im_mask[yy, xx] = 0
        im_mask = (im_mask > 0.5).astype(np.uint8)

        for slice_pos in range(9):
            pos_j = int(math.floor(slice_pos / 3.0))
            pos_i = int(slice_pos % 3)
            x0 = self.STRIDE_SZ * pos_i
            y0 = self.STRIDE_SZ * pos_j
            im_mask_part = im_mask[x0:x0 + self.INPUT_SIZE, y0:y0 + self.INPUT_SIZE]
            assert im_mask_part.shape == (256, 256)
            yield slice_pos, im_mask_part

    def load_train_summary_data(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        fn = self.FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
        df = pd.read_csv(fn)
        return df

    def prep_image_mask_v12(self, area_id, is_valtrain=True):
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("prep_image_mask for {}".format(prefix))
        if is_valtrain:
            fn_list = self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
            fn_mask = self.FMT_VALTRAIN_MASK_STORE.format(prefix)
        else:
            fn_list = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
            fn_mask = self.FMT_VALTEST_MASK_STORE.format(prefix)

        df = pd.read_csv(fn_list, index_col='ImageId')
        df_summary = self.load_train_summary_data(area_id)
        self.logger.info("Prepare image container: {}".format(fn_mask))
        with tb.open_file(fn_mask, 'w') as f:
            for image_id in tqdm.tqdm(df.index, total=len(df)):
                for pos, im_mask in self.get_slice_mask_im(df_summary, image_id):
                    atom = tb.Atom.from_dtype(im_mask.dtype)
                    slice_id = image_id + "_" + str(pos)
                    filters = tb.Filters(complib='blosc', complevel=9)
                    ds = f.create_carray(f.root, slice_id, atom,
                                         im_mask.shape,
                                         filters=filters)
                    ds[:] = im_mask

    def prep_rgb_image_store_train(self, area_id, datapath, is_valtrain=True):
        prefix = self.area_id_to_prefix(area_id)
        bandstats = self.__load_rgb_bandstats(area_id)

        self.logger.info("prep_rgb_image_store_train for {}".format(prefix))
        if is_valtrain:
            fn_list = self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
            fn_store = self.FMT_VALTRAIN_IM_STORE.format(prefix)
        else:
            fn_list = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
            fn_store = self.FMT_VALTEST_IM_STORE.format(prefix)

        df_list = pd.read_csv(fn_list, index_col='ImageId')

        self.logger.info("Image store file: {}".format(fn_store))
        with tb.open_file(fn_store, 'w') as f:
            for image_id in tqdm.tqdm(df_list.index, total=len(df_list)):
                im = self.get_resized_3chan_image_train(image_id, datapath, bandstats)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im.shape,
                                     filters=filters)
                ds[:] = im

    def prep_rgb_image_store_test(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        bandstats = self.__load_rgb_bandstats(area_id)

        self.logger.info("prep_rgb_image_store_test for {}".format(prefix))
        fn_list = self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
        fn_store = self.FMT_TEST_IM_STORE.format(prefix)
        df_list = pd.read_csv(fn_list, index_col='ImageId')

        self.logger.info("Image store file: {}".format(fn_store))
        with tb.open_file(fn_store, 'w') as f:
            for image_id in tqdm.tqdm(df_list.index, total=len(df_list)):
                im = self.get_resized_3chan_image_test(image_id, self.DATA_PATH_TEST, bandstats)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im.shape,
                                     filters=filters)
                ds[:] = im

    def get_resized_3chan_image_train(self, image_id, datapath, bandstats):
        fn = self.get_train_image_path_from_imageid(image_id, datapath)
        with rasterio.open(fn, 'r') as f:
            values = f.read().astype(np.float32)
            for chan_i in range(3):
                min_val = bandstats[chan_i]['min']
                max_val = bandstats[chan_i]['max']
                values[chan_i] = np.clip(values[chan_i], min_val, max_val)
                values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)

        values = np.swapaxes(values, 0, 2)
        values = np.swapaxes(values, 0, 1)
        values = skimage.transform.resize(values, (self.INPUT_SIZE, self.INPUT_SIZE))
        return values

    def get_resized_3chan_image_test(self, image_id, datapath, bandstats):
        fn = self.get_test_image_path_from_imageid(image_id, datapath)
        with rasterio.open(fn, 'r') as f:
            values = f.read().astype(np.float32)
            for chan_i in range(3):
                min_val = bandstats[chan_i]['min']
                max_val = bandstats[chan_i]['max']
                values[chan_i] = np.clip(values[chan_i], min_val, max_val)
                values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)

        values = np.swapaxes(values, 0, 2)
        values = np.swapaxes(values, 0, 1)
        values = skimage.transform.resize(values, (self.INPUT_SIZE, self.INPUT_SIZE))
        return values


    def prep_mul_image_store_train(self, area_id, datapath, is_valtrain=True):
        prefix = self.area_id_to_prefix(area_id)
        bandstats_rgb = self.__load_rgb_bandstats(area_id)
        bandstats_mul = self.__load_mul_bandstats(area_id)

        self.logger.info("prep_mul_image_store_train for ".format(prefix))
        if is_valtrain:
            fn_list = self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
            fn_store = self.FMT_VALTRAIN_MUL_STORE.format(prefix)
        else:
            fn_list = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
            fn_store = self.FMT_VALTEST_MUL_STORE.format(prefix)

        df_list = pd.read_csv(fn_list, index_col='ImageId')

        self.logger.info("Image store file: {}".format(fn_store))
        with tb.open_file(fn_store, 'w') as f:
            for image_id in tqdm.tqdm(df_list.index, total=len(df_list)):
                im = self.get_resized_8chan_image_train(
                    image_id, datapath, bandstats_rgb, bandstats_mul)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im.shape,
                                     filters=filters)
                ds[:] = im

    def prep_mul_image_only_store_train(self, area_id, datapath, is_valtrain=True):
        prefix = self.area_id_to_prefix(area_id)
        bandstats_mul = self.__load_mul_bandstats(area_id)

        self.logger.info("prep_mul_image_only_store_train for ".format(prefix))
        if is_valtrain:
            fn_list = self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
            fn_store = self.FMT_VALTRAIN_MUL_STORE.format(prefix)
        else:
            fn_list = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
            fn_store = self.FMT_VALTEST_MUL_STORE.format(prefix)

        df_list = pd.read_csv(fn_list, index_col='ImageId')

        self.logger.info("Image store file: {}".format(fn_store))
        with tb.open_file(fn_store, 'w') as f:
            for image_id in tqdm.tqdm(df_list.index, total=len(df_list)):
                for slice_pos, im in self.get_slice_8chan_im(image_id,
                                                        datapath,
                                                        bandstats_mul):
                    slice_id = '{}_{}'.format(image_id, slice_pos)
                    atom = tb.Atom.from_dtype(im.dtype)
                    filters = tb.Filters(complib='blosc', complevel=9)
                    ds = f.create_carray(f.root, slice_id, atom, im.shape,
                                         filters=filters)
                    ds[:] = im

    def get_slice_8chan_im(self, image_id, datapath, bandstats, is_test=False):
        fn = self.get_train_image_path_from_imageid(
            image_id, datapath, mul=True)
        if is_test:
            fn = self.get_test_image_path_from_imageid(
                image_id, datapath, mul=True)

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

        for slice_pos in range(9):
            pos_j = int(math.floor(slice_pos / 3.0))
            pos_i = int(slice_pos % 3)
            x0 = self.STRIDE_SZ * pos_i
            y0 = self.STRIDE_SZ * pos_j
            im = values[x0:x0 + self.INPUT_SIZE, y0:y0 + self.INPUT_SIZE]
            assert im.shape == (256, 256, 8)
            yield slice_pos, im

    def prep_mul_image_store_test(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        bandstats_rgb = self.__load_rgb_bandstats(area_id)
        bandstats_mul = self.__load_mul_bandstats(area_id)

        self.logger.info("prep_mul_image_store_test for ".format(prefix))
        fn_list = self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
        fn_store = self.FMT_TEST_MUL_STORE.format(prefix)

        df_list = pd.read_csv(fn_list, index_col='ImageId')

        self.logger.info("Image store file: {}".format(fn_store))
        with tb.open_file(fn_store, 'w') as f:
            for image_id in tqdm.tqdm(df_list.index, total=len(df_list)):
                im = self.get_resized_8chan_image_test(
                    image_id, self.DATA_PATH_TEST, bandstats_rgb, bandstats_mul)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im.shape,
                                     filters=filters)
                ds[:] = im

    def prep_mul_image_store_test_v12(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        bandstats_mul = self.__load_mul_bandstats(area_id)

        self.logger.info("prep_mul_image_store_test for ".format(prefix))
        fn_list = self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
        fn_store = self.FMT_TEST_MUL_STORE.format(prefix)

        df_list = pd.read_csv(fn_list, index_col='ImageId')

        self.logger.info("Image store file: {}".format(fn_store))
        with tb.open_file(fn_store, 'w') as f:
            for image_id in tqdm.tqdm(df_list.index, total=len(df_list)):
                for slice_pos, im in self.get_slice_8chan_im(image_id,
                                                        self.DATA_PATH_TEST,
                                                        bandstats_mul,
                                                        is_test=True):
                    slice_id = '{}_{}'.format(image_id, slice_pos)
                    atom = tb.Atom.from_dtype(im.dtype)
                    filters = tb.Filters(complib='blosc', complevel=9)
                    ds = f.create_carray(f.root, slice_id, atom, im.shape,
                                         filters=filters)
                    ds[:] = im

    def get_resized_8chan_image_train(self, image_id, datapath, bs_rgb, bs_mul):
        im = []

        fn = self.get_train_image_path_from_imageid(image_id, datapath)
        with rasterio.open(fn, 'r') as f:
            values = f.read().astype(np.float32)
            for chan_i in range(3):
                min_val = bs_rgb[chan_i]['min']
                max_val = bs_rgb[chan_i]['max']
                values[chan_i] = np.clip(values[chan_i], min_val, max_val)
                values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
                im.append(skimage.transform.resize(
                    values[chan_i],
                    (self.INPUT_SIZE, self.INPUT_SIZE)))

        fn = self.get_train_image_path_from_imageid(image_id, datapath, mul=True)
        with rasterio.open(fn, 'r') as f:
            values = f.read().astype(np.float32)
            usechannels = [1, 2, 5, 6, 7]
            for chan_i in usechannels:
                min_val = bs_mul[chan_i]['min']
                max_val = bs_mul[chan_i]['max']
                values[chan_i] = np.clip(values[chan_i], min_val, max_val)
                values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
                im.append(skimage.transform.resize(
                    values[chan_i],
                    (self.INPUT_SIZE, self.INPUT_SIZE)))

        im = np.array(im)  # (ch, w, h)
        im = np.swapaxes(im, 0, 2)  # -> (h, w, ch)
        im = np.swapaxes(im, 0, 1)  # -> (w, h, ch)
        return im

    def get_resized_8chan_image_test(self, image_id, datapath, bs_rgb, bs_mul):
        im = []

        fn = self.get_test_image_path_from_imageid(image_id, datapath)
        with rasterio.open(fn, 'r') as f:
            values = f.read().astype(np.float32)
            for chan_i in range(3):
                min_val = bs_rgb[chan_i]['min']
                max_val = bs_rgb[chan_i]['max']
                values[chan_i] = np.clip(values[chan_i], min_val, max_val)
                values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
                im.append(skimage.transform.resize(
                    values[chan_i],
                    (self.INPUT_SIZE, self.INPUT_SIZE)))

        fn = self.get_test_image_path_from_imageid(image_id, datapath, mul=True)
        with rasterio.open(fn, 'r') as f:
            values = f.read().astype(np.float32)
            usechannels = [1, 2, 5, 6, 7]
            for chan_i in usechannels:
                min_val = bs_mul[chan_i]['min']
                max_val = bs_mul[chan_i]['max']
                values[chan_i] = np.clip(values[chan_i], min_val, max_val)
                values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
                im.append(skimage.transform.resize(
                    values[chan_i],
                    (self.INPUT_SIZE, self.INPUT_SIZE)))

        im = np.array(im)  # (ch, w, h)
        im = np.swapaxes(im, 0, 2)  # -> (h, w, ch)
        im = np.swapaxes(im, 0, 1)  # -> (w, h, ch)
        return im

    def _load_train_summary_data(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        fn = self.FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
        df = pd.read_csv(fn)
        # df.loc[:, 'ImageId'] = df.ImageId.str[4:]
        return df

    def __load_rgb_bandstats(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        fn_stats = self.FMT_RGB_BANDCUT_TH_PATH.format(prefix)
        df_stats = pd.read_csv(fn_stats, index_col='area_id')
        r = df_stats.loc[area_id]
        stats_dict = {}
        for chan_i in range(3):
            stats_dict[chan_i] = dict(
                min=r['chan{}_min'.format(chan_i)],
                max=r['chan{}_max'.format(chan_i)])
        return stats_dict

    def __load_mul_bandstats(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        fn_stats = self.FMT_MUL_BANDCUT_TH_PATH.format(prefix)
        df_stats = pd.read_csv(fn_stats, index_col='area_id')
        r = df_stats.loc[area_id]

        stats_dict = {}
        for chan_i in range(8):
            stats_dict[chan_i] = dict(
                min=r['chan{}_min'.format(chan_i)],
                max=r['chan{}_max'.format(chan_i)])
        return stats_dict

    def __load_band_cut_th(self, band_fn):
        df = pd.read_csv(band_fn, index_col='area_id')
        all_band_cut_th = {area_id: {} for area_id in range(2, 6)}
        for area_id, row in df.iterrows():
            for chan_i in range(3):
                all_band_cut_th[area_id][chan_i] = dict(
                    min=row['chan{}_min'.format(chan_i)],
                    max=row['chan{}_max'.format(chan_i)],
                )
        return all_band_cut_th

    def prep_valtrain_valtest_imagelist(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        df = self._load_train_summary_data(area_id)
        df_agg = df.groupby('ImageId').agg('first')

        image_id_list = df_agg.index.tolist()
        np.random.shuffle(image_id_list)
        sz_valtrain = int(len(image_id_list) * 0.7)
        sz_valtest = len(image_id_list) - sz_valtrain

        base_dir = Path(self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)).parent
        if not base_dir.exists():
            base_dir.mkdir(parents=True)

        pd.DataFrame({'ImageId': image_id_list[:sz_valtrain]}).to_csv(
            self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix),
            index=False)
        pd.DataFrame({'ImageId': image_id_list[sz_valtrain:]}).to_csv(
            self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix),
            index=False)

    def prep_test_imagelist(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        image_directory = self.DATA_PATH_TEST + "/PAN/PAN_{prefix:s}_*.tif".format(prefix=prefix)
        image_id_list = glob.glob(image_directory)
        image_id_list = [path.split("PAN_")[-1][:-4] for path in image_id_list]
        pd.DataFrame({'ImageId': image_id_list}).to_csv(
            self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix),
            index=False)

    def get_train_image_path_from_imageid(self, image_id, datapath, mul=False):
        prefix = self.image_id_to_prefix(image_id)
        if mul:
            return self.FMT_TRAIN_MSPEC_IMAGE_PATH.format(
                datapath=datapath, prefix=prefix, image_id=image_id)
        else:
            return self.FMT_TRAIN_RGB_IMAGE_PATH.format(
                datapath=datapath, prefix=prefix, image_id=image_id)

    def get_test_image_path_from_imageid(self, image_id, datapath, mul=False):
        prefix = self.image_id_to_prefix(image_id)
        if mul:
            return self.FMT_TEST_MSPEC_IMAGE_PATH.format(
                datapath=datapath, image_id=image_id)
        else:
            return self.FMT_TEST_RGB_IMAGE_PATH.format(
                datapath=datapath, image_id=image_id)

    def image_id_to_prefix(self, image_id):
        prefix = image_id.split('img')[0][:-1]
        return prefix

    def prefix_to_area_id(self, prefix):
        area_dict = {
            'AOI_2_Vegas': 2,
            'AOI_3_Paris': 3,
            'AOI_4_Shanghai': 4,
            'AOI_5_Khartoum': 5,
        }
        return area_dict[prefix]

    def area_id_to_prefix(self, area_id):
        area_dict = {
            2: 'AOI_2_Vegas',
            3: 'AOI_3_Paris',
            4: 'AOI_4_Shanghai',
            5: 'AOI_5_Khartoum',
        }
        return area_dict[area_id]

    def prep_immean(self, area_id, datapath):
        prefix = self.area_id_to_prefix(area_id)
        X_train = []

        # Load valtrain
        fn_im = self.FMT_VALTRAIN_IM_STORE.format(prefix)
        image_list = pd.read_csv(self.FMT_VALTRAIN_IMAGELIST_PATH.format(
            prefix=prefix)).ImageId.tolist()
        with tb.open_file(fn_im, 'r') as f:
            for idx, image_id in enumerate(image_list):
                im = np.array(f.get_node('/' + image_id))
                im = np.swapaxes(im, 0, 2)
                im = np.swapaxes(im, 1, 2)
                X_train.append(im)

        # Load valtest
        fn_im = self.FMT_VALTEST_IM_STORE.format(prefix)
        image_list = pd.read_csv(self.FMT_VALTEST_IMAGELIST_PATH.format(
            prefix=prefix)).ImageId.tolist()
        with tb.open_file(fn_im, 'r') as f:
            for idx, image_id in enumerate(image_list):
                im = np.array(f.get_node('/' + image_id))
                im = np.swapaxes(im, 0, 2)
                im = np.swapaxes(im, 1, 2)
                X_train.append(im)

        X_mean = np.array(X_train).mean(axis=0)

        fn = self.FMT_IMMEAN.format(prefix)
        self.logger.info("Prepare mean image: {}".format(fn))
        with tb.open_file(fn, 'w') as f:
            atom = tb.Atom.from_dtype(X_mean.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root, 'immean', atom, X_mean.shape,
                                 filters=filters)
            ds[:] = X_mean

    def _remove_interiors(delf, line):
        if "), (" in line:
            line_prefix = line.split('), (')[0]
            line_terminate = line.split('))",')[-1]
            line = (
                line_prefix +
                '))",' +
                line_terminate
            )
        return line

    def prep_mulmean(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        X_train = []

        # Load valtrain
        fn_im = self.FMT_VALTRAIN_MUL_STORE.format(prefix)
        image_list = pd.read_csv(self.FMT_VALTRAIN_IMAGELIST_PATH.format(
            prefix=prefix)).ImageId.tolist()
        with tb.open_file(fn_im, 'r') as f:
            for idx, image_id in enumerate(image_list):
                im = np.array(f.get_node('/' + image_id))
                im = np.swapaxes(im, 0, 2)
                im = np.swapaxes(im, 1, 2)
                X_train.append(im)

        # Load valtest
        fn_im = self.FMT_VALTEST_MUL_STORE.format(prefix)
        image_list = pd.read_csv(self.FMT_VALTEST_IMAGELIST_PATH.format(
            prefix=prefix)).ImageId.tolist()
        with tb.open_file(fn_im, 'r') as f:
            for idx, image_id in enumerate(image_list):
                im = np.array(f.get_node('/' + image_id))
                im = np.swapaxes(im, 0, 2)
                im = np.swapaxes(im, 1, 2)
                X_train.append(im)

        X_mean = np.array(X_train).mean(axis=0)

        fn = self.FMT_MULMEAN.format(prefix)
        self.logger.info("Prepare mean image: {}".format(fn))
        with tb.open_file(fn, 'w') as f:
            atom = tb.Atom.from_dtype(X_mean.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root, 'mulmean', atom, X_mean.shape,
                                 filters=filters)
            ds[:] = X_mean

    def prep_mulmean_v12(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        X_train = []

        # Load valtrain
        fn_im = self.FMT_VALTRAIN_MUL_STORE.format(prefix)
        image_list = pd.read_csv(self.FMT_VALTRAIN_IMAGELIST_PATH.format(
            prefix=prefix)).ImageId.tolist()
        with tb.open_file(fn_im, 'r') as f:
            for idx, image_id in enumerate(image_list):
                slice_pos = 5
                slice_id = image_id + '_' + str(slice_pos)
                im = np.array(f.get_node('/' + slice_id))
                im = np.swapaxes(im, 0, 2)
                im = np.swapaxes(im, 1, 2)
                X_train.append(im)

        # Load valtest
        fn_im = self.FMT_VALTEST_MUL_STORE.format(prefix)
        image_list = pd.read_csv(self.FMT_VALTEST_IMAGELIST_PATH.format(
            prefix=prefix)).ImageId.tolist()
        with tb.open_file(fn_im, 'r') as f:
            for idx, image_id in enumerate(image_list):
                slice_pos = 5
                slice_id = image_id + '_' + str(slice_pos)
                im = np.array(f.get_node('/' + slice_id))
                im = np.swapaxes(im, 0, 2)
                im = np.swapaxes(im, 1, 2)
                X_train.append(im)

        X_mean = np.array(X_train).mean(axis=0)

        fn = self.FMT_MULMEAN.format(prefix)
        self.logger.info("Prepare mean image: {}".format(fn))
        with tb.open_file(fn, 'w') as f:
            atom = tb.Atom.from_dtype(X_mean.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root, 'mulmean', atom, X_mean.shape,
                                 filters=filters)
            ds[:] = X_mean

    def preproc_train(self):
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("Preproc for training on {}".format(prefix))
        if Path(self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)).exists():
            self.logger.info("Generate IMAGELIST csv ... skip")
        else:
            self.logger.info("Generate IMAGELIST csv")
            self.prep_valtrain_valtest_imagelist(area_id)
        if Path(self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)).exists():
            self.logger.info("Generate IMAGELIST csv ... skip")
        else:
            self.logger.info("Generate IMAGELIST csv")
            self.prep_valtrain_valtest_imagelist(area_id)
        if Path(self.FMT_RGB_BANDCUT_TH_PATH.format(prefix)).exists():
            self.logger.info("Generate band stats csv (RGB) ... skip")
        else:
            self.logger.info("Generate band stats csv (RGB)")
            self.calc_rgb_multiband_cut_threshold(area_id, self.DATA_PATH)
        if Path(self.FMT_MUL_BANDCUT_TH_PATH.format(prefix)).exists():
            self.logger.info("Generate band stats csv (MUL) ... skip")
        else:
            self.logger.info("Generate band stats csv (MUL)")
            self.calc_mul_multiband_cut_threshold(area_id, self.DATA_PATH)
        if Path(self.FMT_VALTRAIN_MASK_STORE.format(prefix)).exists():
            self.logger.info("Generate MASK (valtrain) ... skip")
        else:
            self.logger.info("Generate MASK (valtrain)")
            self.prep_image_mask(area_id, is_valtrain=True)
        if Path(self.FMT_VALTEST_MASK_STORE.format(prefix)).exists():
            self.logger.info("Generate MASK (valtest) ... skip")
        else:
            self.logger.info("Generate MASK (valtest)")
            self.prep_image_mask(area_id, is_valtrain=False)
        if Path(self.FMT_VALTRAIN_IM_STORE.format(prefix)).exists():
            self.logger.info("Generate RGB_STORE (valtrain) ... skip")
        else:
            self.logger.info("Generate RGB_STORE (valtrain)")
            self.prep_rgb_image_store_train(area_id, self.DATA_PATH, is_valtrain=True)
        if Path(self.FMT_VALTEST_IM_STORE.format(prefix)).exists():
            self.logger.info("Generate RGB_STORE (valtest) ... skip")
        else:
            self.logger.info("Generate RGB_STORE (valtest)")
            self.prep_rgb_image_store_train(area_id, self.DATA_PATH, is_valtrain=False)
        if Path(self.FMT_VALTRAIN_MUL_STORE.format(prefix)).exists():
            self.logger.info("Generate MUL_STORE (valtrain) ... skip")
        else:
            self.logger.info("Generate MUL_STORE (valtrain)")
            self.prep_mul_image_store_train(area_id, self.DATA_PATH, is_valtrain=True)
        if Path(self.FMT_VALTEST_MUL_STORE.format(prefix)).exists():
            self.logger.info("Generate MUL_STORE (valtest) ... skip")
        else:
            self.logger.info("Generate MUL_STORE (valtest)")
            self.prep_mul_image_store_train(area_id, self.DATA_PATH, is_valtrain=False)
        if Path(self.FMT_IMMEAN.format(prefix)).exists():
            self.logger.info("Generate RGBMEAN ... skip")
        else:
            self.logger.info("Generate RGBMEAN")
            self.prep_immean(area_id, self.DATA_PATH)
        if Path(self.FMT_MULMEAN.format(prefix)).exists():
            self.logger.info("Generate MULMEAN ... skip")
        else:
            self.logger.info("Generate MULMEAN")
            self.prep_mulmean(area_id)
            self.logger.info("Preproc for training on {} ... done".format(prefix))

    def preproc_test(self):
        area_id = self.directory_name_to_area_id(self.DATA_PATH_TEST)
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("preproc_test for {}".format(prefix))
        if Path(self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)).exists():
            self.logger.info("Generate IMAGELIST for inference ... skip")
        else:
            self.logger.info("Generate IMAGELIST for inference")
            self.prep_test_imagelist(area_id)
        if Path(self.FMT_TEST_IM_STORE.format(prefix)).exists():
            self.logger.info("Generate RGB_STORE (test) ... skip")
        else:
            self.logger.info("Generate RGB_STORE (test)")
            self.prep_rgb_image_store_test(area_id)
        if Path(self.FMT_TEST_MUL_STORE.format(prefix)).exists():
            self.logger.info("Generate MUL_STORE (test) ... skip")
        else:
            self.logger.info("Generate MUL_STORE (test)")
            self.prep_mul_image_store_test(area_id)
            self.logger.info("preproc_test for {} ... done".format(prefix))

    def preproc_test_v12(self):
        """ test.sh """
        area_id = self.directory_name_to_area_id(self.DATA_PATH_TEST)
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("preproc_test for {}".format(prefix))

        # Imagelist
        assert Path(self.FMT_TEST_IMAGELIST_PATH.format(
            prefix=prefix)).exists()

        # Image HDF5 store (MUL)
        if Path(self.FMT_TEST_MUL_STORE.format(prefix)).exists():
            self.logger.info("Generate MUL_STORE (test) ... skip")
        else:
            self.logger.info("Generate MUL_STORE (test)")
            self.prep_mul_image_store_test_v12(area_id)

        self.logger.info("preproc_test for {} ... done".format(prefix))

    def preproc_test_v16(self):
        """ test.sh """
        area_id = self.directory_name_to_area_id(self.DATA_PATH_TEST)
        prefix = self.area_id_to_prefix(area_id)
        osmprefix = self.area_id_to_osmprefix(area_id)

        # Mkdir
        if not Path(self.FMT_TEST_OSM_STORE.format(prefix)).parent.exists():
            Path(self.FMT_TEST_OSM_STORE.format(prefix)).parent.mkdir(
                parents=True)

        # OSM serialized subset
        fn_osm = self.FMT_SERIALIZED_OSMDATA.format(osmprefix)
        if Path(fn_osm).exists():
            self.logger.info("Serialize OSM subset ... skip")
        else:
            self.logger.info("Serialize OSM subset")
            self.preproc_osm(area_id, self.DATA_PATH_TEST, is_train=False)

        # OSM layers (test)
        if Path(self.FMT_TEST_OSM_STORE.format(prefix)).exists():
            self.logger.info("Generate OSM_STORE (test) ... skip")
        else:
            self.logger.info("Generate OSM_STORE (test)")
            self.prep_osmlayer_test(area_id, self.DATA_PATH_TEST)

    def _internal_pred_to_poly_file_test(self, area_id, y_pred, min_th=30):
        prefix = self.area_id_to_prefix(area_id)
        # Load test imagelist
        fn_test = self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test, index_col='ImageId')
        # Make parent directory
        fn_out = self.FMT_TESTPOLY_PATH.format(prefix)
        if not Path(fn_out).parent.exists():
            Path(fn_out).parent.mkdir(parents=True)

        # Ensemble individual models and write out output files
        with open(fn_out, 'w') as f:
            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            for idx, image_id in enumerate(df_test.index.tolist()):
                df_poly = self.mask_to_poly(y_pred[idx], min_polygon_area_th=min_th)
                if len(df_poly) > 0:
                    for i, row in df_poly.iterrows():
                        line = "{},{},\"{}\",{:.6f}\n".format(
                            image_id,
                            row.bid,
                            row.wkt,
                            row.area_ratio)
                        line = self._remove_interiors(line)
                        f.write(line)
                else:
                    f.write("{},{},{},0\n".format(
                        image_id,
                        -1,
                        "POLYGON EMPTY"))



    def get_model_parameter(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        fn_hist = self.FMT_VALMODEL_EVALTHHIST.format(prefix)
        best_row = pd.read_csv(fn_hist).sort_values( by='fscore', ascending=False).iloc[0]
        param = dict(min_poly_area=int(best_row['min_area_th']))
        return param

    def testproc(self, y_pred_0, y_pred_1, y_pred_2):
        area_id = self.directory_name_to_area_id(self.DATA_PATH_TEST)
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info(">>>> Test proc for {}".format(prefix))
        self.logger.info("import modules")
        # Ensemble
        self.logger.info("Averaging")
        y_pred = self._internal_validate_predict_best_param_for_all(area_id, rescale_pred_list=[y_pred_0],
                                                                    slice_pred_list=[y_pred_1, y_pred_2])
        # pred to polygon
        param = self.get_model_parameter(area_id)
        self._internal_pred_to_poly_file_test(area_id, y_pred, min_th=param['min_poly_area'],)
        self.logger.info(">>>> Test proc for {} ... done".format(prefix))

    def prep_osmlayer_test(self, area_id, datapath):
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("prep_osmlayer_test for {}".format(prefix))

        fn_list = self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
        fn_store = self.FMT_TEST_OSM_STORE.format(prefix)
        layers = self.extract_osmlayers(area_id)

        df = pd.read_csv(fn_list, index_col='ImageId')
        self.logger.info("Prep osm container: {}".format(fn_store))
        with tb.open_file(fn_store, 'w') as f:
            df_sz = len(df)
            for image_id in tqdm.tqdm(df.index, total=df_sz):
                # fn_tif = test_image_id_to_path(image_id)
                fn_tif = self.get_test_image_path_from_imageid(
                    image_id, datapath, mul=False)
                with rasterio.open(fn_tif, 'r') as fr:
                    values = fr.read(1)
                    masks = []  # rasterize masks
                    for layer_geoms in layers:
                        mask = rasterio.features.rasterize(
                            layer_geoms,
                            out_shape=values.shape,
                            transform=rasterio.guard_transform(
                                fr.transform))
                        masks.append(mask)
                    masks = np.array(masks)
                    masks = np.swapaxes(masks, 0, 2)
                    masks = np.swapaxes(masks, 0, 1)
                assert masks.shape == (650, 650, len(layers))

                # slice of masks
                for slice_pos in range(9):
                    pos_j = int(math.floor(slice_pos / 3.0))
                    pos_i = int(slice_pos % 3)
                    x0 = self.STRIDE_SZ * pos_i
                    y0 = self.STRIDE_SZ * pos_j
                    im = masks[x0:x0 + self.INPUT_SIZE, y0:y0 + self.INPUT_SIZE]
                    assert im.shape == (256, 256, len(layers))
                    slice_id = image_id + "_{}".format(slice_pos)
                    atom = tb.Atom.from_dtype(im.dtype)
                    filters = tb.Filters(complib='blosc', complevel=9)
                    ds = f.create_carray(f.root,
                                         slice_id,
                                         atom,
                                         im.shape,
                                         filters=filters)
                    ds[:] = im

    def _get_valtest_mul_data(self, writer):
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        fn_test = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test)
        fn_im = self.FMT_VALTEST_MUL_STORE.format(prefix)
        fn_mask = self.FMT_VALTEST_MASK_STORE.format(prefix)
        with tb.open_file(fn_im, 'r') as f:
            with tb.open_file(fn_mask, 'r') as af:
                for idx, image_id in tqdm.tqdm(enumerate(df_test.ImageId.tolist())):
                    im = np.array(f.get_node('/' + image_id))
                    im = np.swapaxes(im, 0, 2)
                    im = np.swapaxes(im, 1, 2)
                    mask = np.array(af.get_node('/' + image_id))
                    mask = (mask > 0.5).astype(np.uint8)
                    im=np.transpose(im).astype(np.float64)
                    image_id_prefix = prefix+"_img"
                    image_id_index = image_id.replace(image_id_prefix, "")
                    image_id_index = int(image_id_index)
                    self.createRFRecoad(im, mask, image_id_index, writer)
        writer.close()
        self.logger.info("TFRecoad for validating images has been written to "
                         + self.tfrecords_filename_multi_test)
        return True

    def _get_valtrain_mul_data(self, writer):
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        fn_train = self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
        df_train = pd.read_csv(fn_train)
        fn_im = self.FMT_VALTRAIN_MUL_STORE.format(prefix)
        fn_mask = self.FMT_VALTRAIN_MASK_STORE.format(prefix)
        with tb.open_file(fn_im, 'r') as f:
            with tb.open_file(fn_mask, 'r') as af:
                for idx, image_id in tqdm.tqdm(enumerate(df_train.ImageId.tolist())):
                    im = np.array(f.get_node('/' + image_id))
                    im = np.swapaxes(im, 0, 2)
                    im = np.swapaxes(im, 1, 2)
                    # coff = np.zeros([im.shape[0], im.shape[0]])

                    # for i in range(0, im.shape[0]):  # rows a]re the number of rows in the matrix.
                    #     band1 = im[i].flatten()
                    #     for j in range(0, im.shape[0]):
                    #         band2 = im[j].flatten()
                    #         r = pearsonr(band1, band2)
                    #         coff[i][j]=r[0]
                    # plt.imshow(coff, cmap=plt.cm.ocean)
                    # plt.colorbar()
                    # plt.show()

                    mask = np.array(af.get_node('/' + image_id))
                    mask = (mask > 0.5).astype(np.uint8)
                    im=np.transpose(im).astype(np.float64)
                    image_id_prefix = prefix + "_img"
                    image_id_index = image_id.replace(image_id_prefix, "")
                    image_id_index = int(image_id_index)
                    self.createRFRecoad(im, mask, image_id_index, writer)
        writer.close()
        self.logger.info("TFRecoad for training images has been written to "
                         + self.tfrecords_filename_multi_train)
        return True

    def get_total_numberof_model_count(self, trainer):
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        model_path = trainer.get_least_model_details(self.MODEL_DIR + "/"+ prefix)
        return int(model_path.split("/")[-1].replace("model-", ""))

    def evalfscore(self, trainer, operators, count):
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("Evaluate fscore on validation set: {}".format(prefix))

        rows = []
        for zero_base_epoch in range(1, count+1):
            self.logger.info(">>> Epoch: {}".format(zero_base_epoch))
            enable_tqdm = False
            path = self.MODEL_DIR + "/"+ prefix + "/model-" + str(zero_base_epoch)
            self._internal_validate_fscore_wo_pred_file( area_id,trainer, path, operators, epoch=zero_base_epoch,
                                                         enable_tqdm=True, min_th=self.MIN_POLYGON_AREA)
            evaluate_record = self._calc_fscore_per_aoi(area_id)
            evaluate_record['zero_base_epoch'] = zero_base_epoch
            evaluate_record['min_area_th'] = self.MIN_POLYGON_AREA
            evaluate_record['area_id'] = area_id
            self.logger.info("\n" + json.dumps(evaluate_record, indent=4))
            rows.append(evaluate_record)
        pd.DataFrame(rows).to_csv(self.FMT_VALMODEL_EVALHIST.format(prefix), index=False)

        # find best min-poly-threshold
        df_evalhist = pd.read_csv(self.FMT_VALMODEL_EVALHIST.format(prefix))
        best_row = df_evalhist.sort_values(by='fscore', ascending=False).iloc[0]
        best_epoch = int(best_row.zero_base_epoch)
        best_fscore = best_row.fscore

        # optimize min area th
        rows = []
        for th in [30, 60, 90, 120, 150, 180, 210, 240]:
            self.logger.info(">>> TH: {}".format(th))
            predict_flag = True
            path = self.MODEL_DIR + "/"+ prefix + "/model-" + str(best_epoch)
            self._internal_validate_fscore(area_id, trainer, path, operators, epoch=3, predict=True, min_th=30)
            evaluate_record = self._calc_fscore_per_aoi(area_id)
            evaluate_record['zero_base_epoch'] = best_epoch
            evaluate_record['min_area_th'] = th
            evaluate_record['area_id'] = area_id
            self.logger.info("\n" + json.dumps(evaluate_record, indent=4))
            rows.append(evaluate_record)

        pd.DataFrame(rows).to_csv(self.FMT_VALMODEL_EVALTHHIST.format(prefix), index=False)

        self.logger.info("Evaluate fscore on validation set: {} .. done".format(prefix))

    def evalfscore_v12(self, trainer, operators, count):
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("Evaluate fscore on validation set: {}".format(prefix))

        rows = []
        for zero_base_epoch in range(1, count + 1):
            self.logger.info(">>> Epoch: {}".format(zero_base_epoch))
            enable_tqdm = False
            path = self.MODEL_DIR + "/"+ prefix + "/model-" + str(zero_base_epoch)
            self._internal_validate_fscore_wo_pred_file_v12(area_id, trainer, path, operators, epoch=zero_base_epoch,
                                                            enable_tqdm=True, min_th=self.MIN_POLYGON_AREA)
            evaluate_record = self._calc_fscore_per_aoi(area_id)
            evaluate_record['zero_base_epoch'] = zero_base_epoch
            evaluate_record['min_area_th'] = self.MIN_POLYGON_AREA
            evaluate_record['area_id'] = area_id
            self.logger.info("\n" + json.dumps(evaluate_record, indent=4))
            rows.append(evaluate_record)
        pd.DataFrame(rows).to_csv(self.FMT_VALMODEL_EVALHIST.format(prefix), index=False)

        # find best min-poly-threshold
        df_evalhist = pd.read_csv(self.FMT_VALMODEL_EVALHIST.format(prefix))
        best_row = df_evalhist.sort_values(by='fscore', ascending=False).iloc[0]
        best_epoch = int(best_row.zero_base_epoch)
        best_fscore = best_row.fscore

        # optimize min area th
        rows = []
        for th in [30, 60, 90, 120, 150, 180, 210, 240]:
            self.logger.info(">>> TH: {}".format(th))
            predict_flag = True
            path = self.MODEL_DIR + "/"+ prefix + "/model-" + str(best_epoch)
            self._internal_validate_fscore(area_id, trainer, path, operators, epoch=3, predict=True, min_th=30)
            evaluate_record = self._calc_fscore_per_aoi(area_id)
            evaluate_record['zero_base_epoch'] = best_epoch
            evaluate_record['min_area_th'] = th
            evaluate_record['area_id'] = area_id
            self.logger.info("\n" + json.dumps(evaluate_record, indent=4))
            rows.append(evaluate_record)

        pd.DataFrame(rows).to_csv(self.FMT_VALMODEL_EVALTHHIST.format(prefix), index=False)

        self.logger.info("Evaluate fscore on validation set: {} .. done".format(prefix))


    def evalfscore_v16(self, trainer, operators, count):
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("Evaluate fscore on validation set: {}".format(prefix))

        rows = []
        for zero_base_epoch in range(1, count + 1):
            self.logger.info(">>> Epoch: {}".format(zero_base_epoch))
            enable_tqdm = False
            path = self.MODEL_DIR + "/"+ prefix + "/model-" + str(zero_base_epoch)
            self._internal_validate_fscore_wo_pred_file_v16(area_id, trainer, path, operators, epoch=zero_base_epoch,
                                                        enable_tqdm=True, min_th=self.MIN_POLYGON_AREA)
            evaluate_record = self._calc_fscore_per_aoi(area_id)
            evaluate_record['zero_base_epoch'] = zero_base_epoch
            evaluate_record['min_area_th'] = self.MIN_POLYGON_AREA
            evaluate_record['area_id'] = area_id
            self.logger.info("\n" + json.dumps(evaluate_record, indent=4))
            rows.append(evaluate_record)
        pd.DataFrame(rows).to_csv(self.FMT_VALMODEL_EVALHIST.format(prefix), index=False)

        # find best min-poly-threshold
        df_evalhist = pd.read_csv(self.FMT_VALMODEL_EVALHIST.format(prefix))
        best_row = df_evalhist.sort_values(by='fscore', ascending=False).iloc[0]
        best_epoch = int(best_row.zero_base_epoch)
        best_fscore = best_row.fscore

        # optimize min area th
        rows = []
        for th in [30, 60, 90, 120, 150, 180, 210, 240]:
            self.logger.info(">>> TH: {}".format(th))
            predict_flag = True
            path = self.MODEL_DIR + "/"+ prefix + "/model-" + str(best_epoch)
            self._internal_validate_fscore(area_id, trainer, path, operators, epoch=3, predict=True, min_th=30)
            evaluate_record = self._calc_fscore_per_aoi(area_id)
            evaluate_record['zero_base_epoch'] = best_epoch
            evaluate_record['min_area_th'] = th
            evaluate_record['area_id'] = area_id
            self.logger.info("\n" + json.dumps(evaluate_record, indent=4))
            rows.append(evaluate_record)

        pd.DataFrame(rows).to_csv(self.FMT_VALMODEL_EVALTHHIST.format(prefix), index=False)

        self.logger.info("Evaluate fscore on validation set: {} .. done".format(prefix))

    def _internal_validate_fscore(self, area_id, trainer, path, operators, epoch=3, predict=True, min_th=30,
                                  enable_tqdm=False):
        prefix = self.area_id_to_prefix(area_id)
        # Prediction phase
        self.logger.info("Prediction phase")
        predicted_result, test_image_ids = self._internal_validate_predict(trainer, path, operators, save_pred=True)
        fn = self.FMT_VALTESTPRED_PATH.format(prefix)
        with tb.open_file(fn, 'w') as f:
            atom = tb.Atom.from_dtype(predicted_result.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root, 'pred', atom, predicted_result.shape, filters=filters)
            ds[:] = predicted_result

        # Postprocessing phase
        self.logger.info("Postprocessing phase")
        fn = self.FMT_VALTESTPRED_PATH.format(prefix)
        fn_out = self.FMT_VALTESTPOLY_PATH.format(prefix)
        with open(fn_out, 'w') as f, tb.open_file(fn, 'r') as fr:

            y_pred = np.array(fr.get_node('/pred'))

            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            test_list = test_image_ids.tolist()
            iterator = enumerate(test_list)

            for idx, image_id in tqdm.tqdm(iterator, total=len(test_list)):
                df_poly = self.mask_to_poly(y_pred[idx][0], min_polygon_area_th=min_th)
                if len(df_poly) > 0:
                    for i, row in df_poly.iterrows():
                        line = "{},{},\"{}\",{:.6f}\n".format(
                            prefix+'_img'+str(image_id[0]),
                            row.bid,
                            row.wkt,
                            row.area_ratio)
                        line = self._remove_interiors(line)
                        f.write(line)
                else:
                    f.write("{},{},{},0\n".format(
                        image_id,
                        -1,
                        "POLYGON EMPTY"))

        # ------------------------
        # Validation solution file
        self.logger.info("Validation solution file")
        # if not Path(FMT_VALTESTTRUTH_PATH.format(prefix)).exists():
        if True:
            fn_true = self.FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
            df_true = pd.read_csv(fn_true)
            # # Remove prefix "PAN_"
            # df_true.loc[:, 'ImageId'] = df_true.ImageId.str[4:]
            fn_test = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
            df_test = pd.read_csv(fn_test)
            df_test_image_ids = df_test.ImageId.unique()
            fn_out = self.FMT_VALTESTTRUTH_PATH.format(prefix)
            with open(fn_out, 'w') as f:
                f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
                df_true = df_true[df_true.ImageId.isin(df_test_image_ids)]
                for idx, r in df_true.iterrows():
                    f.write("{},{},\"{}\",{:.6f}\n".format(r.ImageId, r.BuildingId, r.PolygonWKT_Pix, 1.0))

    def _internal_validate_fscore_v16(self, area_id, trainer, path, operators, epoch=3, predict=True, min_th=30,
                                  enable_tqdm=False):
        prefix = self.area_id_to_prefix(area_id)
        # Prediction phase
        self.logger.info("Prediction phase")
        predicted_result, test_image_ids = self._internal_validate_predict(trainer, path, operators, save_pred=True)
        fn = self.FMT_VALTESTPRED_PATH.format(prefix)
        with tb.open_file(fn, 'w') as f:
            atom = tb.Atom.from_dtype(predicted_result.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root, 'pred', atom, predicted_result.shape, filters=filters)
            ds[:] = predicted_result

        # Postprocessing phase
        self.logger.info("Postprocessing phase")
        fn = self.FMT_VALTESTPRED_PATH.format(prefix)
        fn_out = self.FMT_VALTESTPOLY_PATH.format(prefix)
        with open(fn_out, 'w') as f, tb.open_file(fn, 'r') as fr:

            y_pred = np.array(fr.get_node('/pred'))

            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            test_list = test_image_ids.tolist()
            iterator = enumerate(test_list)

        for idx, image_id in tqdm.tqdm(iterator, total=len(test_list)):
            pred_values = np.zeros((650, 650))
            pred_count = np.zeros((650, 650))
            for slice_pos in range(9):
                slice_idx = idx * 9 + slice_pos

                pos_j = int(math.floor(slice_pos / 3.0))
                pos_i = int(slice_pos % 3)
                x0 = self.STRIDE_SZ * pos_i
                y0 = self.STRIDE_SZ * pos_j
                pred_values[x0:x0+self.INPUT_SIZE, y0:y0+self.INPUT_SIZE] += (
                    y_pred[slice_idx][0]
                )
                pred_count[x0:x0+self.INPUT_SIZE, y0:y0+self.INPUT_SIZE] += 1
            pred_values = pred_values / pred_count

            df_poly = self.mask_to_poly(pred_values, min_polygon_area_th=min_th)
            if len(df_poly) > 0:
                for i, row in df_poly.iterrows():
                    line = "{},{},\"{}\",{:.6f}\n".format(
                        image_id,
                        row.bid,
                        row.wkt,
                        row.area_ratio)
                    line = self._remove_interiors(line)
                    f.write(line)
            else:
                f.write("{},{},{},0\n".format(
                    image_id,
                    -1,
                    "POLYGON EMPTY"))

        # ------------------------
        # Validation solution file
        self.logger.info("Validation solution file")
        # if not Path(FMT_VALTESTTRUTH_PATH.format(prefix)).exists():
        if True:
            fn_true = self.FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
            df_true = pd.read_csv(fn_true)
            # # Remove prefix "PAN_"
            # df_true.loc[:, 'ImageId'] = df_true.ImageId.str[4:]
            fn_test = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
            df_test = pd.read_csv(fn_test)
            df_test_image_ids = df_test.ImageId.unique()
            fn_out = self.FMT_VALTESTTRUTH_PATH.format(prefix)
            with open(fn_out, 'w') as f:
                f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
                df_true = df_true[df_true.ImageId.isin(df_test_image_ids)]
                for idx, r in df_true.iterrows():
                    f.write("{},{},\"{}\",{:.6f}\n".format(r.ImageId, r.BuildingId, r.PolygonWKT_Pix, 1.0))

    def _calc_fscore_per_aoi(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        truth_file = self.FMT_VALTESTTRUTH_PATH.format(prefix)
        poly_file = self.FMT_VALTESTPOLY_PATH.format(prefix)
        executable_setup = self.ROOT_DIR +  '/train/visualizer-2.0/visualizer.jar'
        band_info = self.ROOT_DIR + '/train/visualizer-2.0/data/band-triplets.txt'
        cmd = [
            'java',
            '-jar',
            executable_setup,
            '-truth',
            truth_file,
            '-solution',
            poly_file,
            '-no-gui',
            '-band-triplets',
            band_info,
            '-image-dir',
            'pass',
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout_data, stderr_data = proc.communicate()
        lines = [line for line in stdout_data.decode('utf8').split('\n')[-10:]]

        """
    Overall F-score : 0.85029

    AOI_2_Vegas:
      TP       : 27827
      FP       : 4999
      FN       : 4800
      Precision: 0.847712
      Recall   : 0.852883
      F-score  : 0.85029
        """

        if stdout_data.decode('utf8').strip().endswith("Overall F-score : 0"):
            overall_fscore = 0
            tp = 0
            fp = 0
            fn = 0
            precision = 0
            recall = 0
            fscore = 0

        elif len(lines) > 0 and lines[0].startswith("Overall F-score : "):
            assert lines[0].startswith("Overall F-score : ")
            assert lines[2].startswith("AOI_")
            assert lines[3].strip().startswith("TP")
            assert lines[4].strip().startswith("FP")
            assert lines[5].strip().startswith("FN")
            assert lines[6].strip().startswith("Precision")
            assert lines[7].strip().startswith("Recall")
            assert lines[8].strip().startswith("F-score")

            overall_fscore = float(re.findall("([\d\.]+)", lines[0])[0])
            tp = int(re.findall("(\d+)", lines[3])[0])
            fp = int(re.findall("(\d+)", lines[4])[0])
            fn = int(re.findall("(\d+)", lines[5])[0])
            precision = float(re.findall("([\d\.]+)", lines[6])[0])
            recall = float(re.findall("([\d\.]+)", lines[7])[0])
            fscore = float(re.findall("([\d\.]+)", lines[8])[0])
        else:
            self.logger.warn("Unexpected data >>> " + stdout_data.decode('utf8'))
            raise RuntimeError("Unsupported format")

        return {
            'overall_fscore': overall_fscore,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'fscore': fscore,
        }

    def _internal_validate_fscore_wo_pred_file(self, area_id, trainer, path, operators, epoch=3, min_th=30,
                                               enable_tqdm=False):
        prefix = self.area_id_to_prefix(area_id)

        # Prediction phase
        self.logger.info("Prediction phase")
        y_pred, image_ids = self._internal_validate_predict(trainer, path, operators)

        # Postprocessing phase
        self.logger.info("Postprocessing phase")
        fn_test = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test, index_col='ImageId')

        # fn = self.FMT_VALTESTPRED_PATH.format(prefix)
        fn_out = self.FMT_VALTESTPOLY_PATH.format(prefix)
        with open(fn_out, 'w') as f:
            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            test_list = df_test.index.tolist()
            iterator = enumerate(test_list)

            for idx, image_id in tqdm.tqdm(iterator, total=len(test_list)):
                df_poly = self.mask_to_poly(y_pred[idx][0], min_polygon_area_th=min_th)
                if len(df_poly) > 0:
                    for i, row in df_poly.iterrows():
                        line = "{},{},\"{}\",{:.6f}\n".format(
                            image_id,
                            row.bid,
                            row.wkt,
                            row.area_ratio)
                        line = self._remove_interiors(line)
                        f.write(line)
                else:
                    f.write("{},{},{},0\n".format(
                        image_id,
                        -1,
                        "POLYGON EMPTY"))

        # ------------------------
        # Validation solution file
        self.logger.info("Validation solution file")
        fn_true = self.FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
        df_true = pd.read_csv(fn_true)

        # # Remove prefix "PAN_"
        # df_true.loc[:, 'ImageId'] = df_true.ImageId.str[4:]

        fn_test = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test)
        df_test_image_ids = df_test.ImageId.unique()

        fn_out = self.FMT_VALTESTTRUTH_PATH.format(prefix)
        with open(fn_out, 'w') as f:
            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            df_true = df_true[df_true.ImageId.isin(df_test_image_ids)]
            for idx, r in df_true.iterrows():
                f.write("{},{},\"{}\",{:.6f}\n".format(
                    r.ImageId,
                    r.BuildingId,
                    r.PolygonWKT_Pix,
                    1.0))

    def _internal_validate_fscore_wo_pred_file_v12(self, area_id, trainer, path, operators, epoch=3, min_th=30,
                                               enable_tqdm=False):
        prefix = self.area_id_to_prefix(area_id)

        # Prediction phase
        self.logger.info("Prediction phase")
        y_pred, image_ids = self._internal_validate_predict(trainer, path, operators)

        # Postprocessing phase
        self.logger.info("Postprocessing phase")
        fn_test = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test, index_col='ImageId')

        # fn = self.FMT_VALTESTPRED_PATH.format(prefix)
        fn_out = self.FMT_VALTESTPOLY_PATH.format(prefix)
        with open(fn_out, 'w') as f:
            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            test_list = df_test.index.tolist()
            iterator = enumerate(test_list)

            for idx, image_id in tqdm.tqdm(iterator, total=len(test_list)):
                pred_values = np.zeros((650, 650))
                pred_count = np.zeros((650, 650))
                for slice_pos in range(9):
                    slice_idx = idx * 9 + slice_pos
                    pos_j = int(math.floor(slice_pos / 3.0))
                    pos_i = int(slice_pos % 3)
                    x0 = self.STRIDE_SZ * pos_i
                    y0 = self.STRIDE_SZ * pos_j
                    mask_pred = skimage.transform.resize(y_pred[slice_idx][0], (256,256))
                    pred_values[x0:x0 + self.INPUT_SIZE, y0:y0 + self.INPUT_SIZE] += (mask_pred)
                    pred_count[x0:x0 + self.INPUT_SIZE, y0:y0 + self.INPUT_SIZE] += 1
                pred_values = pred_values / pred_count

                df_poly = self.mask_to_poly(pred_values, min_polygon_area_th=min_th)
                if len(df_poly) > 0:
                    for i, row in df_poly.iterrows():
                        line = "{},{},\"{}\",{:.6f}\n".format(
                            image_id,
                            row.bid,
                            row.wkt,
                            row.area_ratio)
                        line = self._remove_interiors(line)
                        f.write(line)
                else:
                    f.write("{},{},{},0\n".format(
                        image_id,
                        -1,
                        "POLYGON EMPTY"))

        # ------------------------
        # Validation solution file
        self.logger.info("Validation solution file")
        fn_true = self.FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
        df_true = pd.read_csv(fn_true)

        # # Remove prefix "PAN_"
        # df_true.loc[:, 'ImageId'] = df_true.ImageId.str[4:]

        fn_test = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test)
        df_test_image_ids = df_test.ImageId.unique()

        fn_out = self.FMT_VALTESTTRUTH_PATH.format(prefix)
        with open(fn_out, 'w') as f:
            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            df_true = df_true[df_true.ImageId.isin(df_test_image_ids)]
            for idx, r in df_true.iterrows():
                f.write("{},{},\"{}\",{:.6f}\n".format(
                    r.ImageId,
                    r.BuildingId,
                    r.PolygonWKT_Pix,
                    1.0))


    def _internal_validate_fscore_wo_pred_file_v16(self, area_id, trainer, path, operators, epoch=3, min_th=30,
                                               enable_tqdm=False):
        prefix = self.area_id_to_prefix(area_id)

        # Prediction phase
        self.logger.info("Prediction phase")
        y_pred, image_ids = self._internal_validate_predict(trainer, path, operators)

        # Postprocessing phase
        self.logger.info("Postprocessing phase")
        fn_test = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test, index_col='ImageId')

        # fn = self.FMT_VALTESTPRED_PATH.format(prefix)
        fn_out = self.FMT_VALTESTPOLY_PATH.format(prefix)
        with open(fn_out, 'w') as f:
            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            test_list = df_test.index.tolist()
            iterator = enumerate(test_list)

            for idx, image_id in tqdm.tqdm(iterator, total=len(test_list)):
                pred_values = np.zeros((650, 650))
                pred_count = np.zeros((650, 650))
                for slice_pos in range(9):
                    slice_idx = idx * 9 + slice_pos
                    pos_j = int(math.floor(slice_pos / 3.0))
                    pos_i = int(slice_pos % 3)
                    x0 = self.STRIDE_SZ * pos_i
                    y0 = self.STRIDE_SZ * pos_j
                    mask_pred = skimage.transform.resize(y_pred[slice_idx][0], (256,256))
                    pred_values[x0:x0 + self.INPUT_SIZE, y0:y0 + self.INPUT_SIZE] += (mask_pred)
                    pred_count[x0:x0 + self.INPUT_SIZE, y0:y0 + self.INPUT_SIZE] += 1
                pred_values = pred_values / pred_count

                df_poly = self.mask_to_poly(pred_values, min_polygon_area_th=min_th)
                if len(df_poly) > 0:
                    for i, row in df_poly.iterrows():
                        line = "{},{},\"{}\",{:.6f}\n".format(
                            image_id,
                            row.bid,
                            row.wkt,
                            row.area_ratio)
                        line = self._remove_interiors(line)
                        f.write(line)
                else:
                    f.write("{},{},{},0\n".format(
                        image_id,
                        -1,
                        "POLYGON EMPTY"))

        # ------------------------
        # Validation solution file
        self.logger.info("Validation solution file")
        fn_true = self.FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
        df_true = pd.read_csv(fn_true)

        # # Remove prefix "PAN_"
        # df_true.loc[:, 'ImageId'] = df_true.ImageId.str[4:]

        fn_test = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test)
        df_test_image_ids = df_test.ImageId.unique()

        fn_out = self.FMT_VALTESTTRUTH_PATH.format(prefix)
        with open(fn_out, 'w') as f:
            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            df_true = df_true[df_true.ImageId.isin(df_test_image_ids)]
            for idx, r in df_true.iterrows():
                f.write("{},{},\"{}\",{:.6f}\n".format(
                    r.ImageId,
                    r.BuildingId,
                    r.PolygonWKT_Pix,
                    1.0))


    def mask_to_poly(self, mask, min_polygon_area_th=30):
        """
        Convert from 256x256 mask to polygons on 650x650 image
        """
        mask = (skimage.transform.resize(mask, (650, 650)) > 0.5).astype(np.uint8)
        shapes = rasterio.features.shapes(mask.astype(np.int16), mask > 0)
        poly_list = []
        mp = shapely.ops.cascaded_union(
            shapely.geometry.MultiPolygon([
                shapely.geometry.shape(shape)
                for shape, value in shapes
            ]))

        if isinstance(mp, shapely.geometry.Polygon):
            df = pd.DataFrame({
                'area_size': [mp.area],
                'poly': [mp],
            })
        else:
            df = pd.DataFrame({
                'area_size': [p.area for p in mp],
                'poly': [p for p in mp],
            })

        df = df[df.area_size > min_polygon_area_th].sort_values(
            by='area_size', ascending=False)
        df.loc[:, 'wkt'] = df.poly.apply(lambda x: shapely.wkt.dumps(
            x, rounding_precision=0))
        df.loc[:, 'bid'] = list(range(1, len(df) + 1))
        df.loc[:, 'area_ratio'] = df.area_size / df.area_size.max()
        return df

    def area_id_to_osmprefix(self, area_id):
        area_id_to_osmprefix_dict = {
            2: 'las-vegas_nevada_osm',
            3: 'paris_france_osm',
            4: 'shanghai_china_osm',
            5: 'ex_e2dQZwVJMniofruzC8HVKsw1dc8uk_osm',
        }
        return area_id_to_osmprefix_dict[area_id]

    def tif_to_latlon(self, path):
        ds = gdal.Open(path)
        width = ds.RasterXSize
        height = ds.RasterYSize
        gt = ds.GetGeoTransform()
        minx = gt[0]
        miny = gt[3] + width * gt[4] + height * gt[5]
        maxx = gt[0] + width * gt[1] + height * gt[2]
        maxy = gt[3]
        return Bunch(
            minx=minx,
            maxx=maxx,
            miny=miny,
            maxy=maxy,
            xcenter=(minx + maxx) / 2.0,
            ycenter=(miny + maxy) / 2.0)

    def location_summary(self, area_id, datapath):
        area_prefix = self.area_id_to_prefix(area_id)
        rows = []

        glob_path = str(
            Path(datapath) /
            Path("PAN/PAN_{prefix:s}_img*.tif")
        ).format(prefix=area_prefix)

        for path in sorted(glob.glob(glob_path)):
            image_id = path.split('/')[-1][:-4]
            pos = self.tif_to_latlon(path)
            rows.append(dict(ImageId=image_id, path=path, pos=pos))

        df_location = pd.DataFrame(rows)
        df_location.loc[:, 'xcenter'] = df_location.pos.apply(lambda x: x.xcenter)
        df_location.loc[:, 'ycenter'] = df_location.pos.apply(lambda x: x.ycenter)
        return df_location

    def location_summary_test(self, area_id, datapath):
        area_prefix = self.area_id_to_prefix(area_id)
        rows = []
        glob_path = str(
            Path(datapath) /
            Path("PAN/PAN_{prefix:s}_img*.tif")
        ).format(prefix=area_prefix)

        for path in sorted(glob.glob(glob_path)):
            image_id = path.split('/')[-1][:-4]
            pos = self.tif_to_latlon(path)
            rows.append(dict(ImageId=image_id, path=path, pos=pos))

        df_location = pd.DataFrame(rows)
        df_location.loc[:, 'xcenter'] = df_location.pos.apply(lambda x: x.xcenter)
        df_location.loc[:, 'ycenter'] = df_location.pos.apply(lambda x: x.ycenter)
        return df_location

    def preproc_osm(self, area_id, datapath, is_train=True):
        self.logger.info("Loading raster...")
        osmprefix = self.area_id_to_osmprefix(area_id)

        # df = pd.concat([
        #     location_summary(area_id),
        #     location_summary_test(area_id),
        # ])
        if is_train:
            df = self.location_summary(area_id, datapath)
        else:
            df = self.location_summary_test(area_id, datapath)

        map_bound = Bunch(
            left=df.sort_values(by='xcenter').iloc[-1]['pos']['maxx'],
            right=df.sort_values(by='xcenter').iloc[0]['pos']['minx'],
            top=df.sort_values(by='ycenter').iloc[-1]['pos']['maxy'],
            bottom=df.sort_values(by='ycenter').iloc[0]['pos']['miny'],
        )
        geom_layers = {}

        fn_osm = self.FMT_SERIALIZED_OSMDATA.format(osmprefix)
        if not Path(fn_osm).exists():
            for layer_name in self.LAYER_NAMES:
                fn_shp = self.FMT_OSMSHAPEFILE.format(
                    name=osmprefix,
                    layer=layer_name)

                if not Path(fn_shp).exists():
                    raise RuntimeError("shp not found: {}".format(fn_shp))

                geom_bounds = shapely.geometry.Polygon([
                    (map_bound.left, map_bound.top),
                    (map_bound.right, map_bound.top),
                    (map_bound.right, map_bound.bottom),
                    (map_bound.left, map_bound.bottom),
                ])
                with fiona.open(fn_shp, 'r') as vector:
                    print("{}: {}".format(layer_name, len(vector)))
                    geoms = []
                    for feat in tqdm.tqdm(vector, total=len(vector)):
                        try:
                            geom = shapely.geometry.shape(feat['geometry'])
                            isec_area = geom.intersection(geom_bounds).area
                            if isec_area > 0:
                                geoms.append([
                                    geom, 'area', feat['properties'],
                                ])
                            elif geom.intersects(geom_bounds):
                                geoms.append([
                                    geom, 'line', feat['properties'],
                                ])
                        except:
                            pass

                    print("{}: {} -> {}".format(
                        layer_name,
                        len(vector),
                        len(geoms)))
                    geom_layers[layer_name] = geoms

            with open(fn_osm, 'wb') as f:
                pickle.dump(geom_layers, f)


    def preproc_train_v16(self):
        """ train.sh """
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        osmprefix = self.area_id_to_osmprefix(area_id)

        # Mkdir
        if not Path(self.FMT_VALTRAIN_OSM_STORE.format(prefix)).parent.exists():
            Path(self.FMT_VALTRAIN_OSM_STORE.format(prefix)).parent.mkdir(
                parents=True)

        # OSM serialized subset
        fn_osm = self.FMT_SERIALIZED_OSMDATA.format(osmprefix)
        if Path(fn_osm).exists():
            self.logger.info("Serialize OSM subset ... skip")
        else:
            self.logger.info("Serialize OSM subset")
            self.preproc_osm(area_id, self.DATA_PATH, is_train=True)

        # OSM layers (valtrain)
        if Path(self.FMT_VALTRAIN_OSM_STORE.format(prefix)).exists():
            self.logger.info("Generate OSM_STORE (valtrain) ... skip")
        else:
            self.logger.info("Generate OSM_STORE (valtrain)")
            self.prep_osmlayer_train(area_id, self.DATA_PATH, is_valtrain=True)

        # OSM layers (valtest)
        if Path(self.FMT_VALTEST_OSM_STORE.format(prefix)).exists():
            self.logger.info("Generate OSM_STORE (valtest) ... skip")
        else:
            self.logger.info("Generate OSM_STORE (valtest)")
            self.prep_osmlayer_train(area_id, self.DATA_PATH, is_valtrain=False)

    def _internal_validate_predict(self, trainer, path, operators, save_pred=True):
        return trainer.predictor(path, operators)

    def get_mapzen_osm_name(self, area_id):
        area_id_to_mapzen_name = {
            2: 'las-vegas_nevada_osm',
            3: 'paris_france_osm',
            4: 'shanghai_china_osm',
            5: 'ex_e2dQZwVJMniofruzC8HVKsw1dc8uk_osm',
        }
        mapzen_name = area_id_to_mapzen_name[area_id]
        return mapzen_name

    def extract_buildings_geoms(self, area_id):
        mapzen_name = self.get_mapzen_osm_name(area_id)
        fn_osm = self.FMT_SERIALIZED_OSMDATA.format(mapzen_name)
        with open(fn_osm, 'rb') as f:
            osm = pickle.load(f)

        geoms = [
            geom
            for geom, type_name, properties in osm['buildings']
            if type_name == 'area'
        ]
        return geoms

    def extract_waterarea_geoms(self, area_id):
        mapzen_name = self.get_mapzen_osm_name(area_id)
        fn_osm = self.FMT_SERIALIZED_OSMDATA.format(mapzen_name)
        with open(fn_osm, 'rb') as f:
            osm = pickle.load(f)

        geoms = [
            geom
            for geom, type_name, properties in osm['waterareas']
            if type_name == 'area'
        ]
        return geoms

    def extract_landusages_industrial_geoms(self, area_id):
        mapzen_name = self.get_mapzen_osm_name(area_id)
        fn_osm = self.FMT_SERIALIZED_OSMDATA.format(mapzen_name)
        with open(fn_osm, 'rb') as f:
            osm = pickle.load(f)

        geoms = [
            geom
            for geom, type_name, properties in osm['landusages']
            if type_name == 'area' and properties['type'] == 'industrial'
        ]
        return geoms

    def extract_landusages_farm_and_forest_geoms(self, area_id):
        mapzen_name = self.get_mapzen_osm_name(area_id)
        fn_osm = self.FMT_SERIALIZED_OSMDATA.format(mapzen_name)
        with open(fn_osm, 'rb') as f:
            osm = pickle.load(f)

        geoms = [
            geom
            for geom, type_name, properties in osm['landusages']
            if type_name == 'area' and properties['type'] in [
                'forest',
                'farmyard',
            ]
        ]
        return geoms

    def extract_landusages_residential_geoms(self, area_id):
        mapzen_name = self.get_mapzen_osm_name(area_id)
        fn_osm = self.FMT_SERIALIZED_OSMDATA.format(mapzen_name)
        with open(fn_osm, 'rb') as f:
            osm = pickle.load(f)

        geoms = [
            geom
            for geom, type_name, properties in osm['landusages']
            if type_name == 'area' and properties['type'] == 'residential'
        ]
        return geoms

    def extract_roads_geoms(self, area_id):
        mapzen_name = self.get_mapzen_osm_name(area_id)
        fn_osm = self.FMT_SERIALIZED_OSMDATA.format(mapzen_name)
        with open(fn_osm, 'rb') as f:
            osm = pickle.load(f)

        geoms = [
            geom
            for geom, type_name, properties in osm['roads']
            if type_name == 'line' and properties['type'] != 'subway'
        ]
        return geoms

    def extract_osmlayers(self, area_id):
        if area_id == 2:
            return [
                self.extract_buildings_geoms(area_id),
                self.extract_landusages_industrial_geoms(area_id),
                self.extract_landusages_residential_geoms(area_id),
                self.extract_roads_geoms(area_id),
            ]
        elif area_id == 3:
            return [
                self.extract_buildings_geoms(area_id),
                self.extract_landusages_farm_and_forest_geoms(area_id),
                self.extract_landusages_industrial_geoms(area_id),
                self.extract_landusages_residential_geoms(area_id),
                self.extract_roads_geoms(area_id),
            ]
        elif area_id == 4:
            return [
                self.extract_waterarea_geoms(area_id),
                self.extract_landusages_industrial_geoms(area_id),
                self.extract_landusages_residential_geoms(area_id),
                self.extract_roads_geoms(area_id),
            ]
        elif area_id == 5:
            return [
                self.extract_waterarea_geoms(area_id),
                self.extract_landusages_industrial_geoms(area_id),
                self.extract_landusages_residential_geoms(area_id),
                self.extract_roads_geoms(area_id),
            ]
        else:
            raise RuntimeError("area_id must be in range(2, 6): {}".foramt(area_id))

    def prep_osmlayer_train(self, area_id, datapath, is_valtrain=False):
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("prep_osmlayer_train for {}".format(prefix))

        if is_valtrain:
            fn_list = self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
            fn_store = self.FMT_VALTRAIN_OSM_STORE.format(prefix)
        else:
            fn_list = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
            fn_store = self.FMT_VALTEST_OSM_STORE.format(prefix)

        layers = self.extract_osmlayers(area_id)

        df = pd.read_csv(fn_list, index_col='ImageId')
        self.logger.info("Prep osm container: {}".format(fn_store))
        with tb.open_file(fn_store, 'w') as f:
            df_sz = len(df)
            for image_id in tqdm.tqdm(df.index, total=df_sz):
                # fn_tif = train_image_id_to_path(image_id)
                fn_tif = self.get_train_image_path_from_imageid(image_id, datapath, mul=False)
                with rasterio.open(fn_tif, 'r') as fr:
                    values = fr.read(1)
                    masks = []  # rasterize masks
                    for layer_geoms in layers:
                        mask = rasterio.features.rasterize(
                            layer_geoms,
                            out_shape=values.shape,
                            transform=rasterio.guard_transform(
                                fr.transform))
                        masks.append(mask)
                    masks = np.array(masks)
                    masks = np.swapaxes(masks, 0, 2)
                    masks = np.swapaxes(masks, 0, 1)
                assert masks.shape == (650, 650, len(layers))

                # slice of masks
                for slice_pos in range(9):
                    pos_j = int(math.floor(slice_pos / 3.0))
                    pos_i = int(slice_pos % 3)
                    x0 = self.STRIDE_SZ * pos_i
                    y0 = self.STRIDE_SZ * pos_j
                    im = masks[x0:x0 + self.INPUT_SIZE, y0:y0 + self.INPUT_SIZE]
                    assert im.shape == (256, 256, len(layers))

                    slice_id = image_id + "_{}".format(slice_pos)
                    atom = tb.Atom.from_dtype(im.dtype)
                    filters = tb.Filters(complib='blosc', complevel=9)
                    ds = f.create_carray(f.root,
                                         slice_id,
                                         atom,
                                         im.shape,
                                         filters=filters)
                    ds[:] = im

    def get_mul_mean_image(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        with tb.open_file(self.FMT_MULMEAN.format(prefix), 'r') as f:
            im_mean = np.array(f.get_node('/mulmean'))
        return im_mean

    def preproc_train_v8(self):
        """ train.sh """
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("Preproc for training on {}".format(prefix))

        # Working directory
        working_dir = Path(self.FMT_VALTRAIN_MASK_STORE.format(prefix)).parent
        if not working_dir.exists():
            working_dir.mkdir(parents=True)

        # Imagelist (from v5)
        assert Path(self.FMT_VALTRAIN_IMAGELIST_PATH.format(
            prefix=prefix)).exists()
        assert Path(self.FMT_VALTEST_IMAGELIST_PATH.format(
            prefix=prefix)).exists()

        # Band stats (MUL)
        assert Path(self.FMT_MUL_BANDCUT_TH_PATH.format(prefix)).exists()

        # Mask (Target output)
        if Path(self.FMT_VALTRAIN_MASK_STORE.format(prefix)).exists():
            self.logger.info("Generate MASK (valtrain) ... skip")
        else:
            self.logger.info("Generate MASK (valtrain)")
            self.prep_image_mask_v8(area_id, is_valtrain=True)
        if Path(self.FMT_VALTEST_MASK_STORE.format(prefix)).exists():
            self.logger.info("Generate MASK (valtest) ... skip")
        else:
            self.logger.info("Generate MASK (valtest)")
            self.prep_image_mask_v12(area_id, is_valtrain=False)

        # Image HDF5 store (MUL)
        if Path(self.FMT_VALTRAIN_MUL_STORE.format(prefix)).exists():
            self.logger.info("Generate MUL_STORE (valtrain) ... skip")
        else:
            self.logger.info("Generate MUL_STORE (valtrain)")
            self.prep_mul_image_only_store_train(area_id, self.DATA_PATH, is_valtrain=True)
        if Path(self.FMT_VALTEST_MUL_STORE.format(prefix)).exists():
            self.logger.info("Generate MUL_STORE (valtest) ... skip")
        else:
            self.logger.info("Generate MUL_STORE (valtest)")
            self.prep_mul_image_only_store_train(area_id, self.DATA_PATH, is_valtrain=False)

        # Image Mean (MUL)
        if Path(self.FMT_MULMEAN.format(prefix)).exists():
            self.logger.info("Generate MULMEAN ... skip")
        else:
            self.logger.info("Generate MULMEAN")
            # self.prep_mulmean(area_id)
            self.prep_mulmean_v12(area_id)

        # DONE!
        self.logger.info("Preproc for training on {} ... done".format(prefix))


    def preproc_train_v12(self):
        """ train.sh """
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("Preproc for training on {}".format(prefix))

        # Working directory
        working_dir = Path(self.FMT_VALTRAIN_MASK_STORE.format(prefix)).parent
        if not working_dir.exists():
            working_dir.mkdir(parents=True)

        # Imagelist (from v5)
        assert Path(self.FMT_VALTRAIN_IMAGELIST_PATH.format(
            prefix=prefix)).exists()
        assert Path(self.FMT_VALTEST_IMAGELIST_PATH.format(
            prefix=prefix)).exists()

        # Band stats (MUL)
        assert Path(self.FMT_MUL_BANDCUT_TH_PATH.format(prefix)).exists()

        # Mask (Target output)
        if Path(self.FMT_VALTRAIN_MASK_STORE.format(prefix)).exists():
            self.logger.info("Generate MASK (valtrain) ... skip")
        else:
            self.logger.info("Generate MASK (valtrain)")
            self.prep_image_mask_v12(area_id, is_valtrain=True)
        if Path(self.FMT_VALTEST_MASK_STORE.format(prefix)).exists():
            self.logger.info("Generate MASK (valtest) ... skip")
        else:
            self.logger.info("Generate MASK (valtest)")
            self.prep_image_mask_v12(area_id, is_valtrain=False)

        # Image HDF5 store (MUL)
        if Path(self.FMT_VALTRAIN_MUL_STORE.format(prefix)).exists():
            self.logger.info("Generate MUL_STORE (valtrain) ... skip")
        else:
            self.logger.info("Generate MUL_STORE (valtrain)")
            self.prep_mul_image_only_store_train(area_id, self.DATA_PATH, is_valtrain=True)
        if Path(self.FMT_VALTEST_MUL_STORE.format(prefix)).exists():
            self.logger.info("Generate MUL_STORE (valtest) ... skip")
        else:
            self.logger.info("Generate MUL_STORE (valtest)")
            self.prep_mul_image_only_store_train(area_id, self.DATA_PATH, is_valtrain=False)

        # Image Mean (MUL)
        if Path(self.FMT_MULMEAN.format(prefix)).exists():
            self.logger.info("Generate MULMEAN ... skip")
        else:
            self.logger.info("Generate MULMEAN")
            # self.prep_mulmean(area_id)
            self.prep_mulmean_v12(area_id)

        # DONE!
        self.logger.info("Preproc for training on {} ... done".format(prefix))

    def validate(self, trainer, operators, training_iters=4, display_step=2, restore=True):
        model_name = "model_" + self.MODEL_NAME
        epochs_for_vaidating = int(self.plugin_config[model_name]["validate"]["epochs"])
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info(">> validate sub-command: {}".format(prefix))
        model_path=self.MODEL_DIR + "/" + prefix
        if not Path(model_path).exists():
            Path(model_path).mkdir(parents=True)
        self.logger.info("load valtrain")
        trainer.train(operators, model_path, training_iters, epochs_for_vaidating, display_step, restore)

    def generate_valtest_batch(self, writer):
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        df_train = pd.read_csv(self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix))
        fn_im = self.FMT_VALTEST_MUL_STORE.format(prefix)
        fn_mask = self.FMT_VALTEST_MASK_STORE.format(prefix)
        fn_osm = self.FMT_VALTEST_OSM_STORE.format(prefix)
        self.logger.info(">> validate sub-command: {}".format(prefix))
        dict_n_osm_layers = {
            2: 4,
            3: 5,
            4: 4,
            5: 4,
        }
        osm_layers = dict_n_osm_layers[area_id]
        n_input_layers = 8 + osm_layers
        self.logger.info("Validate step for {}".format(prefix))
        X_mean = self.get_mul_mean_image(area_id)
        X_osm_mean = np.zeros((osm_layers, self.INPUT_SIZE, self.INPUT_SIZE))
        immean = np.vstack([X_mean, X_osm_mean])

        with tb.open_file(fn_im, 'r') as f_im, \
                tb.open_file(fn_osm, 'r') as f_osm, \
                tb.open_file(fn_mask, 'r') as f_mask:
                    for idx, image_id in tqdm.tqdm(enumerate(df_train.ImageId.tolist())):
                        for slice_pos in range(9):
                            slice_id = image_id + "_" + str(slice_pos)
                            im = np.array(f_im.get_node('/' + slice_id))
                            im = np.swapaxes(im, 0, 2)
                            im = np.swapaxes(im, 1, 2)
                            im2 = np.array(f_osm.get_node('/' + slice_id))
                            im2 = np.swapaxes(im2, 0, 2)
                            im2 = np.swapaxes(im2, 1, 2)

                            im = np.vstack([im, im2])
                            im =  im - immean
                            im = np.transpose(im).astype(np.float64)
                            mask = np.array(f_mask.get_node('/' + slice_id))
                            mask = (mask > 0).astype(np.uint8)

                            image_id_prefix = prefix + "_img"
                            image_id_index = image_id.replace(image_id_prefix, "")
                            image_id_index = int(image_id_index)
                            self.createRFRecoad(im, mask, image_id_index, writer)
        writer.close()
        self.logger.info("TFRecoad for validating for v16 images have been written to "
                         + self.tfrecords_filename_multi_train)
        return True

    def generate_valtrain_batch(self, writer):
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        df_train = pd.read_csv(self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix))
        fn_im = self.FMT_VALTRAIN_MUL_STORE.format(prefix)
        fn_mask = self.FMT_VALTRAIN_MASK_STORE.format(prefix)
        fn_osm = self.FMT_VALTRAIN_OSM_STORE.format(prefix)
        self.logger.info(">> validate sub-command: {}".format(prefix))
        dict_n_osm_layers = {
            2: 4,
            3: 5,
            4: 4,
            5: 4,
        }
        osm_layers = dict_n_osm_layers[area_id]
        self.logger.info("Validate step for {}".format(prefix))
        X_mean = self.get_mul_mean_image(area_id)
        X_osm_mean = np.zeros((osm_layers, self.INPUT_SIZE, self.INPUT_SIZE))
        immean = np.vstack([X_mean, X_osm_mean])

        with tb.open_file(fn_im, 'r') as f_im, \
                tb.open_file(fn_osm, 'r') as f_osm, \
                tb.open_file(fn_mask, 'r') as f_mask:
                    for idx, image_id in tqdm.tqdm(enumerate(df_train.ImageId.tolist())):
                        for slice_pos in range(9):
                            slice_id = image_id + "_" + str(slice_pos)
                            im = np.array(f_im.get_node('/' + slice_id))
                            im = np.swapaxes(im, 0, 2)
                            im = np.swapaxes(im, 1, 2)
                            im2 = np.array(f_osm.get_node('/' + slice_id))
                            im2 = np.swapaxes(im2, 0, 2)
                            im2 = np.swapaxes(im2, 1, 2)

                            im = np.vstack([im, im2])
                            im =  im - immean
                            im = np.transpose(im).astype(np.float64)
                            mask = np.array(f_mask.get_node('/' + slice_id))
                            mask = (mask > 0).astype(np.uint8)

                            image_id_prefix = prefix + "_img"
                            image_id_index = image_id.replace(image_id_prefix, "")
                            image_id_index = int(image_id_index)
                            self.createRFRecoad(im, mask, image_id_index, writer)
        writer.close()
        self.logger.info("TFRecoad for validating for v16 images have been written to "
                         + self.tfrecords_filename_multi_test)
        return True

    def validate_v16(self, datapath):
        if not Path(self.MODEL_DIR).exists():
            Path(self.MODEL_DIR).mkdir(parents=True)
        self.logger.info("load valtrain")
        self.logger.info("Instantiate U-Net model")

    def get_valtest_data_v12(self, writer):
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        fn_test = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test)
        fn_im = self.FMT_VALTEST_MUL_STORE.format(prefix)
        fn_mask = self.FMT_VALTEST_MASK_STORE.format(prefix)

        with tb.open_file(fn_im, 'r') as image_list:
            with tb.open_file(fn_mask, 'r') as masks_list:
                for idx, image_id in tqdm.tqdm(enumerate(df_test.ImageId.tolist())):
                    for slice_pos in range(9):
                        slice_id = image_id + "_" + str(slice_pos)
                        im = np.array(image_list.get_node('/' + slice_id))
                        im = np.swapaxes(im, 0, 2)
                        im = np.swapaxes(im, 1, 2)
                        mask = np.array(masks_list.get_node('/' + slice_id))
                        mask = (mask > 0.5).astype(np.uint8)
                        im = np.transpose(im).astype(np.float64)
                        image_id_prefix = prefix + "_img"
                        image_id_index = image_id.replace(image_id_prefix, "")
                        image_id_index = int(image_id_index)
                        self.createRFRecoad(im, mask, image_id_index, writer)
        writer.close()
        self.logger.info("TFRecoad for validating for model v12 images have been written to "
                         + self.tfrecords_filename_multi_test)
        return True

    def get_valtrain_data_v12(self, writer):
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        fn_train = self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
        df_train = pd.read_csv(fn_train)
        fn_im = self.FMT_VALTRAIN_MUL_STORE.format(prefix)
        fn_mask = self.FMT_VALTRAIN_MASK_STORE.format(prefix)

        with tb.open_file(fn_im, 'r') as image_list:
            with tb.open_file(fn_mask, 'r') as masks_list:
                for idx, image_id in tqdm.tqdm(enumerate(df_train.ImageId.tolist())):
                    for slice_pos in range(9):
                        slice_id = image_id + "_" + str(slice_pos)
                        im = np.array(image_list.get_node('/' + slice_id))
                        im = np.swapaxes(im, 0, 2)
                        im = np.swapaxes(im, 1, 2)
                        mask = np.array(masks_list.get_node('/' + slice_id))
                        mask = (mask > 0.5).astype(np.uint8)
                        im = np.transpose(im).astype(np.float64)
                        image_id_prefix = prefix + "_img"
                        image_id_index = image_id.replace(image_id_prefix, "")
                        image_id_index = int(image_id_index)
                        self.createRFRecoad(im, mask, image_id_index, writer)
        writer.close()
        self.logger.info("TFRecoad for training for model v12 images has been written to "
                         + self.tfrecords_filename_multi_train)
        return True

    def _get_model_parameter(self, area_id, model_name):
        prefix = self.area_id_to_prefix(area_id)
        FMT_VALMODEL_EVALTHHIST = self.FMT_VALMODEL_EVALTHHIST.replace(self.MODEL_NAME, model_name)
        fn_hist = FMT_VALMODEL_EVALTHHIST.format(prefix)
        best_row = pd.read_csv(fn_hist).sort_values(
            by='fscore',
            ascending=False,
        ).iloc[0]

        param = dict(
            fn_epoch=int(best_row['zero_base_epoch']),
            min_poly_area=int(best_row['min_area_th']),
        )
        return param

    def _internal_validate_predict_best_param(self, model_name, trainer, operators,
                                              enable_tqdm=False):
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        param = self._get_model_parameter(area_id, model_name)
        epoch = param['fn_epoch']
        path = self.MODEL_DIR +"/"+prefix + "/model-" + str(epoch)
        y_pred, images_ids = self._internal_validate_predict(trainer, path, operators)
        return y_pred, images_ids

    def _internal_validate_predict_best_param_for_all(self, area_id,
                                              save_pred=True,
                                              enable_tqdm=False,
                                              rescale_pred_list=[],
                                              slice_pred_list=[]):
        prefix = self.area_id_to_prefix(area_id)

        # Load valtest imagelist
        fn_valtest = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_valtest = pd.read_csv(fn_valtest, index_col='ImageId')

        pred_values_array = np.zeros((len(df_valtest), 650, 650))
        for idx, image_id in enumerate(df_valtest.index.tolist()):
            pred_values = np.zeros((650, 650))
            pred_count = np.zeros((650, 650))
            for slice_pos in range(9):
                slice_idx = idx * 9 + slice_pos
                pos_j = int(math.floor(slice_pos / 3.0))
                pos_i = int(slice_pos % 3)
                x0 = self.STRIDE_SZ * pos_i
                y0 = self.STRIDE_SZ * pos_j
                for slice_pred in slice_pred_list:
                    mask_a = skimage.transform.resize(slice_pred[slice_idx][0], (256, 256))
                    pred_values[x0:x0 + self.INPUT_SIZE, y0:y0 + self.INPUT_SIZE] += (mask_a)
                    pred_count[x0:x0 + self.INPUT_SIZE, y0:y0 + self.INPUT_SIZE] += 1

            for rescale_pred in rescale_pred_list:
                y_pred_idx = skimage.transform.resize(rescale_pred[idx][0], (650, 650))
                pred_values += y_pred_idx
            pred_count += 1

            # Normalize
            pred_values = pred_values / pred_count
            pred_values_array[idx, :, :] = pred_values

        return pred_values_array

    def _internal_pred_to_poly_file_final(self, area_id,
                                    y_pred,
                                    min_th=30):
        """
        Write out valtest poly and truepoly
        """
        prefix = self.area_id_to_prefix(area_id)

        # Load valtest imagelist
        fn_valtest = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_valtest = pd.read_csv(fn_valtest, index_col='ImageId')

        # Make parent directory
        fn_out = self.FMT_VALTESTPOLY_PATH.format(prefix)
        if not Path(fn_out).parent.exists():
            Path(fn_out).parent.mkdir(parents=True)

        # Ensemble individual models and write out output files
        with open(fn_out, 'w') as f:
            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            for idx, image_id in enumerate(df_valtest.index.tolist()):
                df_poly = self.mask_to_poly(y_pred[idx], min_polygon_area_th=min_th)
                if len(df_poly) > 0:
                    for i, row in df_poly.iterrows():
                        line = "{},{},\"{}\",{:.6f}\n".format(
                            image_id,
                            row.bid,
                            row.wkt,
                            row.area_ratio)
                        line = self._remove_interiors(line)
                        f.write(line)
                else:
                    f.write("{},{},{},0\n".format(
                        image_id,
                        -1,
                        "POLYGON EMPTY"))

        # Validation solution file
        fn_true = self.FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
        df_true = pd.read_csv(fn_true)

        # # Remove prefix "PAN_"
        # df_true.loc[:, 'ImageId'] = df_true.ImageId.str[4:]

        fn_valtest = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_valtest = pd.read_csv(fn_valtest)
        df_valtest_image_ids = df_valtest.ImageId.unique()

        fn_out = self.FMT_VALTESTTRUTH_PATH.format(prefix)
        with open(fn_out, 'w') as f:
            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            df_true = df_true[df_true.ImageId.isin(df_valtest_image_ids)]
            for idx, r in df_true.iterrows():
                line = "{},{},\"{}\",{:.6f}\n".format(
                    r.ImageId,
                    r.BuildingId,
                    r.PolygonWKT_Pix,
                    1.0)
                f.write(line)

    def evalfscore_v17(self, y_pred_0, y_pred_1, y_pred_2):
        area_id = self.directory_name_to_area_id(self.DATA_PATH)
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("Evaluate fscore on validation set: {}".format(prefix))
        self.logger.info("Averaging")
        y_pred = self._internal_validate_predict_best_param_for_all(area_id, rescale_pred_list=[y_pred_0],
            slice_pred_list=[y_pred_1, y_pred_2])

        fn_out = self.FMT_VALTESTPOLY_PATH.format(prefix)
        if not Path(fn_out).parent.exists():
            Path(fn_out).parent.mkdir(parents=True)

        # optimize min area th
        rows = []
        for th in [30, 60, 90, 120, 150, 180, 210, 240]:
            self.logger.info(">>> TH: {}".format(th))
            # predict_flag = True
            self._internal_pred_to_poly_file_final( area_id, y_pred, min_th=th)
            evaluate_record = self._calc_fscore_per_aoi(area_id)
            evaluate_record['min_area_th'] = th
            evaluate_record['area_id'] = area_id
            self.logger.info("\n" + json.dumps(evaluate_record, indent=4))
            rows.append(evaluate_record)

        pd.DataFrame(rows).to_csv(self.FMT_VALMODEL_EVALTHHIST.format(prefix), index=False)
        self.logger.info("Evaluate fscore on validation set: {} .. done".format(prefix))

    def get_model(self, type_of_data):

        model_name = "model_"+self.MODEL_NAME
        batch_size_for_training = int(self.plugin_config[model_name]["train"]["batch_size"])
        training_epochs = int(self.plugin_config[model_name]["train"]["epochs"])
        batch_size_for_validating = int(self.plugin_config[model_name]["validate"]["batch_size"])
        validating_epochs = int(self.plugin_config[model_name]["validate"]["epochs"])
        # batch_size_for_net = int(self.plugin_config[model_name]["train"]["batch_size"])
        additianl_channals = 0
        if self.MODEL_NAME == "v16":
            additianl_channals = 4
        generator_train = GISDataProvider(self.plugin_config_dir, self,
                                          additianl_channals=additianl_channals,
                                          type=type_of_data, train=True)
        generator_test = GISDataProvider(self.plugin_config_dir, self,
                                         additianl_channals=additianl_channals,
                                         type=type_of_data, train=False)

        net = unet.Unet(channels=generator_train.channels, n_class=generator_train.classes, layers=3,
                        features_root=16)
        trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2),
                               batch_size=batch_size_for_training,
                               verification_batch_size=1)

        queue_loader_train = QueueLoader(self.plugin_config_dir, self, type=type_of_data,
                                         batch_size=batch_size_for_training, additianl_channals=additianl_channals,
                                         num_epochs=training_epochs, train=True)
        queue_loader_validate = QueueLoader(self.plugin_config_dir, self, type=type_of_data,
                                            batch_size=batch_size_for_validating,
                                            additianl_channals=additianl_channals, num_epochs=validating_epochs, train=False)
        place_holder = namedtuple('modelTrain', 'train_dataset test_dataset loader_train loader_test')
        operators = place_holder(train_dataset=generator_train, test_dataset=generator_test,
                                 loader_train=queue_loader_train, loader_test=queue_loader_validate)
        return net, trainer, operators



    def execute(self):
        self.logger.addHandler(self.handler)
        if not Path(self.ROOT_DIR).exists():
            Path(self.ROOT_DIR).mkdir(parents=True)
        if not Path(self.MODEL_DIR).exists():
            Path(self.MODEL_DIR).mkdir(parents=True)
        if not Path(self.IMAGE_DIR).exists():
            Path(self.IMAGE_DIR).mkdir(parents=True)


# plugin_config = "/home/geesara/project/satelliteimageclassifier/config/config.json"
# dataprocessor = DataProcessor(plugin_config)
# dataprocessor.execute()

# dataprocessor.preproc_train("/data/train/AOI_5_Khartoum_Train")
# writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_train)
# dataprocessor._get_valtrain_mul_data(5, writer_multi)
# writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_test)
# dataprocessor._get_valtest_mul_data(5, writer_multi)
