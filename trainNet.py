
from logging import getLogger, Formatter, StreamHandler, INFO, FileHandler
from pathlib import Path
import subprocess
import argparse
import math
import glob
import sys
import json
import re
import warnings

import scipy
import tqdm
import click
import tables as tb
import pandas as pd
import numpy as np
# from keras.models import Model
# from keras.engine.topology import merge as merge_l
# from keras.layers import (
#     Input, Convolution2D, MaxPooling2D, UpSampling2D,
#     Reshape, core, Dropout,
#     Activation, BatchNormalization)
# from keras.optimizers import Adam
# from keras.callbacks import ModelCheckpoint, EarlyStopping, History
# from keras import backend as K
import skimage.transform
import skimage.morphology
import rasterio.features
import shapely.wkt
import shapely.ops
import shapely.geometry



class ModleTrain():

    def __init__(self, plugin_config):
        self.MODEL_NAME = 'v9s'
        self.ORIGINAL_SIZE = 650
        self.INPUT_SIZE = 256

        self.LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'
        self.BASE_DIR = "/data/train"
        self.WORKING_DIR = "/data/working"
        self.IMAGE_DIR = "/data/working/images/{}".format('v5')
        self.MODEL_DIR = "/data/working/models/{}".format(self.MODEL_NAME)
        self.FN_SOLUTION_CSV = "/data/output/{}.csv".format(self.MODEL_NAME)

        # Parameters
        self.MIN_POLYGON_AREA = 30

        # Input files
        self.FMT_TRAIN_SUMMARY_PATH = str(
            Path(self.BASE_DIR) /
            Path("{prefix:s}_Train/") /
            Path("summaryData/{prefix:s}_Train_Building_Solutions.csv"))
        self.FMT_TRAIN_RGB_IMAGE_PATH = str(
            Path(self.BASE_DIR) /
            Path("{prefix:s}_Train/") /
            Path("RGB-PanSharpen/RGB-PanSharpen_{image_id:s}.tif"))
        self.FMT_TEST_RGB_IMAGE_PATH = str(
            Path(self.BASE_DIR) /
            Path("{prefix:s}_Test_public/") /
            Path("RGB-PanSharpen/RGB-PanSharpen_{image_id:s}.tif"))
        self.FMT_TRAIN_MSPEC_IMAGE_PATH = str(
            Path(self.BASE_DIR) /
            Path("{prefix:s}_Train/") /
            Path("MUL-PanSharpen/MUL-PanSharpen_{image_id:s}.tif"))
        self.FMT_TEST_MSPEC_IMAGE_PATH = str(
            Path(self.BASE_DIR) /
            Path("{prefix:s}_Test_public/") /
            Path("MUL-PanSharpen/MUL-PanSharpen_{image_id:s}.tif"))

        # Preprocessing result
        self.FMT_BANDCUT_TH_PATH = self.IMAGE_DIR + "/bandcut{}.csv"
        self.FMT_MUL_BANDCUT_TH_PATH = self.IMAGE_DIR + "/mul_bandcut{}.csv"

        # Image list, Image container and mask container
        self.FMT_VALTRAIN_IMAGELIST_PATH = self.IMAGE_DIR + "/{prefix:s}_valtrain_ImageId.csv"
        self.FMT_VALTEST_IMAGELIST_PATH = self.IMAGE_DIR + "/{prefix:s}_valtest_ImageId.csv"
        self.FMT_VALTRAIN_IM_STORE = self.IMAGE_DIR + "/valtrain_{}_im.h5"
        self.FMT_VALTEST_IM_STORE = self.IMAGE_DIR + "/valtest_{}_im.h5"
        self.FMT_VALTRAIN_MASK_STORE = self.IMAGE_DIR + "/valtrain_{}_mask.h5"
        self.FMT_VALTEST_MASK_STORE = self.IMAGE_DIR + "/valtest_{}_mask.h5"
        self.FMT_VALTRAIN_MUL_STORE = self.IMAGE_DIR + "/valtrain_{}_mul.h5"
        self.FMT_VALTEST_MUL_STORE = self.IMAGE_DIR + "/valtest_{}_mul.h5"

        self.FMT_TRAIN_IMAGELIST_PATH = self.IMAGE_DIR + "/{prefix:s}_train_ImageId.csv"
        self.FMT_TEST_IMAGELIST_PATH = self.IMAGE_DIR + "/{prefix:s}_test_ImageId.csv"
        self.FMT_TRAIN_IM_STORE = self.IMAGE_DIR + "/train_{}_im.h5"
        self.FMT_TEST_IM_STORE = self.IMAGE_DIR + "/test_{}_im.h5"
        self.FMT_TRAIN_MASK_STORE = self.IMAGE_DIR + "/train_{}_mask.h5"
        self.FMT_TRAIN_MUL_STORE = self.IMAGE_DIR + "/train_{}_mul.h5"
        self.FMT_TEST_MUL_STORE = self.IMAGE_DIR + "/test_{}_mul.h5"

        self.FMT_IMMEAN = self.IMAGE_DIR + "/{}_immean.h5"
        self.FMT_MULMEAN = self.IMAGE_DIR + "/{}_mulmean.h5"

        # Model files
        self.FMT_VALMODEL_PATH = self.MODEL_DIR + "/{}_val_weights.h5"
        self.FMT_FULLMODEL_PATH = self.MODEL_DIR + "/{}_full_weights.h5"
        self.FMT_VALMODEL_HIST = self.MODEL_DIR + "/{}_val_hist.csv"
        self.FMT_VALMODEL_EVALHIST = self.MODEL_DIR + "/{}_val_evalhist.csv"
        self.FMT_VALMODEL_EVALTHHIST = self.MODEL_DIR + "/{}_val_evalhist_th.csv"

        # Prediction & polygon result
        self.FMT_TESTPRED_PATH = self.MODEL_DIR + "/{}_pred.h5"
        self.FMT_VALTESTPRED_PATH = self.MODEL_DIR + "/{}_eval_pred.h5"
        self.FMT_VALTESTPOLY_PATH = self.MODEL_DIR + "/{}_eval_poly.csv"
        self.FMT_VALTESTTRUTH_PATH =self.MODEL_DIR + "/{}_eval_poly_truth.csv"
        self.FMT_VALTESTPOLY_OVALL_PATH = self.MODEL_DIR + "/eval_poly.csv"
        self.FMT_VALTESTTRUTH_OVALL_PATH = self.MODEL_DIR + "/eval_poly_truth.csv"
        self.FMT_TESTPOLY_PATH = self.MODEL_DIR + "/{}_poly.csv"

        # Model related files (others)
        self.FMT_VALMODEL_LAST_PATH = self.MODEL_DIR + "/{}_val_weights_last.h5"
        self.FMT_FULLMODEL_LAST_PATH = self.MODEL_DIR + "/{}_full_weights_last.h5"

        # Logger
        warnings.simplefilter("ignore", UserWarning)
        self.handler = StreamHandler()
        self.handler.setLevel(INFO)
        self.handler.setFormatter(Formatter(self.LOGFORMAT))

        self.fh_handler = FileHandler(".{}.log".format(self.MODEL_NAME))
        self.fh_handler.setFormatter(Formatter(self.LOGFORMAT))
        self.logger = getLogger('spacenet2')
        self.logger.setLevel(INFO)
        self.logger.addHandler(self.handler)
        self.logger.addHandler(self.fh_handler)

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


    def _remove_interiors(self, line):
        if "), (" in line:
            line_prefix = line.split('), (')[0]
            line_terminate = line.split('))",')[-1]
            line = (
                line_prefix +
                '))",' +
                line_terminate
            )
        return line


    def __load_band_cut_th(self,  band_fn, bandsz=3):
        df = pd.read_csv(band_fn, index_col='area_id')
        all_band_cut_th = {area_id: {} for area_id in range(2, 6)}
        for area_id, row in df.iterrows():
            for chan_i in range(bandsz):
                all_band_cut_th[area_id][chan_i] = dict(
                    min=row['chan{}_min'.format(chan_i)],
                    max=row['chan{}_max'.format(chan_i)],
                )
        return all_band_cut_th


    def _calc_fscore_per_aoi(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        truth_file =  self.FMT_VALTESTTRUTH_PATH.format(prefix)
        poly_file =  self.FMT_VALTESTPOLY_PATH.format(prefix)

        cmd = [
            'java',
            '-jar',
            '/root/visualizer-2.0/visualizer.jar',
            '-truth',
            truth_file,
            '-solution',
            poly_file,
            '-no-gui',
            '-band-triplets',
            '/root/visualizer-2.0/data/band-triplets.txt',
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

    def _get_model_parameter(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        fn_hist =  self.FMT_VALMODEL_EVALTHHIST.format(prefix)
        best_row = pd.read_csv(fn_hist).sort_values(
            by='fscore',
            ascending=False,
        ).iloc[0]

        param = dict(
            fn_epoch=int(best_row['zero_base_epoch']),
            min_poly_area=int(best_row['min_area_th']),
        )
        return param


    def get_resized_raster_3chan_image(self, image_id, band_cut_th=None):
        fn = self.train_image_id_to_path(image_id)
        with rasterio.open(fn, 'r') as f:
            values = f.read().astype(np.float32)
            for chan_i in range(3):
                min_val = band_cut_th[chan_i]['min']
                max_val = band_cut_th[chan_i]['max']
                values[chan_i] = np.clip(values[chan_i], min_val, max_val)
                values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
        values = np.swapaxes(values, 0, 2)
        values = np.swapaxes(values, 0, 1)
        values = skimage.transform.resize(values, ( self.INPUT_SIZE,  self.INPUT_SIZE))
        return values


    def get_resized_raster_3chan_image_test(self, image_id, band_cut_th=None):
        fn = self.test_image_id_to_path(image_id)
        with rasterio.open(fn, 'r') as f:
            values = f.read().astype(np.float32)
            for chan_i in range(3):
                min_val = band_cut_th[chan_i]['min']
                max_val = band_cut_th[chan_i]['max']
                values[chan_i] = np.clip(values[chan_i], min_val, max_val)
                values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
        values = np.swapaxes(values, 0, 2)
        values = np.swapaxes(values, 0, 1)
        values = skimage.transform.resize(values, ( self.INPUT_SIZE,  self.INPUT_SIZE))
        return values


    def image_mask_resized_from_summary(self, df, image_id):
        im_mask = np.zeros((650, 650))
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
        im_mask = skimage.transform.resize(im_mask, ( self.INPUT_SIZE,  self.INPUT_SIZE))
        im_mask = (im_mask > 0.5).astype(np.uint8)
        return im_mask


    def train_test_image_prep(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        df_train = pd.read_csv(
             self.FMT_TRAIN_IMAGELIST_PATH.format(prefix=prefix),
            index_col='ImageId')
        df_test = pd.read_csv(
             self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix),
            index_col='ImageId')
        band_cut_th = self.__load_band_cut_th(
             self.FMT_BANDCUT_TH_PATH.format(prefix))[area_id]
        df_summary = self._load_train_summary_data(area_id)

        fn =  self.FMT_TRAIN_IM_STORE.format(prefix)
        self.logger.info("Prepare image container: {}".format(fn))
        with tb.open_file(fn, 'w') as f:
            for image_id in tqdm.tqdm(df_train.index, total=len(df_train)):
                im = self.get_resized_raster_3chan_image(image_id, band_cut_th)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im.shape, filters=filters)
                ds[:] = im

        fn =  self.FMT_TEST_IM_STORE.format(prefix)
        self.logger.info("Prepare image container: {}".format(fn))
        with tb.open_file(fn, 'w') as f:
            for image_id in tqdm.tqdm(df_test.index, total=len(df_test)):
                im = self.get_resized_raster_3chan_image_test(image_id, band_cut_th)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im.shape, filters=filters)
                ds[:] = im

        fn =  self.FMT_TRAIN_MASK_STORE.format(prefix)
        self.logger.info("Prepare image container: {}".format(fn))
        with tb.open_file(fn, 'w') as f:
            for image_id in tqdm.tqdm(df_train.index, total=len(df_train)):
                im_mask = self.image_mask_resized_from_summary(df_summary, image_id)
                atom = tb.Atom.from_dtype(im_mask.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im_mask.shape, filters=filters)
                ds[:] = im_mask


    def valtrain_test_image_prep(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("valtrain_test_image_prep for {}".format(prefix))

        df_train = pd.read_csv(
             self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix),
            index_col='ImageId')
        df_test = pd.read_csv(
             self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix),
            index_col='ImageId')
        band_cut_th = self.__load_band_cut_th(
             self.FMT_BANDCUT_TH_PATH.format(prefix))[area_id]
        df_summary = self._load_train_summary_data(area_id)

        fn =  self.FMT_VALTRAIN_IM_STORE.format(prefix)
        self.logger.info("Prepare image container: {}".format(fn))
        with tb.open_file(fn, 'w') as f:
            for image_id in tqdm.tqdm(df_train.index, total=len(df_train)):
                im = self.get_resized_raster_3chan_image(image_id, band_cut_th)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im.shape,
                                     filters=filters)
                ds[:] = im

        fn =  self.FMT_VALTEST_IM_STORE.format(prefix)
        self.logger.info("Prepare image container: {}".format(fn))
        with tb.open_file(fn, 'w') as f:
            for image_id in tqdm.tqdm(df_test.index, total=len(df_test)):
                im = self.get_resized_raster_3chan_image(image_id, band_cut_th)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im.shape,
                                     filters=filters)
                ds[:] = im

        fn =  self.FMT_VALTRAIN_MASK_STORE.format(prefix)
        self.logger.info("Prepare image container: {}".format(fn))
        with tb.open_file(fn, 'w') as f:
            for image_id in tqdm.tqdm(df_train.index, total=len(df_train)):
                im_mask = self.image_mask_resized_from_summary(df_summary, image_id)
                atom = tb.Atom.from_dtype(im_mask.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im_mask.shape,
                                     filters=filters)
                ds[:] = im_mask

        fn =  self.FMT_VALTEST_MASK_STORE.format(prefix)
        self.logger.info("Prepare image container: {}".format(fn))
        with tb.open_file(fn, 'w') as f:
            for image_id in tqdm.tqdm(df_test.index, total=len(df_test)):
                im_mask = self.image_mask_resized_from_summary(df_summary, image_id)
                atom = tb.Atom.from_dtype(im_mask.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im_mask.shape,
                                     filters=filters)
                ds[:] = im_mask


    def train_test_mul_image_prep(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        df_train = pd.read_csv(
             self.FMT_TRAIN_IMAGELIST_PATH.format(prefix=prefix),
            index_col='ImageId')
        df_test = pd.read_csv(
             self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix),
            index_col='ImageId')
        band_rgb_th = self.__load_band_cut_th(
             self.FMT_BANDCUT_TH_PATH.format(prefix))[area_id]
        band_mul_th = self.__load_band_cut_th(
             self.FMT_MUL_BANDCUT_TH_PATH.format(prefix), bandsz=8)[area_id]
        df_summary = self._load_train_summary_data(area_id)

        fn =  self.FMT_TRAIN_MUL_STORE.format(prefix)
        self.logger.info("Prepare image container: {}".format(fn))
        with tb.open_file(fn, 'w') as f:
            for image_id in tqdm.tqdm(df_train.index, total=len(df_train)):
                im = self.get_resized_raster_8chan_image(
                    image_id, band_rgb_th, band_mul_th)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im.shape,
                                     filters=filters)
                ds[:] = im

        fn =  self.FMT_TEST_MUL_STORE.format(prefix)
        self.logger.info("Prepare image container: {}".format(fn))
        with tb.open_file(fn, 'w') as f:
            for image_id in tqdm.tqdm(df_test.index, total=len(df_test)):
                im = self.get_resized_raster_8chan_image_test(
                    image_id, band_rgb_th, band_mul_th)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im.shape,
                                     filters=filters)
                ds[:] = im


    def valtrain_test_mul_image_prep(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("valtrain_test_image_prep for {}".format(prefix))

        df_train = pd.read_csv(
             self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix),
            index_col='ImageId')
        df_test = pd.read_csv(
             self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix),
            index_col='ImageId')
        band_rgb_th = self.__load_band_cut_th(
             self.FMT_BANDCUT_TH_PATH.format(prefix))[area_id]
        band_mul_th = self.__load_band_cut_th(
             self.FMT_MUL_BANDCUT_TH_PATH.format(prefix), bandsz=8)[area_id]
        df_summary = self._load_train_summary_data(area_id)

        fn =  self.FMT_VALTRAIN_MUL_STORE.format(prefix)
        self.logger.info("Prepare image container: {}".format(fn))
        with tb.open_file(fn, 'w') as f:
            for image_id in tqdm.tqdm(df_train.index, total=len(df_train)):
                im = self.get_resized_raster_8chan_image(
                    image_id, band_rgb_th, band_mul_th)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im.shape,
                                     filters=filters)
                ds[:] = im

        fn =  self.FMT_VALTEST_MUL_STORE.format(prefix)
        self.logger.info("Prepare image container: {}".format(fn))
        with tb.open_file(fn, 'w') as f:
            for image_id in tqdm.tqdm(df_test.index, total=len(df_test)):
                im = self.get_resized_raster_8chan_image(
                    image_id, band_rgb_th, band_mul_th)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im.shape,
                                     filters=filters)
                ds[:] = im


    def _load_train_summary_data(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        fn =  self.FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
        df = pd.read_csv(fn)
        return df


    def split_val_train_test(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        df = self._load_train_summary_data(area_id)
        df_agg = df.groupby('ImageId').agg('first')
        image_id_list = df_agg.index.tolist()
        np.random.shuffle(image_id_list)
        sz_valtrain = int(len(image_id_list) * 0.7)
        sz_valtest = len(image_id_list) - sz_valtrain

        pd.DataFrame({'ImageId': image_id_list[:sz_valtrain]}).to_csv(
             self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix),
            index=False)
        pd.DataFrame({'ImageId': image_id_list[sz_valtrain:]}).to_csv(
             self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix),
            index=False)


    def train_image_id_to_mspec_path(self, image_id):
        prefix = self.image_id_to_prefix(image_id)
        fn =  self.FMT_TRAIN_MSPEC_IMAGE_PATH.format(
            prefix=prefix,
            image_id=image_id)
        return fn


    def test_image_id_to_mspec_path(self, image_id):
        prefix = self.image_id_to_prefix(image_id)
        fn =  self.FMT_TEST_MSPEC_IMAGE_PATH.format(
            prefix=prefix,
            image_id=image_id)
        return fn


    def train_image_id_to_path(self, image_id):
        prefix = self.image_id_to_prefix(image_id)
        fn =  self.FMT_TRAIN_RGB_IMAGE_PATH.format(
            prefix=prefix,
            image_id=image_id)
        return fn


    def test_image_id_to_path(self, image_id):
        prefix = self.image_id_to_prefix(image_id)
        fn =  self.FMT_TEST_RGB_IMAGE_PATH.format(
            prefix=prefix,
            image_id=image_id)
        return fn


    def image_id_to_prefix(self, image_id):
        prefix = image_id.split('img')[0][:-1]
        return prefix


    def calc_multiband_cut_threshold(self, area_id):
        rows = []
        band_cut_th = self.__calc_multiband_cut_threshold(area_id)
        prefix = self.area_id_to_prefix(area_id)
        row = dict(prefix=self.area_id_to_prefix(area_id))
        row['area_id'] = area_id
        for chan_i in band_cut_th.keys():
            row['chan{}_max'.format(chan_i)] = band_cut_th[chan_i]['max']
            row['chan{}_min'.format(chan_i)] = band_cut_th[chan_i]['min']
        rows.append(row)
        pd.DataFrame(rows).to_csv( self.FMT_BANDCUT_TH_PATH.format(prefix), index=False)


    def __calc_multiband_cut_threshold(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        band_values = {k: [] for k in range(3)}
        band_cut_th = {k: dict(max=0, min=0) for k in range(3)}

        image_id_list = pd.read_csv( self.FMT_VALTRAIN_IMAGELIST_PATH.format(
            prefix=prefix)).ImageId.tolist()
        for image_id in tqdm.tqdm(image_id_list[:500]):
            image_fn = self.train_image_id_to_path(image_id)
            with rasterio.open(image_fn, 'r') as f:
                values = f.read().astype(np.float32)
                for i_chan in range(3):
                    values_ = values[i_chan].ravel().tolist()
                    values_ = np.array(
                        [v for v in values_ if v != 0]
                    )  # Remove sensored mask
                    band_values[i_chan].append(values_)

        image_id_list = pd.read_csv( self.FMT_VALTEST_IMAGELIST_PATH.format(
            prefix=prefix)).ImageId.tolist()
        for image_id in tqdm.tqdm(image_id_list[:500]):
            image_fn = self.train_image_id_to_path(image_id)
            with rasterio.open(image_fn, 'r') as f:
                values = f.read().astype(np.float32)
                for i_chan in range(3):
                    values_ = values[i_chan].ravel().tolist()
                    values_ = np.array(
                        [v for v in values_ if v != 0]
                    )  # Remove sensored mask
                    band_values[i_chan].append(values_)

        for i_chan in range(3):
            band_values[i_chan] = np.concatenate(
                band_values[i_chan]).ravel()
            band_cut_th[i_chan]['max'] = scipy.percentile(
                band_values[i_chan], 98)
            band_cut_th[i_chan]['min'] = scipy.percentile(
                band_values[i_chan], 2)
        return band_cut_th


    def calc_mul_multiband_cut_threshold(self, area_id):
        rows = []
        band_cut_th = self.__calc_mul_multiband_cut_threshold(area_id)
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


    def __calc_mul_multiband_cut_threshold(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        band_values = {k: [] for k in range(8)}
        band_cut_th = {k: dict(max=0, min=0) for k in range(8)}

        image_id_list = pd.read_csv( self.FMT_VALTRAIN_IMAGELIST_PATH.format(
            prefix=prefix)).ImageId.tolist()
        for image_id in tqdm.tqdm(image_id_list[:500]):
            image_fn = self.train_image_id_to_mspec_path(image_id)
            with rasterio.open(image_fn, 'r') as f:
                values = f.read().astype(np.float32)
                for i_chan in range(8):
                    values_ = values[i_chan].ravel().tolist()
                    values_ = np.array(
                        [v for v in values_ if v != 0]
                    )  # Remove sensored mask
                    band_values[i_chan].append(values_)

        image_id_list = pd.read_csv( self.FMT_VALTEST_IMAGELIST_PATH.format(
            prefix=prefix)).ImageId.tolist()
        for image_id in tqdm.tqdm(image_id_list[:500]):
            image_fn = self.train_image_id_to_mspec_path(image_id)
            with rasterio.open(image_fn, 'r') as f:
                values = f.read().astype(np.float32)
                for i_chan in range(8):
                    values_ = values[i_chan].ravel().tolist()
                    values_ = np.array(
                        [v for v in values_ if v != 0]
                    )  # Remove sensored mask
                    band_values[i_chan].append(values_)

        for i_chan in range(8):
            band_values[i_chan] = np.concatenate(
                band_values[i_chan]).ravel()
            band_cut_th[i_chan]['max'] = scipy.percentile(
                band_values[i_chan], 98)
            band_cut_th[i_chan]['min'] = scipy.percentile(
                band_values[i_chan], 2)
        return band_cut_th


    def get_unet(self):
        print ("kkkk")
        # conv_params = dict(activation='relu', border_mode='same')
        # merge_params = dict(mode='concat', concat_axis=1)
        # inputs = Input((8, 256, 256))
        # conv1 = Convolution2D(32, 3, 3, **conv_params)(inputs)
        # conv1 = Convolution2D(32, 3, 3, **conv_params)(conv1)
        # pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        #
        # conv2 = Convolution2D(64, 3, 3, **conv_params)(pool1)
        # conv2 = Convolution2D(64, 3, 3, **conv_params)(conv2)
        # pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        #
        # conv3 = Convolution2D(128, 3, 3, **conv_params)(pool2)
        # conv3 = Convolution2D(128, 3, 3, **conv_params)(conv3)
        # pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        #
        # conv4 = Convolution2D(256, 3, 3, **conv_params)(pool3)
        # conv4 = Convolution2D(256, 3, 3, **conv_params)(conv4)
        # pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
        #
        # conv5 = Convolution2D(512, 3, 3, **conv_params)(pool4)
        # conv5 = Convolution2D(512, 3, 3, **conv_params)(conv5)
        #
        # up6 = merge_l([UpSampling2D(size=(2, 2))(conv5), conv4], **merge_params)
        # conv6 = Convolution2D(256, 3, 3, **conv_params)(up6)
        # conv6 = Convolution2D(256, 3, 3, **conv_params)(conv6)
        #
        # up7 = merge_l([UpSampling2D(size=(2, 2))(conv6), conv3], **merge_params)
        # conv7 = Convolution2D(128, 3, 3, **conv_params)(up7)
        # conv7 = Convolution2D(128, 3, 3, **conv_params)(conv7)
        #
        # up8 = merge_l([UpSampling2D(size=(2, 2))(conv7), conv2], **merge_params)
        # conv8 = Convolution2D(64, 3, 3, **conv_params)(up8)
        # conv8 = Convolution2D(64, 3, 3, **conv_params)(conv8)
        #
        # up9 = merge_l([UpSampling2D(size=(2, 2))(conv8), conv1], **merge_params)
        # conv9 = Convolution2D(32, 3, 3, **conv_params)(up9)
        # conv9 = Convolution2D(32, 3, 3, **conv_params)(conv9)
        #
        # conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)
        # adam = Adam()
        #
        # model = Model(input=inputs, output=conv10)
        # model.compile(optimizer=adam,
        #               loss='binary_crossentropy',
        #               metrics=['accuracy', jaccard_coef, jaccard_coef_int])
        # return model


    def jaccard_coef(self, y_true, y_pred):
        print ("-")
        # smooth = 1e-12
        # intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
        # sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])
        # jac = (intersection + smooth) / (sum_ - intersection + smooth)
        # return K.mean(jac)


    def jaccard_coef_int(self, y_true, y_pred):
        print ("-")
        # smooth = 1e-12
        # y_pred_pos = K.round(K.clip(y_pred, 0, 1))
        # intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
        # sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])
        # jac = (intersection + smooth) / (sum_ - intersection + smooth)
        # return K.mean(jac)


    def generate_test_batch(self, area_id,
                            batch_size=64,
                            immean=None,
                            enable_tqdm=False):
        prefix = self.area_id_to_prefix(area_id)
        df_test = pd.read_csv( self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix))
        fn_im =  self.FMT_TEST_MUL_STORE.format(prefix)

        image_id_list = df_test.ImageId.tolist()

        if enable_tqdm:
            pbar = tqdm.tqdm(total=len(image_id_list))

        while 1:
            total_sz = len(image_id_list)
            n_batch = int(math.floor(total_sz / batch_size) + 1)
            with tb.open_file(fn_im, 'r') as f_im:
                for i_batch in range(n_batch):
                    target_image_ids = image_id_list[
                        i_batch*batch_size:(i_batch+1)*batch_size
                    ]
                    if len(target_image_ids) == 0:
                        continue

                    X_test = []
                    y_test = []
                    for image_id in target_image_ids:
                        im = np.array(f_im.get_node('/' + image_id))
                        im = np.swapaxes(im, 0, 2)
                        im = np.swapaxes(im, 1, 2)

                        X_test.append(im)
                        mask = np.zeros(( self.INPUT_SIZE,  self.INPUT_SIZE)).astype(np.uint8)
                        y_test.append(mask)
                    X_test = np.array(X_test)
                    y_test = np.array(y_test)
                    y_test = y_test.reshape((-1, 1,  self.INPUT_SIZE,  self.INPUT_SIZE))

                    if immean is not None:
                        X_test = X_test - immean

                    if enable_tqdm:
                        pbar.update(y_test.shape[0])

                    yield (X_test, y_test)

            if enable_tqdm:
                pbar.close()


    def get_resized_raster_8chan_image_test(self, image_id, band_rgb_th, band_mul_th):
        """
        RGB + multispectral (total: 8 channels)
        """
        im = []

        fn = self.test_image_id_to_path(image_id)
        with rasterio.open(fn, 'r') as f:
            values = f.read().astype(np.float32)
            for chan_i in range(3):
                min_val = band_rgb_th[chan_i]['min']
                max_val = band_rgb_th[chan_i]['max']
                values[chan_i] = np.clip(values[chan_i], min_val, max_val)
                values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
                im.append(skimage.transform.resize(
                    values[chan_i],
                    ( self.INPUT_SIZE,  self.INPUT_SIZE)))

        fn = self.test_image_id_to_mspec_path(image_id)
        with rasterio.open(fn, 'r') as f:
            values = f.read().astype(np.float32)
            usechannels = [1, 2, 5, 6, 7]
            for chan_i in usechannels:
                min_val = band_mul_th[chan_i]['min']
                max_val = band_mul_th[chan_i]['max']
                values[chan_i] = np.clip(values[chan_i], min_val, max_val)
                values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
                im.append(skimage.transform.resize(
                    values[chan_i],
                    ( self.INPUT_SIZE,  self.INPUT_SIZE)))

        im = np.array(im)  # (ch, w, h)
        im = np.swapaxes(im, 0, 2)  # -> (h, w, ch)
        im = np.swapaxes(im, 0, 1)  # -> (w, h, ch)
        return im


    def get_resized_raster_8chan_image(self, image_id, band_rgb_th, band_mul_th):
        """
        RGB + multispectral (total: 8 channels)
        """
        im = []

        fn = self.train_image_id_to_path(image_id)
        with rasterio.open(fn, 'r') as f:
            values = f.read().astype(np.float32)
            for chan_i in range(3):
                min_val = band_rgb_th[chan_i]['min']
                max_val = band_rgb_th[chan_i]['max']
                values[chan_i] = np.clip(values[chan_i], min_val, max_val)
                values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
                im.append(skimage.transform.resize(
                    values[chan_i],
                    ( self.INPUT_SIZE,  self.INPUT_SIZE)))

        fn = self.train_image_id_to_mspec_path(image_id)
        with rasterio.open(fn, 'r') as f:
            values = f.read().astype(np.float32)
            usechannels = [1, 2, 5, 6, 7]
            for chan_i in usechannels:
                min_val = band_mul_th[chan_i]['min']
                max_val = band_mul_th[chan_i]['max']
                values[chan_i] = np.clip(values[chan_i], min_val, max_val)
                values[chan_i] = (values[chan_i] - min_val) / (max_val - min_val)
                im.append(skimage.transform.resize(
                    values[chan_i],
                    ( self.INPUT_SIZE,  self.INPUT_SIZE)))

        im = np.array(im)  # (ch, w, h)
        im = np.swapaxes(im, 0, 2)  # -> (h, w, ch)
        im = np.swapaxes(im, 0, 1)  # -> (w, h, ch)
        return im


    def _get_train_mul_data(self, area_id):
        """
        RGB + multispectral (total: 8 channels)
        """
        prefix = self.area_id_to_prefix(area_id)
        fn_train =  self.FMT_TRAIN_IMAGELIST_PATH.format(prefix=prefix)
        df_train = pd.read_csv(fn_train)

        X_train = []
        fn_im =  self.FMT_TRAIN_MUL_STORE.format(prefix)
        with tb.open_file(fn_im, 'r') as f:
            for idx, image_id in enumerate(df_train.ImageId.tolist()):
                im = np.array(f.get_node('/' + image_id))
                im = np.swapaxes(im, 0, 2)
                im = np.swapaxes(im, 1, 2)
                X_train.append(im)
        X_train = np.array(X_train)

        y_train = []
        fn_mask =  self.FMT_TRAIN_MASK_STORE.format(prefix)
        with tb.open_file(fn_mask, 'r') as f:
            for idx, image_id in enumerate(df_train.ImageId.tolist()):
                mask = np.array(f.get_node('/' + image_id))
                mask = (mask > 0.5).astype(np.uint8)
                y_train.append(mask)
        y_train = np.array(y_train)
        y_train = y_train.reshape((-1, 1,  self.INPUT_SIZE,  self.INPUT_SIZE))

        return X_train, y_train


    def _get_test_mul_data(self, area_id):
        """
        RGB + multispectral (total: 8 channels)
        """
        prefix = self.area_id_to_prefix(area_id)
        fn_test =  self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test)

        X_test = []
        fn_im =  self.FMT_TEST_MUL_STORE.format(prefix)
        with tb.open_file(fn_im, 'r') as f:
            for idx, image_id in enumerate(df_test.ImageId.tolist()):
                im = np.array(f.get_node('/' + image_id))
                im = np.swapaxes(im, 0, 2)
                im = np.swapaxes(im, 1, 2)
                X_test.append(im)
        X_test = np.array(X_test)

        return X_test


    def _get_valtest_mul_data(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        fn_test =  self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test)

        X_val = []
        fn_im =  self.FMT_VALTEST_MUL_STORE.format(prefix)
        with tb.open_file(fn_im, 'r') as f:
            for idx, image_id in enumerate(df_test.ImageId.tolist()):
                im = np.array(f.get_node('/' + image_id))
                im = np.swapaxes(im, 0, 2)
                im = np.swapaxes(im, 1, 2)
                X_val.append(im)
        X_val = np.array(X_val)

        y_val = []
        fn_mask =  self.FMT_VALTEST_MASK_STORE.format(prefix)
        with tb.open_file(fn_mask, 'r') as f:
            for idx, image_id in enumerate(df_test.ImageId.tolist()):
                mask = np.array(f.get_node('/' + image_id))
                mask = (mask > 0.5).astype(np.uint8)
                y_val.append(mask)
        y_val = np.array(y_val)
        y_val = y_val.reshape((-1, 1,  self.INPUT_SIZE,  self.INPUT_SIZE))

        return X_val, y_val


    def _get_valtrain_mul_data(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        fn_train =  self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
        df_train = pd.read_csv(fn_train)

        X_val = []
        fn_im =  self.FMT_VALTRAIN_MUL_STORE.format(prefix)
        with tb.open_file(fn_im, 'r') as f:
            for idx, image_id in enumerate(df_train.ImageId.tolist()):
                im = np.array(f.get_node('/' + image_id))
                im = np.swapaxes(im, 0, 2)
                im = np.swapaxes(im, 1, 2)
                X_val.append(im)
        X_val = np.array(X_val)

        y_val = []
        fn_mask =  self.FMT_VALTRAIN_MASK_STORE.format(prefix)
        with tb.open_file(fn_mask, 'r') as f:
            for idx, image_id in enumerate(df_train.ImageId.tolist()):
                mask = np.array(f.get_node('/' + image_id))
                mask = (mask > 0.5).astype(np.uint8)
                y_val.append(mask)
        y_val = np.array(y_val)
        y_val = y_val.reshape((-1, 1,  self.INPUT_SIZE,  self.INPUT_SIZE))

        return X_val, y_val


    def get_mul_mean_image(self, area_id):
        prefix = self.area_id_to_prefix(area_id)

        with tb.open_file( self.FMT_MULMEAN.format(prefix), 'r') as f:
            im_mean = np.array(f.get_node('/mulmean'))
        return im_mean


    def preproc_stage3(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        if not Path( self.FMT_VALTEST_MUL_STORE.format(prefix)).exists():
            self.valtrain_test_mul_image_prep(area_id)
        if not Path( self.FMT_TEST_MUL_STORE.format(prefix)).exists():
            self.train_test_mul_image_prep(area_id)

        # mean image for subtract preprocessing
        X1, _ = self._get_train_mul_data(area_id)
        X2 = self._get_test_mul_data(area_id)
        X = np.vstack([X1, X2])
        print(X.shape)
        X_mean = X.mean(axis=0)

        fn =  self.FMT_MULMEAN.format(prefix)
        self.logger.info("Prepare mean image: {}".format(fn))
        with tb.open_file(fn, 'w') as f:
            atom = tb.Atom.from_dtype(X_mean.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root, 'mulmean', atom, X_mean.shape,
                                 filters=filters)
            ds[:] = X_mean


    def _internal_test_predict_best_param(self, area_id,
                                          save_pred=True):
        prefix = self.area_id_to_prefix(area_id)
        param = self._get_model_parameter(area_id)
        epoch = param['fn_epoch']
        min_th = param['min_poly_area']

        # Prediction phase
        self.logger.info("Prediction phase: {}".format(prefix))

        X_mean = self.get_mul_mean_image(area_id)

        # Load model weights
        # Predict and Save prediction result
        fn =  self.FMT_TESTPRED_PATH.format(prefix)
        fn_model =  self.FMT_VALMODEL_PATH.format(prefix + '_{epoch:02d}')
        fn_model = fn_model.format(epoch=epoch)
        model = self.get_unet()
        model.load_weights(fn_model)

        fn_test =  self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test, index_col='ImageId')

        y_pred = model.predict_generator(
            self.generate_test_batch(
                area_id,
                batch_size=64,
                immean=X_mean,
                enable_tqdm=True,
            ),
            val_samples=len(df_test),
        )
        del model

        # Save prediction result
        if save_pred:
            with tb.open_file(fn, 'w') as f:
                atom = tb.Atom.from_dtype(y_pred.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, 'pred', atom, y_pred.shape,
                                     filters=filters)
                ds[:] = y_pred

        return y_pred


    def _internal_test(self, area_id, enable_tqdm=False):
        prefix = self.area_id_to_prefix(area_id)
        y_pred = self._internal_test_predict_best_param(area_id, save_pred=False)

        param = self._get_model_parameter(area_id)
        min_th = param['min_poly_area']

        # Postprocessing phase
        self.logger.info("Postprocessing phase")
        fn_test =  self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test, index_col='ImageId')

        fn_out =  self.FMT_TESTPOLY_PATH.format(prefix)
        with open(fn_out, 'w') as f:
            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")

            test_image_list = df_test.index.tolist()
            for idx, image_id in tqdm.tqdm(enumerate(test_image_list),
                                           total=len(test_image_list)):
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


    def validate_score(self, area_id):
        """
        Calc competition score
        """
        prefix = self.area_id_to_prefix(area_id)

        # Prediction phase
        if not Path( self.FMT_VALTESTPRED_PATH.format(prefix)).exists():
            X_val, y_val = self._get_valtest_mul_data(area_id)
            X_mean = self.get_mul_mean_image(area_id)

            # Load model weights
            # Predict and Save prediction result
            model = self.get_unet()
            model.load_weights( self.FMT_VALMODEL_PATH.format(prefix))
            y_pred = model.predict(X_val - X_mean, batch_size=8, verbose=1)
            del model

            # Save prediction result
            fn =  self.FMT_VALTESTPRED_PATH.format(prefix)
            with tb.open_file(fn, 'w') as f:
                atom = tb.Atom.from_dtype(y_pred.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, 'pred', atom, y_pred.shape,
                                     filters=filters)
                ds[:] = y_pred

        # Postprocessing phase
        if not Path( self.FMT_VALTESTPOLY_PATH.format(prefix)).exists():
            fn_test =  self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
            df_test = pd.read_csv(fn_test, index_col='ImageId')
            fn =  self.FMT_VALTESTPRED_PATH.format(prefix)
            with tb.open_file(fn, 'r') as f:
                y_pred = np.array(f.get_node('/pred'))
            print(y_pred.shape)

            fn_out =  self.FMT_VALTESTPOLY_PATH.format(prefix)
            with open(fn_out, 'w') as f:
                f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
                for idx, image_id in enumerate(df_test.index.tolist()):
                    df_poly = self.mask_to_poly(y_pred[idx][0])
                    if len(df_poly) > 0:
                        for i, row in df_poly.iterrows():
                            f.write("{},{},\"{}\",{:.6f}\n".format(
                                image_id,
                                row.bid,
                                row.wkt,
                                row.area_ratio))
                    else:
                        f.write("{},{},{},0\n".format(
                            image_id,
                            -1,
                            "POLYGON EMPTY"))

            # update fn_out
            with open(fn_out, 'r') as f:
                lines = f.readlines()
            with open(fn_out, 'w') as f:
                f.write(lines[0])
                for line in lines[1:]:
                    line = self._remove_interiors(line)
                    f.write(line)

        # Validation solution file
        if not Path( self.FMT_VALTESTTRUTH_PATH.format(prefix)).exists():
            fn_true =  self.FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
            df_true = pd.read_csv(fn_true)
            # # Remove prefix "PAN_"
            # df_true.loc[:, 'ImageId'] = df_true.ImageId.str[4:]

            fn_test =  self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
            df_test = pd.read_csv(fn_test)
            df_test_image_ids = df_test.ImageId.unique()

            fn_out =  self.FMT_VALTESTTRUTH_PATH.format(prefix)
            with open(fn_out, 'w') as f:
                f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
                df_true = df_true[df_true.ImageId.isin(df_test_image_ids)]
                for idx, r in df_true.iterrows():
                    f.write("{},{},\"{}\",{:.6f}\n".format(
                        r.ImageId,
                        r.BuildingId,
                        r.PolygonWKT_Pix,
                        1.0))


    def validate_all_score(self):
        header_line = []
        lines = []
        for area_id in range(2, 6):
            prefix = self.area_id_to_prefix(area_id)
            assert Path( self.FMT_VALTESTTRUTH_PATH.format(prefix)).exists()
            with open( self.FMT_VALTESTTRUTH_PATH.format(prefix), 'r') as f:
                header_line = f.readline()
                lines += f.readlines()

        with open( self.FMT_VALTESTTRUTH_OVALL_PATH, 'w') as f:
            f.write(header_line)
            for line in lines:
                f.write(line)

        # Predicted polygons
        header_line = []
        lines = []
        for area_id in range(2, 6):
            prefix = self.area_id_to_prefix(area_id)
            assert Path( self.FMT_VALTESTPOLY_PATH.format(prefix)).exists()
            with open( self.FMT_VALTESTPOLY_PATH.format(prefix), 'r') as f:
                header_line = f.readline()
                lines += f.readlines()

        with open( self.FMT_VALTESTPOLY_OVALL_PATH, 'w') as f:
            f.write(header_line)
            for line in lines:
                f.write(line)


    def generate_valtest_batch(self, area_id,
                               batch_size=8,
                               immean=None,
                               enable_tqdm=False):
        prefix = self.area_id_to_prefix(area_id)
        df_train = pd.read_csv( self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix))
        fn_im =  self.FMT_VALTEST_MUL_STORE.format(prefix)
        fn_mask =  self.FMT_VALTEST_MASK_STORE.format(prefix)

        image_id_list = df_train.ImageId.tolist()

        if enable_tqdm:
            pbar = tqdm.tqdm(total=len(image_id_list))

        while 1:
            total_sz = len(image_id_list)
            n_batch = int(math.floor(total_sz / batch_size) + 1)
            with tb.open_file(fn_im, 'r') as f_im,\
                    tb.open_file(fn_mask, 'r') as f_mask:
                for i_batch in range(n_batch):
                    target_image_ids = image_id_list[
                        i_batch*batch_size:(i_batch+1)*batch_size
                    ]
                    if len(target_image_ids) == 0:
                        continue

                    X_train = []
                    y_train = []
                    for image_id in target_image_ids:
                        im = np.array(f_im.get_node('/' + image_id))
                        im = np.swapaxes(im, 0, 2)
                        im = np.swapaxes(im, 1, 2)

                        X_train.append(im)
                        mask = np.array(f_mask.get_node('/' + image_id))
                        mask = (mask > 0).astype(np.uint8)
                        y_train.append(mask)
                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    y_train = y_train.reshape((-1, 1,  self.INPUT_SIZE,  self.INPUT_SIZE))

                    if immean is not None:
                        X_train = X_train - immean

                    if enable_tqdm:
                        pbar.update(y_train.shape[0])

                    yield (X_train, y_train)

            if enable_tqdm:
                pbar.close()


    def generate_valtrain_batch(self, area_id, batch_size=8, immean=None):
        prefix = self.area_id_to_prefix(area_id)
        df_train = pd.read_csv( self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix))
        fn_im =  self.FMT_VALTRAIN_MUL_STORE.format(prefix)
        fn_mask =  self.FMT_VALTRAIN_MASK_STORE.format(prefix)

        image_id_list = df_train.ImageId.tolist()
        np.random.shuffle(image_id_list)

        while 1:
            total_sz = len(image_id_list)
            n_batch = int(math.floor(total_sz / batch_size) + 1)
            with tb.open_file(fn_im, 'r') as f_im,\
                    tb.open_file(fn_mask, 'r') as f_mask:
                for i_batch in range(n_batch):
                    target_image_ids = image_id_list[
                        i_batch*batch_size:(i_batch+1)*batch_size
                    ]
                    if len(target_image_ids) == 0:
                        continue

                    X_train = []
                    y_train = []
                    for image_id in target_image_ids:
                        im = np.array(f_im.get_node('/' + image_id))
                        im = np.swapaxes(im, 0, 2)
                        im = np.swapaxes(im, 1, 2)
                        X_train.append(im)
                        mask = np.array(f_mask.get_node('/' + image_id))
                        mask = (mask > 0).astype(np.uint8)
                        y_train.append(mask)
                    X_train = np.array(X_train)
                    y_train = np.array(y_train)
                    y_train = y_train.reshape((-1, 1,  self.INPUT_SIZE,  self.INPUT_SIZE))

                    if immean is not None:
                        X_train = X_train - immean

                    yield (X_train, y_train)


    def _get_test_data(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        fn_test =  self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test)

        X_test = []
        fn_im =  self.FMT_TEST_IM_STORE.format(prefix)
        with tb.open_file(fn_im, 'r') as f:
            for idx, image_id in enumerate(df_test.ImageId.tolist()):
                im = np.array(f.get_node('/' + image_id))
                im = np.swapaxes(im, 0, 2)
                im = np.swapaxes(im, 1, 2)
                X_test.append(im)
        X_test = np.array(X_test)

        return X_test


    def _get_valtest_data(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        fn_test =  self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test)

        X_val = []
        fn_im =  self.FMT_VALTEST_IM_STORE.format(prefix)
        with tb.open_file(fn_im, 'r') as f:
            for idx, image_id in enumerate(df_test.ImageId.tolist()):
                im = np.array(f.get_node('/' + image_id))
                im = np.swapaxes(im, 0, 2)
                im = np.swapaxes(im, 1, 2)
                X_val.append(im)
        X_val = np.array(X_val)

        y_val = []
        fn_mask =  self.FMT_VALTEST_MASK_STORE.format(prefix)
        with tb.open_file(fn_mask, 'r') as f:
            for idx, image_id in enumerate(df_test.ImageId.tolist()):
                mask = np.array(f.get_node('/' + image_id))
                mask = (mask > 0.5).astype(np.uint8)
                y_val.append(mask)
        y_val = np.array(y_val)
        y_val = y_val.reshape((-1, 1,  self.INPUT_SIZE,  self.INPUT_SIZE))

        return X_val, y_val


    def _get_valtrain_data(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        fn_train =  self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
        df_train = pd.read_csv(fn_train)

        X_val = []
        fn_im =  self.FMT_VALTRAIN_IM_STORE.format(prefix)
        with tb.open_file(fn_im, 'r') as f:
            for idx, image_id in enumerate(df_train.ImageId.tolist()):
                im = np.array(f.get_node('/' + image_id))
                im = np.swapaxes(im, 0, 2)
                im = np.swapaxes(im, 1, 2)
                X_val.append(im)
        X_val = np.array(X_val)

        y_val = []
        fn_mask =  self.FMT_VALTRAIN_MASK_STORE.format(prefix)
        with tb.open_file(fn_mask, 'r') as f:
            for idx, image_id in enumerate(df_train.ImageId.tolist()):
                mask = np.array(f.get_node('/' + image_id))
                mask = (mask > 0.5).astype(np.uint8)
                y_val.append(mask)
        y_val = np.array(y_val)
        y_val = y_val.reshape((-1, 1,  self.INPUT_SIZE,  self.INPUT_SIZE))

        return X_val, y_val


    def predict(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        X_test = self._get_test_mul_data(area_id)
        X_mean = self.get_mul_mean_image(area_id)

        # Load model weights
        # Predict and Save prediction result
        model = self.get_unet()
        model.load_weights( self.FMT_VALMODEL_PATH.format(prefix))
        y_pred = model.predict(X_test - X_mean, batch_size=8, verbose=1)
        del model

        # Save prediction result
        fn =  self.FMT_TESTPRED_PATH.format(prefix)
        with tb.open_file(fn, 'w') as f:
            atom = tb.Atom.from_dtype(y_pred.dtype)
            filters = tb.Filters(complib='blosc', complevel=9)
            ds = f.create_carray(f.root, 'pred', atom, y_pred.shape,
                                 filters=filters)
            ds[:] = y_pred


    def _internal_validate_predict_best_param(self, area_id,
                                              enable_tqdm=False):
        param = self._get_model_parameter(area_id)
        epoch = param['fn_epoch']
        y_pred = self._internal_validate_predict(
            area_id,
            epoch=epoch,
            save_pred=False,
            enable_tqdm=enable_tqdm)

        return y_pred


    def _internal_validate_predict(self, area_id,
                                   epoch=3,
                                   save_pred=True,
                                   enable_tqdm=False):
        prefix = self.area_id_to_prefix(area_id)
        X_mean = self.get_mul_mean_image(area_id)

        # Load model weights
        # Predict and Save prediction result
        fn_model =  self.FMT_VALMODEL_PATH.format(prefix + '_{epoch:02d}')
        fn_model = fn_model.format(epoch=epoch)
        model = self.get_unet()
        model.load_weights(fn_model)

        fn_test =  self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test, index_col='ImageId')

        y_pred = model.predict_generator(
            self.generate_valtest_batch(
                area_id,
                batch_size=64,
                immean=X_mean,
                enable_tqdm=enable_tqdm,
            ),
            val_samples=len(df_test),
        )
        del model

        # Save prediction result
        if save_pred:
            fn =  self.FMT_VALTESTPRED_PATH.format(prefix)
            with tb.open_file(fn, 'w') as f:
                atom = tb.Atom.from_dtype(y_pred.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root,
                                     'pred',
                                     atom,
                                     y_pred.shape,
                                     filters=filters)
                ds[:] = y_pred
        return y_pred


    def _internal_validate_fscore_wo_pred_file(self, area_id, epoch=3,
                                               min_th=30,
                                               enable_tqdm=False):
        prefix = self.area_id_to_prefix(area_id)
        min_th = self.MIN_POLYGON_AREA

        # Prediction phase
        self.logger.info("Prediction phase")
        y_pred = self._internal_validate_predict(
            area_id,
            epoch=epoch,
            save_pred=False,
            enable_tqdm=enable_tqdm)

        # Postprocessing phase
        self.logger.info("Postprocessing phase")
        fn_test = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test, index_col='ImageId')

        fn =  self.FMT_VALTESTPRED_PATH.format(prefix)
        fn_out =  self.FMT_VALTESTPOLY_PATH.format(prefix)
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
        fn_true =  self.FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
        df_true = pd.read_csv(fn_true)

        # # Remove prefix "PAN_"
        # df_true.loc[:, 'ImageId'] = df_true.ImageId.str[4:]

        fn_test =  self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test)
        df_test_image_ids = df_test.ImageId.unique()

        fn_out =  self.FMT_VALTESTTRUTH_PATH.format(prefix)
        with open(fn_out, 'w') as f:
            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            df_true = df_true[df_true.ImageId.isin(df_test_image_ids)]
            for idx, r in df_true.iterrows():
                f.write("{},{},\"{}\",{:.6f}\n".format(
                    r.ImageId,
                    r.BuildingId,
                    r.PolygonWKT_Pix,
                    1.0))


    def _internal_validate_fscore(self, area_id,
                                  epoch=3,
                                  predict=True,
                                  min_th=30,
                                  enable_tqdm=False):
        prefix = self.area_id_to_prefix(area_id)

        # Prediction phase
        self.logger.info("Prediction phase")
        if predict:
            self._internal_validate_predict(
                area_id,
                epoch=epoch,
                enable_tqdm=enable_tqdm)

        # Postprocessing phase
            self.logger.info("Postprocessing phase")
        fn_test =  self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test, index_col='ImageId')

        fn =  self.FMT_VALTESTPRED_PATH.format(prefix)
        fn_out =  self.FMT_VALTESTPOLY_PATH.format(prefix)
        with open(fn_out, 'w') as f,\
                tb.open_file(fn, 'r') as fr:

            y_pred = np.array(fr.get_node('/pred'))

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
        # if not Path( self.FMT_VALTESTTRUTH_PATH.format(prefix)).exists():
        if True:
            fn_true =  self.FMT_TRAIN_SUMMARY_PATH.format(prefix=prefix)
            df_true = pd.read_csv(fn_true)
            # # Remove prefix "PAN_"
            # df_true.loc[:, 'ImageId'] = df_true.ImageId.str[4:]

            fn_test =  self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
            df_test = pd.read_csv(fn_test)
            df_test_image_ids = df_test.ImageId.unique()

            fn_out =  self.FMT_VALTESTTRUTH_PATH.format(prefix)
            with open(fn_out, 'w') as f:
                f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
                df_true = df_true[df_true.ImageId.isin(df_test_image_ids)]
                for idx, r in df_true.iterrows():
                    f.write("{},{},\"{}\",{:.6f}\n".format(
                        r.ImageId,
                        r.BuildingId,
                        r.PolygonWKT_Pix,
                        1.0))

    def validate(self, datapath):
        area_id = self.directory_name_to_area_id(datapath)
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info(">> validate sub-command: {}".format(prefix))

        X_mean = self.get_mul_mean_image(area_id)
        X_val, y_val = self._get_valtest_mul_data(area_id)
        X_val = X_val - X_mean

        if not Path( self.MODEL_DIR).exists():
            Path( self.MODEL_DIR).mkdir(parents=True)

        self.logger.info("load valtrain")
        X_trn, y_trn = self._get_valtrain_mul_data(area_id)
        X_trn = X_trn - X_mean

        # model = get_unet()
        # model_checkpoint = ModelCheckpoint(
        #      self.FMT_VALMODEL_PATH.format(prefix + "_{epoch:02d}"),
        #     monitor='val_jaccard_coef_int',
        #     save_best_only=False)
        # model_earlystop = EarlyStopping(
        #     monitor='val_jaccard_coef_int',
        #     patience=10,
        #     verbose=0,
        #     mode='max')
        # model_history = History()
        #
        # df_train = pd.read_csv( self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix))
        # logger.info("Fit")
        # model.fit(
        #     X_trn, y_trn,
        #     nb_epoch=200,
        #     shuffle=True,
        #     verbose=1,
        #     validation_data=(X_val, y_val),
        #     callbacks=[model_checkpoint, model_earlystop, model_history])
        # model.save_weights( self.FMT_VALMODEL_LAST_PATH.format(prefix))
        #
        # # Save evaluation history
        # pd.DataFrame(model_history.history).to_csv(
        #      self.FMT_VALMODEL_HIST.format(prefix), index=False)
        # logger.info(">> validate sub-command: {} ... Done".format(prefix))


    # @cli.command()
    # @click.argument('datapath', type=str)
    def testproc(self, datapath):
        area_id = self.directory_name_to_area_id(datapath)
        prefix = self.area_id_to_prefix(area_id)

        self.logger.info(">>>> Test proc for {}".format(prefix))
        self._internal_test(area_id)
        self.logger.info(">>>> Test proc for {} ... done".format(prefix))


    # @cli.command()
    # @click.argument('datapath', type=str)
    def evalfscore(self, datapath):
        area_id = self.directory_name_to_area_id(datapath)
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("Evaluate fscore on validation set: {}".format(prefix))

        # for each epoch
        # if not Path( self.FMT_VALMODEL_EVALHIST.format(prefix)).exists():
        if True:
            df_hist = pd.read_csv( self.FMT_VALMODEL_HIST.format(prefix))
            df_hist.loc[:, 'epoch'] = list(range(1, len(df_hist) + 1))

            rows = []
            for zero_base_epoch in range(0, len(df_hist)):
                self.logger.info(">>> Epoch: {}".format(zero_base_epoch))
                self._internal_validate_fscore_wo_pred_file(
                    area_id,
                    epoch=zero_base_epoch,
                    enable_tqdm=True,
                    min_th= self.MIN_POLYGON_AREA)
                evaluate_record = self._calc_fscore_per_aoi(area_id)
                evaluate_record['zero_base_epoch'] = zero_base_epoch
                evaluate_record['min_area_th'] =  self.MIN_POLYGON_AREA
                evaluate_record['area_id'] = area_id
                self.logger.info("\n" + json.dumps(evaluate_record, indent=4))
                rows.append(evaluate_record)

            pd.DataFrame(rows).to_csv(
                 self.FMT_VALMODEL_EVALHIST.format(prefix),
                index=False)

        # find best min-poly-threshold
        df_evalhist = pd.read_csv( self.FMT_VALMODEL_EVALHIST.format(prefix))
        best_row = df_evalhist.sort_values(by='fscore', ascending=False).iloc[0]
        best_epoch = int(best_row.zero_base_epoch)
        best_fscore = best_row.fscore

        # optimize min area th
        rows = []
        for th in [30, 60, 90, 120, 150, 180, 210, 240]:
            self.logger.info(">>> TH: {}".format(th))
            predict_flag = False
            if th == 30:
                predict_flag = True

                self._internal_validate_fscore(
                area_id,
                epoch=best_epoch,
                enable_tqdm=True,
                min_th=th,
                predict=predict_flag)
            evaluate_record = self._calc_fscore_per_aoi(area_id)
            evaluate_record['zero_base_epoch'] = best_epoch
            evaluate_record['min_area_th'] = th
            evaluate_record['area_id'] = area_id
            self.logger.info("\n" + json.dumps(evaluate_record, indent=4))
            rows.append(evaluate_record)

        pd.DataFrame(rows).to_csv(
             self.FMT_VALMODEL_EVALTHHIST.format(prefix),
            index=False)

        self.logger.info("Evaluate fscore on validation set: {} .. done".format(prefix))


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


    def postproc(self, area_id):
        # Mask to poly
        print(area_id)
        prefix = self.area_id_to_prefix(area_id)
        fn_test =  self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test, index_col='ImageId')
        fn =  self.FMT_TESTPRED_PATH.format(prefix)
        with tb.open_file(fn, 'r') as f:
            y_pred = np.array(f.get_node('/pred'))
        print(y_pred.shape)

        fn_out =  self.FMT_TESTPOLY_PATH.format(prefix)
        with open(fn_out, 'w') as f:
            f.write("ImageId,BuildingId,PolygonWKT_Pix,Confidence\n")
            for idx, image_id in enumerate(df_test.index.tolist()):
                df_poly = self.mask_to_poly(y_pred[idx][0])
                if len(df_poly) > 0:
                    for i, row in df_poly.iterrows():
                        f.write("{},{},\"{}\",{:.6f}\n".format(
                            image_id,
                            row.bid,
                            row.wkt,
                            row.area_ratio))
                else:
                    f.write("{},{},{},0\n".format(
                        image_id,
                        -1,
                        "POLYGON EMPTY"))


    def merge(self):
        df_list = []
        for area_id in range(2, 6):
            prefix = self.area_id_to_prefix(area_id)
            df_part = pd.read_csv(
                 self.FMT_TESTPOLY_PATH.format(prefix))
            df_list.append(df_part)
        df = pd.concat(df_list)
        df.to_csv( self.FN_SOLUTION_CSV, index=False)

        with open( self.FN_SOLUTION_CSV, 'r') as f:
            lines = f.readlines()
        with open( self.FN_SOLUTION_CSV, 'w') as f:
            f.write(lines[0])
            for line in lines[1:]:
                line = self._remove_interiors(line)
                f.write(line)


plugin_config = "/home/geesara/project/satelliteimageclassifier/config/config.json"
dataprocessor = ModleTrain(plugin_config)
numberOfChannels = 3
dataprocessor.validate("/data/train/AOI_5_Khartoum_Train")
