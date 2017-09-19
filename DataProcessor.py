from __future__ import print_function
import json
import os
import sys
from logging import getLogger, Formatter, StreamHandler, INFO
from pathlib import Path
import glob
import warnings
import scipy
import tqdm
import tables as tb
import pandas as pd
import numpy as np
import skimage.transform
import rasterio
import shapely.wkt
from PIL import Image
import numpy as np
import skimage.io as io
import tensorflow as tf

# Logger
warnings.simplefilter("ignore", UserWarning)

class DataProcessor():

    def __init__(self, plugin_config):
        spacenet_util_dir = os.getcwd() + '/spaceNetUtilities'
        sys.path.extend([spacenet_util_dir])
        self.MODEL_NAME = 'v5'
        self.ORIGINAL_SIZE = 650
        self.INPUT_SIZE = 256
        self.BASE_TRAIN_DIR = "/data/train"
        self.WORKING_DIR = "/data/working"
        self.IMAGE_DIR = "/data/working/images/v5"
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
        self.FMT_MUL_BANDCUT_TH_PATH = self.IMAGE_DIR + "/mul_bandcut{}.csv"
        self.FMT_VALTRAIN_IMAGELIST_PATH = self.IMAGE_DIR + "/{prefix:s}_valtrain_ImageId.csv"
        self.FMT_VALTRAIN_MASK_STORE = self.IMAGE_DIR + "/valtrain_{}_mask.h5"
        self.FMT_VALTRAIN_IM_STORE = self.IMAGE_DIR + "/valtrain_{}_im.h5"
        self.FMT_VALTRAIN_MUL_STORE = self.IMAGE_DIR + "/valtrain_{}_mul.h5"
        self.FMT_VALTEST_IMAGELIST_PATH = self.IMAGE_DIR + "/{prefix:s}_valtest_ImageId.csv"
        self.FMT_VALTEST_MASK_STORE = self.IMAGE_DIR + "/valtest_{}_mask.h5"
        self.FMT_VALTEST_IM_STORE = self.IMAGE_DIR + "/valtest_{}_im.h5"
        self.FMT_VALTEST_MUL_STORE = self.IMAGE_DIR + "/valtest_{}_mul.h5"
        self.FMT_IMMEAN = self.IMAGE_DIR + "/{}_immean.h5"
        self.FMT_MULMEAN = self.IMAGE_DIR + "/{}_mulmean.h5"
        self.FMT_TEST_IMAGELIST_PATH = self.IMAGE_DIR + "/{prefix:s}_test_ImageId.csv"
        self.FMT_TEST_IM_STORE = self.IMAGE_DIR + "/test_{}_im.h5"
        self.FMT_TEST_MUL_STORE = self.IMAGE_DIR + "/test_{}_mul.h5"

        self.handler = StreamHandler()
        self.handler.setLevel(INFO)
        self.handler.setFormatter(Formatter('%(asctime)s %(levelname)s %(message)s'))
        self.logger = getLogger('spacenet')
        self.logger.setLevel(INFO)
        np.random.seed(1145141919)
        self.LOGFORMAT = '%(asctime)s %(levelname)s %(message)s'
        self.BASE_DIR = "/data/train"
        self.MODEL_DIR = "/data/working/models/{}".format(self.MODEL_NAME)
        self.FN_SOLUTION_CSV = "/data/output/{}.csv".format(self.MODEL_NAME)

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
        self.FMT_MUL_BANDCUT_TH_PATH = self.IMAGE_DIR + "/mul_bandcut{}.csv"
        self.tfrecords_filename_rgb = self.IMAGE_DIR + '/rgb_segmentation.tfrecords'
        self.tfrecords_filename_multi = self.IMAGE_DIR + '/multi_segmentation.tfrecords'
        # self.writer_rgb = tf.python_io.TFRecordWriter(self.tfrecords_filename_rgb)
        self.writer_multi = tf.python_io.TFRecordWriter(self.tfrecords_filename_multi)

        with open(plugin_config) as plugin_config:
            self.plugin_config = json.load(plugin_config)

    def createRFRecoad(self, img, annotation, writer):
        height = img.shape[0]
        width = img.shape[1]
        channels = img.shape[2]
        mask_height = annotation.shape[0]
        mask_width = annotation.shape[1]
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
            'mask_raw': self._bytes_feature(annotation_raw)}))
        writer.write(example.SerializeToString())

    def read_and_decode(self, filename, numberOfChannels):
        filename_queue = tf.train.string_input_producer([filename], num_epochs=1)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        features = tf.parse_single_example(
            serialized_example,
            # Defaults are not specified since both keys are required.
            features={
                'image_height': tf.FixedLenFeature([], tf.int64),
                'image_width': tf.FixedLenFeature([], tf.int64),
                'channels': tf.FixedLenFeature([], tf.int64),
                'mask_height': tf.FixedLenFeature([], tf.int64),
                'mask_width': tf.FixedLenFeature([], tf.int64),
                'mask_channels': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'mask_raw': tf.FixedLenFeature([], tf.string)
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        annotation = tf.decode_raw(features['mask_raw'], tf.uint8)

        height = tf.cast(features['image_height'], tf.int32)
        width = tf.cast(features['image_width'], tf.int32)
        channels = tf.cast(features['channels'], tf.int32)

        print (channels)

        mask_height = tf.cast(features['mask_height'], tf.int32)
        mask_width = tf.cast(features['mask_width'], tf.int32)
        mask_channels = tf.cast(features['mask_channels'], tf.int32)

        image_shape = tf.stack([height, width, numberOfChannels])
        annotation_shape = tf.stack([mask_height, mask_width, 1])

        image = tf.reshape(image, image_shape)
        annotation = tf.reshape(annotation, annotation_shape)

        # image_size_const = tf.constant((self.INPUT_SIZE, self.INPUT_SIZE, 3), dtype=tf.int32)
        # annotation_size_const = tf.constant((self.INPUT_SIZE, self.INPUT_SIZE, 1), dtype=tf.int32)

        # Random transformations can be put here: right before you crop images
        # to predefined size. To get more information look at the stackoverflow
        # question linked above.

        resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                               target_height=self.INPUT_SIZE,
                                                               target_width=self.INPUT_SIZE)

        resized_annotation = tf.image.resize_image_with_crop_or_pad(image=annotation,
                                                                    target_height=self.INPUT_SIZE,
                                                                    target_width=self.INPUT_SIZE)

        images, annotations = tf.train.shuffle_batch([resized_image, resized_annotation],
                                                      batch_size=1,
                                                      capacity=30,
                                                      num_threads=1,
                                                      min_after_dequeue=1)

        return images, annotations

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

    def prep_rgb_image_store_test(self, area_id, datapath):
        prefix = self.area_id_to_prefix(area_id)
        bandstats = self.__load_rgb_bandstats(area_id)

        self.logger.info("prep_rgb_image_store_test for {}".format(prefix))
        fn_list = self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)
        fn_store = self.FMT_TEST_IM_STORE.format(prefix)
        df_list = pd.read_csv(fn_list, index_col='ImageId')

        self.logger.info("Image store file: {}".format(fn_store))
        with tb.open_file(fn_store, 'w') as f:
            for image_id in tqdm.tqdm(df_list.index, total=len(df_list)):
                im = self.get_resized_3chan_image_test(image_id, datapath, bandstats)
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

    def prep_mul_image_store_test(self, area_id, datapath):
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
                    image_id, datapath, bandstats_rgb, bandstats_mul)
                atom = tb.Atom.from_dtype(im.dtype)
                filters = tb.Filters(complib='blosc', complevel=9)
                ds = f.create_carray(f.root, image_id, atom, im.shape,
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

    def prep_test_imagelist(self, area_id, datapath):
        prefix = self.area_id_to_prefix(area_id)

        image_id_list = glob.glob(str(
            Path(datapath) /
            Path("./PAN/PAN_{prefix:s}_*.tif")).format(prefix=prefix))
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

    def preproc_train(self, datapath):
        """ train.sh """
        area_id = self.directory_name_to_area_id(datapath)
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("Preproc for training on {}".format(prefix))

        # Imagelist
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

        # Band stats (RGB)
        if Path(self.FMT_RGB_BANDCUT_TH_PATH.format(prefix)).exists():
            self.logger.info("Generate band stats csv (RGB) ... skip")
        else:
            self.logger.info("Generate band stats csv (RGB)")
            self.calc_rgb_multiband_cut_threshold(area_id, datapath)

        # Band stats (MUL)
        if Path(self.FMT_MUL_BANDCUT_TH_PATH.format(prefix)).exists():
            self.logger.info("Generate band stats csv (MUL) ... skip")
        else:
            self.logger.info("Generate band stats csv (MUL)")
            self.calc_mul_multiband_cut_threshold(area_id, datapath)

        # Mask (Target output)
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

        # Image HDF5 store (RGB)
        if Path(self.FMT_VALTRAIN_IM_STORE.format(prefix)).exists():
            self.logger.info("Generate RGB_STORE (valtrain) ... skip")
        else:
            self.logger.info("Generate RGB_STORE (valtrain)")
            self.prep_rgb_image_store_train(area_id, datapath, is_valtrain=True)
        if Path(self.FMT_VALTEST_IM_STORE.format(prefix)).exists():
            self.logger.info("Generate RGB_STORE (valtest) ... skip")
        else:
            self.logger.info("Generate RGB_STORE (valtest)")
            self.prep_rgb_image_store_train(area_id, datapath, is_valtrain=False)

        # Image HDF5 store (MUL)
        if Path(self.FMT_VALTRAIN_MUL_STORE.format(prefix)).exists():
            self.logger.info("Generate MUL_STORE (valtrain) ... skip")
        else:
            self.logger.info("Generate MUL_STORE (valtrain)")
            self.prep_mul_image_store_train(area_id, datapath, is_valtrain=True)
        if Path(self.FMT_VALTEST_MUL_STORE.format(prefix)).exists():
            self.logger.info("Generate MUL_STORE (valtest) ... skip")
        else:
            self.logger.info("Generate MUL_STORE (valtest)")
            self.prep_mul_image_store_train(area_id, datapath, is_valtrain=False)

        # Image Mean (RGB)
        if Path(self.FMT_IMMEAN.format(prefix)).exists():
            self.logger.info("Generate RGBMEAN ... skip")
        else:
            self.logger.info("Generate RGBMEAN")
            self.prep_immean(area_id, datapath)

        # Image Mean (MUL)
        if Path(self.FMT_MULMEAN.format(prefix)).exists():
            self.logger.info("Generate MULMEAN ... skip")
        else:
            self.logger.info("Generate MULMEAN")
            self.prep_mulmean(area_id)

        # DONE!
            self.logger.info("Preproc for training on {} ... done".format(prefix))

    def preproc_test(self, datapath):
        """ test.sh """
        area_id = self.directory_name_to_area_id(datapath)
        prefix = self.area_id_to_prefix(area_id)
        self.logger.info("preproc_test for {}".format(prefix))

        # Imagelist
        if Path(self.FMT_TEST_IMAGELIST_PATH.format(prefix=prefix)).exists():
            self.logger.info("Generate IMAGELIST for inference ... skip")
        else:
            self.logger.info("Generate IMAGELIST for inference")
            self.prep_test_imagelist(area_id, datapath)

        # Image HDF5 store (RGB)
        if Path(self.FMT_TEST_IM_STORE.format(prefix)).exists():
            self.logger.info("Generate RGB_STORE (test) ... skip")
        else:
            self.logger.info("Generate RGB_STORE (test)")
            self.prep_rgb_image_store_test(area_id, datapath)

        # Image HDF5 store (MUL)
        if Path(self.FMT_TEST_MUL_STORE.format(prefix)).exists():
            self.logger.info("Generate MUL_STORE (test) ... skip")
        else:
            self.logger.info("Generate MUL_STORE (test)")
            self.prep_mul_image_store_test(area_id, datapath)

            self.logger.info("preproc_test for {} ... done".format(prefix))

    def _get_valtest_mul_data(self, area_id):
        prefix = self.area_id_to_prefix(area_id)
        fn_test = self.FMT_VALTEST_IMAGELIST_PATH.format(prefix=prefix)
        df_test = pd.read_csv(fn_test)

        X_val = []
        fn_im = self.FMT_VALTEST_MUL_STORE.format(prefix)
        with tb.open_file(fn_im, 'r') as f:
            for idx, image_id in enumerate(df_test.ImageId.tolist()):
                im = np.array(f.get_node('/' + image_id))
                im = np.swapaxes(im, 0, 2)
                im = np.swapaxes(im, 1, 2)
                X_val.append(im)
        X_val = np.array(X_val)

        y_val = []
        fn_mask = self.FMT_VALTEST_MASK_STORE.format(prefix)
        with tb.open_file(fn_mask, 'r') as f:
            for idx, image_id in enumerate(df_test.ImageId.tolist()):
                mask = np.array(f.get_node('/' + image_id))
                mask = (mask > 0.5).astype(np.uint8)
                y_val.append(mask)
        y_val = np.array(y_val)
        y_val = y_val.reshape((-1, 1, self.INPUT_SIZE, self.INPUT_SIZE))

        return X_val, y_val




    def _get_valtrain_mul_data(self, area_id, writer):
        prefix = self.area_id_to_prefix(area_id)
        fn_train = self.FMT_VALTRAIN_IMAGELIST_PATH.format(prefix=prefix)
        df_train = pd.read_csv(fn_train)
        fn_im = self.FMT_VALTRAIN_MUL_STORE.format(prefix)
        fn_mask = self.FMT_VALTRAIN_MASK_STORE.format(prefix)
        with tb.open_file(fn_im, 'r') as f:
            with tb.open_file(fn_mask, 'r') as af:
                for idx, image_id in enumerate(df_train.ImageId.tolist()):
                    im = np.array(f.get_node('/' + image_id))
                    im = np.swapaxes(im, 0, 2)
                    im = np.swapaxes(im, 1, 2)
                    im = im[0:3]
                    mask = np.array(af.get_node('/' + image_id))
                    mask = (mask > 0.5).astype(np.uint8)
                    self.createRFRecoad(im, mask, writer)
        writer.close()
        return "File has been written..."


    def execute(self):
        self.logger.addHandler(self.handler)


plugin_config = "/home/geesara/project/satelliteimageclassifier/config/config.json"
dataprocessor = DataProcessor(plugin_config)
dataprocessor.execute()
numberOfChannels = 3
# dataprocessor.preproc_train("/data/train/AOI_5_Khartoum_Train")
dataprocessor._get_valtrain_mul_data(5, dataprocessor.writer_multi)
image, annotation = dataprocessor.read_and_decode(dataprocessor.tfrecords_filename_multi, numberOfChannels)

















#
# init_op = tf.group(tf.global_variables_initializer(),
#                    tf.local_variables_initializer())
#
# with tf.Session()  as sess:
#     sess.run(init_op)
#
#     # coord = tf.train.Coordinator()
#     # threads = tf.train.start_queue_runners(coord=coord)
#
#     # Let's read off 3 batches just for example
#     # for i in xrange(1):
#     img, anno = sess.run([image, annotation])
#     print(img[0, :, :, :].shape)
#
#     print('current batch')
#
#     # We selected the batch size of two
#     # So we should get two image pairs in each batch
#     # Let's make sure it is random
#
#     io.imshow(img[0, :, :, :])
#     io.show()
#
#     io.imshow(anno[0, :, :, 0])
#     io.show()
#
#     io.imshow(img[1, :, :, :])
#     io.show()
#
#     io.imshow(anno[1, :, :, 0])
#     io.show()
#
#     # coord.request_stop()
#     # coord.join(threads)
