from DataProcessor import DataProcessor
from tf_unetnew.tf_unet import unet

import matplotlib.pyplot as plt
import os
import tensorflow as tf
from DataProviderGSI import GISDataProvider
from QueueLoader import QueueLoader
plt.rcParams['image.cmap'] = 'gist_earth'
from collections import namedtuple

plugin_config = "./config/config.json"
type_of_data="multi"

dataprocessor = DataProcessor(plugin_config, base_model_name="v7", model_name="v7", image_dir="v7")
dataprocessor.execute()

dataprocessor.preproc_train()
writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_train)
dataprocessor._get_valtrain_mul_data(writer_multi)
writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_test)
dataprocessor._get_valtest_mul_data(writer_multi)

dataprocessor = DataProcessor(plugin_config, base_model_name="v7", model_name="v12", image_dir="v12")
dataprocessor.execute()

dataprocessor.preproc_train_v12()
writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_train)
dataprocessor.get_valtrain_data_v12(writer_multi)
writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_test)
dataprocessor.get_valtest_data_v12(writer_multi)

dataprocessor = DataProcessor(plugin_config, base_model_name="v7", model_name="v16", image_dir="v16",
                              is_final=True)
dataprocessor.execute()
dataprocessor.preproc_train_v16()
writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_test)
dataprocessor.generate_valtest_batch( writer_multi)
writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_train)
dataprocessor.generate_valtrain_batch(writer_multi)
