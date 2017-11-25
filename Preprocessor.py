import matplotlib.pyplot as plt
import tensorflow as tf

from DataProcessor import DataProcessor

plt.rcParams['image.cmap'] = 'gist_earth'

plugin_config = "./config/config.json"
type_of_data="multi"

dataprocessor = DataProcessor(plugin_config, base_model_name="v1", model_name="v1", image_dir="v1")
dataprocessor.execute()

dataprocessor.preproc_train()
writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_train)
dataprocessor._get_valtrain_mul_data(writer_multi)
writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_test)
dataprocessor._get_valtest_mul_data(writer_multi)
writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_final_test)
dataprocessor.preproc_test(writer_multi)


dataprocessor = DataProcessor(plugin_config, base_model_name="v1", model_name="v2", image_dir="v2")
dataprocessor.execute()
#
dataprocessor.preproc_train_v2()
writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_train)
dataprocessor.get_valtrain_data_v2(writer_multi)
writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_test)
dataprocessor.get_valtest_data_v2(writer_multi)
writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_final_test)
dataprocessor.preproc_test_v2(writer_multi)

dataprocessor = DataProcessor(plugin_config, base_model_name="v1", model_name="v3", image_dir="v3",
                              is_final=True)
dataprocessor.execute()
dataprocessor.preproc_train_v3()
writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_test)
dataprocessor.generate_valtest_batch( writer_multi)
writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_train)
dataprocessor.generate_valtrain_batch(writer_multi)
writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_final_test)
dataprocessor.preproc_test_v3(writer_multi)