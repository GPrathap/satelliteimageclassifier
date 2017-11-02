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
datapath="/data/train/AOI_5_Khartoum_Train"

# dataprocessor = DataProcessor(plugin_config, base_model_name="v7", model_name="v7", image_dir="v7")
# dataprocessor.execute()
#
# dataprocessor.preproc_train("/data/train/AOI_5_Khartoum_Train")
# writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_train)
# dataprocessor._get_valtrain_mul_data(5, writer_multi)
# writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_test)
# dataprocessor._get_valtest_mul_data(5, writer_multi)

# dataprocessor = DataProcessor(plugin_config, base_model_name="v7", model_name="v12", image_dir="v12")
# dataprocessor.execute()

# dataprocessor.preproc_train_v12("/data/train/AOI_5_Khartoum_Train")
# writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_train)
# dataprocessor.get_valtrain_data_v12(5, writer_multi)
# writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_test)
# dataprocessor.get_valtest_data_v12(5, writer_multi)

dataprocessor = DataProcessor(plugin_config, base_model_name="v7", model_name="v16", image_dir="v16",
                              is_final=True)
dataprocessor.execute()
# dataprocessor.preproc_train_v16(datapath)
# writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_test)
# dataprocessor.generate_valtest_batch(datapath, writer_multi)
# writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_train)
# dataprocessor.generate_valtrain_batch(datapath, writer_multi)


batch_size_for_net=1
batch_size_for__test_net=1
epochs=20
epochs_test=1
additianl_channals = 0



if dataprocessor.MODEL_NAME=="v16":
    additianl_channals = 4

generator_train = GISDataProvider(plugin_config, dataprocessor, additianl_channals=additianl_channals,
                                  type=type_of_data, train=True)
generator_test = GISDataProvider(plugin_config, dataprocessor, additianl_channals=additianl_channals,
                                 type=type_of_data, train=False)

net = unet.Unet(channels=generator_train.channels + additianl_channals, n_class=generator_train.classes, layers=3,
                    features_root=16)
trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2),
                       batch_size=batch_size_for_net,
                       verification_batch_size=1)

queue_loader_train = QueueLoader(plugin_config, dataprocessor, type=type_of_data,
                                 batch_size=batch_size_for_net, additianl_channals=additianl_channals,
                                 num_epochs=epochs, train=True)
queue_loader_validate = QueueLoader(plugin_config, dataprocessor, type=type_of_data,
                                    batch_size=batch_size_for__test_net,
                                    additianl_channals=additianl_channals, num_epochs=epochs_test, train=False)
place_holder = namedtuple('modelTrain', 'train_dataset test_dataset loader_train loader_test')
operators = place_holder(train_dataset=generator_train, test_dataset=generator_test,
                         loader_train=queue_loader_train, loader_test=queue_loader_validate)

dataprocessor.validate(datapath, trainer, operators, training_iters=4, num_epochs=epochs, display_step=2,
                       restore=True)

# dataprocessor.evalfscore(datapath, trainer, operators, 3)
# dataprocessor.preproc_train("/data/train/AOI_5_Khartoum_Train")
# writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_train)
# dataprocessor._get_valtrain_mul_data(5, writer_multi)
# writer_multi = tf.python_io.TFRecordWriter(dataprocessor.tfrecords_filename_multi_test)
# dataprocessor._get_valtest_mul_data(5, writer_multi)



