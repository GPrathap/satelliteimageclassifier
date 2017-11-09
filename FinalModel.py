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

dataprocessor_v7 = DataProcessor(plugin_config, base_model_name="v7", model_name="v7", image_dir="v7")
dataprocessor_v7.execute()
net_v7, trainer_v7, operators_v7 = dataprocessor_v7.get_model(type_of_data)
y_pred_0, images_idsv7 = dataprocessor_v7._internal_validate_predict_best_param("v7", trainer_v7, operators_v7,
                                              enable_tqdm=False)


dataprocessor_v12 = DataProcessor(plugin_config, base_model_name="v7", model_name="v12", image_dir="v12")
dataprocessor_v12.execute()
net_v12, trainer_v12, operators_v12 = dataprocessor_v12.get_model(type_of_data)
y_pred_1, images_idsv12 = dataprocessor_v12._internal_validate_predict_best_param("v12", trainer_v12, operators_v12,
                                              enable_tqdm=False)


dataprocessor_v16 = DataProcessor(plugin_config, base_model_name="v7", model_name="v16", image_dir="v16", is_final=True)
dataprocessor_v16.execute()
net_v16, trainer_v16, operators_v16 = dataprocessor_v16.get_model(type_of_data)
y_pred_2, images_idsv16 = dataprocessor_v16._internal_validate_predict_best_param("v16", trainer_v16, operators_v16,
                                              enable_tqdm=False)

dataprocessor_v17 = DataProcessor(plugin_config, base_model_name="v7", model_name="v17", image_dir="v17",
                            is_final=False)
dataprocessor_v17.execute()
dataprocessor_v17.evalfscore_v17(y_pred_0, y_pred_1, y_pred_2)

## TODO get this thing working
# dataprocessor_v17.testproc(y_pred_0, y_pred_1, y_pred_2)


