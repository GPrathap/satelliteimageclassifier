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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

dataprocessor_v3 = DataProcessor(plugin_config, base_model_name="v1", model_name="v3",
                                 image_dir="v3", is_final=True)
dataprocessor_v3.execute()

net_v3, trainer_v3, operators_v3 = dataprocessor_v3.get_model(type_of_data)
dataprocessor_v3.validate(trainer_v3, operators_v3, display_step=2, restore=True)
number_of_models = dataprocessor_v3.get_total_numberof_model_count(trainer_v3)


dataprocessor_v3.evalfscore_v16(trainer_v3, operators_v3, number_of_models)



