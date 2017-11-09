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

dataprocessor_v16 = DataProcessor(plugin_config, base_model_name="v7", model_name="v16", image_dir="v16", is_final=True)
dataprocessor_v16.execute()

net_v16, trainer_v16, operators_v16 = dataprocessor_v16.get_model(type_of_data)
dataprocessor_v16.validate(trainer_v16, operators_v16, display_step=2, restore=True)
number_of_models = dataprocessor_v16.get_total_numberof_model_count(trainer_v16)
dataprocessor_v16.evalfscore_v16(trainer_v16, operators_v16, number_of_models)



