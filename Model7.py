from DataProcessor import DataProcessor
from tf_unetnew.tf_unet import unet

import matplotlib.pyplot as plt
import os
import tensorflow as tf
from DataProviderGSI import GISDataProvider
from QueueLoader import QueueLoader
plt.rcParams['image.cmap'] = 'gist_earth'
from collections import namedtuple

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

plugin_config = "./config/config.json"
type_of_data="multi"

dataprocessor_v7 = DataProcessor(plugin_config, base_model_name="v7", model_name="v7", image_dir="v7")
dataprocessor_v7.execute()
net_v7, trainer_v7, operators_v7 = dataprocessor_v7.get_model(type_of_data)
dataprocessor_v7.validate(trainer_v7, operators_v7, display_step=2, restore=True)
number_of_models = dataprocessor_v7.get_total_numberof_model_count(trainer_v7)
dataprocessor_v7.evalfscore(trainer_v7, operators_v7, number_of_models)