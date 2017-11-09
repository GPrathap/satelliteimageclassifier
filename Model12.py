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

dataprocessor_v12 = DataProcessor(plugin_config, base_model_name="v7", model_name="v12", image_dir="v12")
dataprocessor_v12.execute()

net_v12, trainer_v12, operators_v12 = dataprocessor_v12.get_model(type_of_data)
dataprocessor_v12.validate(trainer_v12, operators_v12, display_step=2, restore=True)
number_of_models = dataprocessor_v12.get_total_numberof_model_count(trainer_v12)
dataprocessor_v12.evalfscore_v12(trainer_v12, operators_v12, number_of_models)