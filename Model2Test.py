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

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

dataprocessor_v2 = DataProcessor(plugin_config, base_model_name="v1", model_name="v2", image_dir="v2")
dataprocessor_v2.execute()

net_v2, trainer_v2, operators_v2 = dataprocessor_v2.get_test_model(type_of_data)
predictiong, test_image_ids = dataprocessor_v2.test(trainer_v2, operators_v2, display_step=2, restore=True)

print(predictiong)