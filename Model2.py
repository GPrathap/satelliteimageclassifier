import matplotlib.pyplot as plt

from DataProcessor import DataProcessor

plt.rcParams['image.cmap'] = 'gist_earth'

plugin_config = "./config/config.json"
type_of_data="multi"

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

dataprocessor_v2 = DataProcessor(plugin_config, base_model_name="v1", model_name="v2", image_dir="v2")
dataprocessor_v2.execute()

net_v2, trainer_v2, operators_v2 = dataprocessor_v2.get_model(type_of_data)
dataprocessor_v2.validate(trainer_v2, operators_v2, display_step=2, restore=True)
number_of_models = dataprocessor_v2.get_total_numberof_model_count(trainer_v2)
dataprocessor_v2.evalfscore_v12(trainer_v2, operators_v2, number_of_models)