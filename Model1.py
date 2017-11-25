import matplotlib.pyplot as plt

from DataProcessor import DataProcessor

plt.rcParams['image.cmap'] = 'gist_earth'

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

plugin_config = "./config/config.json"
type_of_data="multi"

dataprocessor_v1 = DataProcessor(plugin_config, base_model_name="v1", model_name="v1", image_dir="v1")
dataprocessor_v1.execute()

net_v1, trainer_v1, operators_v1 = dataprocessor_v1.get_model(type_of_data)
dataprocessor_v1.validate(trainer_v1, operators_v1, display_step=2, restore=True)
number_of_models = dataprocessor_v1.get_total_numberof_model_count(trainer_v1)
dataprocessor_v1.evalfscore(trainer_v1, operators_v1, number_of_models)