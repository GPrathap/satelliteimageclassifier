import matplotlib.pyplot as plt

from DataProcessor import DataProcessor

plt.rcParams['image.cmap'] = 'gist_earth'

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

plugin_config = "./config/config.json"
type_of_data="multi"

dataprocessor_v1 = DataProcessor(plugin_config, base_model_name="v1", model_name="v1", image_dir="v1")
dataprocessor_v1.execute()

net_v1, trainer_v1, operators_v1 = dataprocessor_v1.get_test_model(type_of_data)
predictiong, test_image_ids = dataprocessor_v1.test(trainer_v1, operators_v1, display_step=2, restore=True)
