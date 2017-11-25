import matplotlib.pyplot as plt

from DataProcessor import DataProcessor

plt.rcParams['image.cmap'] = 'gist_earth'

plugin_config = "./config/config.json"
type_of_data="multi"

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

dataprocessor_v3 = DataProcessor(plugin_config, base_model_name="v1", model_name="v3",
                                 image_dir="v3", is_final=True)
dataprocessor_v3.execute()

net_v3, trainer_v3, operators_v3 = dataprocessor_v3.get_test_model(type_of_data)
predictiong, test_image_ids = dataprocessor_v3.test(trainer_v3, operators_v3, display_step=2, restore=True)




