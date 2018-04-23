import matplotlib.pyplot as plt

from DataProcessor import DataProcessor

plt.rcParams['image.cmap'] = 'gist_earth'

plugin_config = "./config/config.json"
type_of_data="multi"


dataprocessor_v1 = DataProcessor(plugin_config, base_model_name="v1", model_name="v1", image_dir="v1")
dataprocessor_v1.execute()
net_v1, trainer_v1, operators_v1 = dataprocessor_v1.get_test_model(type_of_data)
y_pred_0, test_image_ids_v1 = dataprocessor_v1.test(trainer_v1, operators_v1, display_step=2, restore=True)

dataprocessor_v2 = DataProcessor(plugin_config, base_model_name="v1", model_name="v2", image_dir="v2")
dataprocessor_v2.execute()
net_v2, trainer_v2, operators_v2 = dataprocessor_v2.get_test_model(type_of_data)
y_pred_1, test_image_ids_v2 = dataprocessor_v2.test(trainer_v2, operators_v2, display_step=2, restore=True)


dataprocessor_v3 = DataProcessor(plugin_config, base_model_name="v1", model_name="v3", image_dir="v3",
                                 is_final=True)
dataprocessor_v3.execute()
net_v3, trainer_v3, operators_v3 = dataprocessor_v3.get_test_model(type_of_data)
y_pred_2, test_image_ids_v3 = dataprocessor_v3.test(trainer_v3, operators_v3, display_step=2, restore=True)



dataprocessor_v17 = DataProcessor(plugin_config, base_model_name="v1", model_name="vfinal", image_dir="vfinal",
                            is_final=False)
dataprocessor_v17.execute()
dataprocessor_v17.testproc(y_pred_0, y_pred_1, y_pred_2)


