import matplotlib.pyplot as plt

from DataProcessor import DataProcessor

plt.rcParams['image.cmap'] = 'gist_earth'

plugin_config = "./config/config.json"
type_of_data="multi"

dataprocessor_v7 = DataProcessor(plugin_config, base_model_name="v1", model_name="v1", image_dir="v1")
dataprocessor_v7.execute()
net_v7, trainer_v7, operators_v7 = dataprocessor_v7.get_model(type_of_data)
y_pred_0, images_idsv7 = dataprocessor_v7._internal_validate_predict_best_param("v1", trainer_v7, operators_v7,
                                              enable_tqdm=False)


dataprocessor_v12 = DataProcessor(plugin_config, base_model_name="v1", model_name="v2", image_dir="v2")
dataprocessor_v12.execute()
net_v12, trainer_v12, operators_v12 = dataprocessor_v12.get_model(type_of_data)
y_pred_1, images_idsv12 = dataprocessor_v12._internal_validate_predict_best_param("v2", trainer_v12, operators_v12,
                                              enable_tqdm=False)


dataprocessor_v16 = DataProcessor(plugin_config, base_model_name="v1", model_name="v3", image_dir="v3", is_final=True)
dataprocessor_v16.execute()
net_v16, trainer_v16, operators_v16 = dataprocessor_v16.get_model(type_of_data)
y_pred_2, images_idsv16 = dataprocessor_v16._internal_validate_predict_best_param("v3", trainer_v16, operators_v16,
                                              enable_tqdm=False)

dataprocessor_v17 = DataProcessor(plugin_config, base_model_name="v1", model_name="v4", image_dir="v4",
                            is_final=False)
dataprocessor_v17.execute()
dataprocessor_v17.evalfscore_v17(y_pred_0, y_pred_1, y_pred_2)



