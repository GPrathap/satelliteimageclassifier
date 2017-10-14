import json

import glob
import json

import numpy as np
from PIL import Image

class GISDataProvider():
    def __init__(self, plugin_config, type,a_min=None, a_max=None, train=True):
        with open(plugin_config) as plugin_config:
            self.plugin_config = json.load(plugin_config)
            self.a_min = a_min if a_min is not None else -np.inf
            self.a_max = a_max if a_min is not None else np.inf
            if(type=="rgb"):
                self.channels = 3
                if(train):
                    self.file_name = str(self.plugin_config["tfrecords_filename_rgb_train"])
                else:
                    self.file_name = str(self.plugin_config["tfrecords_filename_rgb_test"])
            else:
                self.channels = int(self.plugin_config["multi_band_size"])
                if (train):
                    self.file_name = str(self.plugin_config["tfrecords_filename_multi_train"])
                else:
                    self.file_name = str(self.plugin_config["tfrecords_filename_multi_test"])
            self.classes = 2
            self.width = int(self.plugin_config["width_of_image"])
            self.height = int(self.plugin_config["height_of_image"])

    def _load_data_and_label(self, data, label):
        train_data = self._process_data(data)
        labels = self._process_labels(label)
        train_data, labels = self._post_process(train_data, labels)
        nx = data.shape[1]
        ny = data.shape[0]
        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.classes),

    def _process_labels(self, label):
        if self.classes == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.classes), dtype=np.float32)
            labels[..., 1] = label
            labels[..., 0] = ~label
            return labels
        return label

    def _process_data(self, data):
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        data /= np.amax(data)
        return data

    def _post_process(self, data, labels):
        return data, labels

    def __call__(self, data, labels):
        size_of_batch = data.shape[0]
        width_of_image = self.width
        height_of_image = self.height
        batch_x = np.zeros((size_of_batch, width_of_image, height_of_image, self.channels))
        batch_y = np.zeros((size_of_batch, width_of_image, height_of_image, self.classes))
        for i in range(data.shape[0]):
                batch_x[i], batch_y[i] = self._load_data_and_label(np.reshape(data[i],
                                                       (width_of_image, height_of_image, self.channels)),
                                                       np.reshape(labels[i], (width_of_image, height_of_image)).
                                                       astype(np.bool))
        return batch_x, batch_y
