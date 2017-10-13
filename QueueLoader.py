import json
import os
import tensorflow as tf

class QueueLoader():
    def __init__(self, plugin_config, type ,batch_size, num_epochs, num_threads=1,
                 min_after_dequeue=1000, train=True):
        with open(plugin_config) as plugin_config:
            self.plugin_config = json.load(plugin_config)
            if(type=="rgb"):
                self.img_shape = [int(self.plugin_config["width_of_image"]),
                                  int(self.plugin_config["height_of_image"]), 3]
                self.label_shape = [int(self.plugin_config["width_of_image"]),
                                    int(self.plugin_config["height_of_image"]), 1]
            else:
                self.img_shape = [int(self.plugin_config["width_of_image"]),
                                  int(self.plugin_config["height_of_image"]),
                                  int(self.plugin_config["multi_band_size"])]
                self.label_shape = [int(self.plugin_config["width_of_image"]),
                                    int(self.plugin_config["height_of_image"]), 1]
            if(train):
                if(type=="rgb"):
                    file_name = str(self.plugin_config["tfrecords_filename_rgb_train"])
                else:
                    file_name = str(self.plugin_config["tfrecords_filename_rgb_test"])
            else:
                if (type == "rgb"):
                    file_name = str(self.plugin_config["tfrecords_filename_multi_train"])
                else:
                    file_name = str(self.plugin_config["tfrecords_filename_multi_train"])

            self.images, self.labels = self.readFromTFRecords(file_name, batch_size, num_epochs,
               num_threads, min_after_dequeue)

    def readFromTFRecords(self, filename, batch_size, num_epochs, num_threads=2,
                          min_after_dequeue=1000):
        def read_and_decode(filename_queue):
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            features = tf.parse_single_example(
                serialized_example,
                features={
                    'image_height': tf.FixedLenFeature([], tf.int64),
                    'image_width': tf.FixedLenFeature([], tf.int64),
                    'channels': tf.FixedLenFeature([], tf.int64),
                    'mask_height': tf.FixedLenFeature([], tf.int64),
                    'mask_width': tf.FixedLenFeature([], tf.int64),
                    'mask_channels': tf.FixedLenFeature([], tf.int64),
                    'image_raw': tf.FixedLenFeature([], tf.string),
                    'mask_raw': tf.FixedLenFeature([], tf.string)
                })
            image = tf.decode_raw(features['image_raw'], tf.float64)
            label = tf.decode_raw(features['mask_raw'], tf.int8)
            image = tf.reshape(image, [1, self.img_shape[0] *self.img_shape[1]* self.img_shape[2]])
            label = tf.reshape(label, [1, self.label_shape[0] *self.label_shape[1]* 1])
            image = tf.cast(image, tf.float64)
            label = tf.cast(label, tf.float64)
            return image, label

        filename_queue = tf.train.string_input_producer([filename], num_epochs=num_epochs)
        image, sparse_label = read_and_decode(filename_queue)
        images, sparse_labels = tf.train.shuffle_batch(
            [image, sparse_label], batch_size=batch_size, num_threads=num_threads,
            min_after_dequeue=min_after_dequeue,
            capacity=min_after_dequeue + (num_threads + 1) * batch_size
        )
        return images, sparse_labels

    def convertToTFRecords(self, images, labels, num_examples, filename):
        rows = images.shape[1]
        cols = images.shape[2]
        depth = images.shape[3]
        writer = tf.python_io.TFRecordWriter(os.path.join('data_log', filename))
        for index in xrange(num_examples):
            image_raw = images[index].tostring()
            example = tf.train.Example(features=tf.train.Features(feature={
                'height': tf.train.Feature(int64_list=tf.train.Int64List(value=[rows])),
                'width': tf.train.Feature(int64_list=tf.train.Int64List(value=[cols])),
                'depth': tf.train.Feature(int64_list=tf.train.Int64List(value=[depth])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[labels[index]])),
                'image_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_raw]))
            }))
            writer.write(example.SerializeToString())
        writer.close()
