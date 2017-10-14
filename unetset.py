import matplotlib.pyplot as plt
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

from QueueLoader import QueueLoader
from tf_unet.tf_unet.image_util import GISDataProvider
plt.rcParams['image.cmap'] = 'gist_earth'
from tf_unet.tf_unet import unet
from collections import namedtuple


plugin_config = "/home/geesara/project/satelliteimageclassifier/config/config.json"
type_of_data="multi"

generator_train = GISDataProvider(plugin_config, type, train=True)
generator_test = GISDataProvider(plugin_config, type, train=False)
batch_size_for_net=2
epochs=20
net = unet.Unet(channels=generator_train.channels, n_class=generator_train.classes, layers=3,
                features_root=16)
trainer = unet.Trainer(net, optimizer="momentum", opt_kwargs=dict(momentum=0.2), batch_size=batch_size_for_net,
                       verification_batch_size=1)
queue_loader_train = QueueLoader(plugin_config, type=type_of_data, batch_size=batch_size_for_net, num_epochs=epochs,
                                 train=True)
queue_loader_validate = QueueLoader(plugin_config, type=type_of_data, batch_size=batch_size_for_net,
                                num_epochs=epochs, train=False)
place_holder = namedtuple('modelTrain', 'train_dataset test_dataset loader_train loader_test')
operators = place_holder(train_dataset=generator_train, test_dataset=generator_test, loader_train=queue_loader_train,
                 loader_test=queue_loader_validate)
path = trainer.train(operators, "./unet_trained", training_iters=4, epochs=epochs, display_step=2, restore=True)

print("")

