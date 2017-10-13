
import os
import pdb
import sys
import argparse

import numpy as np
import tensorflow as tf

from cnn import CNN
from QueueLoader import Queue_loader

from skimage import data, io


def train(args):

    queue_loader = Queue_loader(batch_size=3, num_epochs=args.ep)
    # model = CNN(args.lr, args.b_size, queue_loader.num_batches)
    # model.build(queue_loader.images)
    # model.loss(queue_loader.labels)
    # train_op = model.train()

    # saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))

    print ('Start training')
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        ep = 0
        step = 1
        print "----->>>"
        while not coord.should_stop():
            print "-----<<<>>>"
            sd, fg = sess.run([queue_loader.images, queue_loader.labels])

            print("-----value of sd ")
            print(fg.shape)
            # io.imshow(sd)
            # io.show()

            print "-----"
            # loss, _ = sess.run([model.loss_op, train_op])
            # if step % 10 == 0:
            #     print ('epoch: %2d, step: %2d, loss: %.4f' % (ep+1, step, loss))
            #
            # if step % queue_loader.num_batches == 0:
            #     print ('epoch: %2d, step: %2d, loss: %.4f, epoch %2d done.' % (ep+1, step, loss, ep+1))
            #     checkpoint_path = os.path.join('data_log', 'cifar.ckpt')
            #     saver.save(sess, checkpoint_path, global_step=ep+1)
            #     step = 1
            #     ep += 1
            # else:
            #     step += 1
    except tf.errors.OutOfRangeError:
        print ('\nDone training, epoch limit: %d reached.' % (args.ep))
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()
    print ('Done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='train.')
    parser.add_argument('--lr', metavar='', type=float, default=1e-3, help='learning rate.')
    parser.add_argument('--ep', metavar='', type=int, default=1, help='number of epochs.')
    parser.add_argument('--b_size', metavar='', type=int, default=2, help='batch size.')
    args, unparsed = parser.parse_known_args()
    if len(unparsed) != 0: sys.exit('Unknown argument: {}'.format(unparsed))
    if args.train:
        train(args)
    if not args.train:
        parser.print_help()
