import numpy as np
import random
from osgeo import gdal
import cv2
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('zooming', 4.0, 'Amount to rescale original image.')
flags.DEFINE_float('learning_rate', 0.0002, 'Initial learning rate for ADAM solver.')
flags.DEFINE_integer('ws', 28, 'Window size for image clip.')
flags.DEFINE_integer('filter_size', 9, 'Filter size for convolution layer.')
flags.DEFINE_integer('filters', 32, 'Number of filters in convolution layer.')
flags.DEFINE_integer('batch_size', 64, 'Mini-batch size.')
flags.DEFINE_integer('max_runup_steps', 100001, 'Maximum number of steps with deconvolution tied to convolution.')
flags.DEFINE_integer('max_layer_steps', 100001, 'Maximum number of steps per layer.')
flags.DEFINE_integer('total_layers', 5, 'Number of perturbative layers in generative network.')
flags.DEFINE_integer('gpu',0,'GPU index.')
flags.DEFINE_bool('save',True,'Save output network')
flags.DEFINE_integer('numberOfBands',3,'Number of bands in training image.')
flags.DEFINE_float('delta',2.0,' Minimum dB improvement for new layer measured in PSNR.')
flags.DEFINE_float('min_alpha',0.0001,'Minimum value of perturbative layer parameter.')
flags.DEFINE_integer('layers_trained',0,'Number of layers previously trained.')
flags.DEFINE_string('training_image','3band_AOI_1_RIO_img1.tif','Training image.')
flags.DEFINE_integer('precision',8,'Number of bits per bands.')
flags.DEFINE_integer('convolutions_per_layer',2,'Number of convolutional layers in each perturbative layer.')


cpu = "/cpu:0"
gpu = "/gpu:"+str(FLAGS.gpu)
prefix = str(FLAGS.numberOfBands)+'-band'
numberOfBands = FLAGS.numberOfBands
maxp = 2.0**(1.0*FLAGS.precision) - 1.0


# Define the path for saving the neural network
summary_name = prefix + "." +str(FLAGS.total_layers) +"."
layer_name = './checkpoints/'+prefix+'/'+summary_name



# open training data and compute initial cost function averaged over the entire image
ds = gdal.Open(FLAGS.training_image)
im_raw = np.swapaxes(np.swapaxes(ds.ReadAsArray(),0,1), 1,2)
im_small = cv2.resize(im_raw, (int(im_raw.shape[1]/FLAGS.zooming), int(im_raw.shape[0]/FLAGS.zooming)))
im_blur_cubic = cv2.resize(im_small, (im_raw.shape[1], im_raw.shape[0]), interpolation = cv2.INTER_CUBIC)
im_blur = cv2.resize(im_small, (im_raw.shape[1], im_raw.shape[0]))
initial_MSE = np.sum( np.square( (im_raw[:,:,0:numberOfBands]/maxp - im_blur[:,:,0:numberOfBands]/maxp) ) ) / (im_raw.shape[0] * im_raw.shape[1] * numberOfBands * 1.0 )
initial_PSNR = -10.0*np.log(initial_MSE)/np.log(10.0)






# Define CosmiQNet
# since the number of layers is determined at runtime, we initialize variables as arrays
sr = range(FLAGS.total_layers)
sr_cost = range(FLAGS.total_layers)
optimizer_layer = range(FLAGS.total_layers)
optimizer_all = range(FLAGS.total_layers)
W = range(FLAGS.total_layers)
Wo = range(FLAGS.total_layers)
b = range(FLAGS.total_layers)
bo = range(FLAGS.total_layers)
conv = range(FLAGS.total_layers)
inlayer = range(FLAGS.total_layers)
outlayer = range(FLAGS.total_layers)
alpha = range(FLAGS.total_layers)
beta = range(FLAGS.total_layers)
for i in range(FLAGS.total_layers):
    W[i] = range(FLAGS.convolutions_per_layer)
    b[i] = range(FLAGS.convolutions_per_layer)
    conv[i] = range(FLAGS.convolutions_per_layer)
deconv = range(FLAGS.total_layers)
MSE_sr = range(FLAGS.total_layers)
PSNR_sr = range(FLAGS.total_layers)

# todo fix this
layers = 4
scale = 2

#Generator
with tf.device(gpu):
    x8 = tf.placeholder(tf.float32, shape=[None, FLAGS.ws, FLAGS.ws, 8]) # 8-band input
    x3 = tf.placeholder(tf.float32, shape=[None, scale * FLAGS.ws, scale * FLAGS.ws, 3]) # 3-band ipnput
    label_distance = tf.placeholder(tf.float32, shape=[None, FLAGS.ws, FLAGS.ws, 1]) # distance transform as a label

    for i in range(layers):
        alpha[i] = tf.Variable(0.9, name='alpha_' + str(i))
        beta[i] = tf.maximum( 0.0 , tf.minimum ( 1.0 , alpha[i] ), name='beta_'+str(i))
        bi[i] = tf.Variable(tf.constant(0.0,shape=[FLAGS.filters]), name='bi_'+str(i))
        bo[i] = tf.Variable(tf.constant(0.0,shape=[FLAGS.filters]), name='bo_'+str(i))
        Wo[i] = tf.Variable(tf.truncated_normal([FLAGS.filter_size,FLAGS.filter_size,1,FLAGS.filters], stddev=0.1), name='Wo_'+str(i))  #
        if 0 == i:
            # First layer project 11 bands onto one distance transform band
            Wi3 = tf.Variable(tf.truncated_normal([FLAGS.filter_size,FLAGS.filter_size,3,FLAGS.filters], stddev=0.1), name='Wi_'+str(i)+'l3')
            Wi8 = tf.Variable(tf.truncated_normal([FLAGS.filter_size,FLAGS.filter_size,8,FLAGS.filters], stddev=0.1), name='Wi_'+str(i)+'l8')
            z3 = tf.nn.conv2d( x3, Wi3, strides=[1,scale,scale,1], padding='SAME')
            z8 = tf.nn.conv2d( x8, Wi8, strides=[1,1,1,1], padding='SAME')
            z[i] = tf.nn.bias_add(tf.nn.relu(tf.nn.bias_add(tf.add(z3, z8), bi[i], name='conv_'+str(i))), bo[i])
            vars_Wb = [Wi3,Wi8,Wo[i],bi[i],bo[i]]
        else:
            # non-initial bands are perturbations of previous bands output
            inlayer[i] = outlayer[i-1]
            Wi[i] = tf.Variable(tf.truncated_normal([FLAGS.filter_size,FLAGS.filter_size,1,FLAGS.filters], stddev=0.1), name='Wi_'+str(i))
            z[i] = tf.nn.bias_add(tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d( inlayer[i], Wi[i], strides=[1,1,1,1], padding='SAME'), bi[i], name='conv_'+str(i))), bo[i])
            vars_Wb = [Wi[i],Wo[i],bi[i],bo[i], alpha[i]]

        labelout[i] = tf.nn.conv2d_transpose( z[i], Wo[i], [FLAGS.batch_size,FLAGS.ws,FLAGS.ws,1] ,strides=[1,1,1,1], padding='SAME')
        if 0 == i:
            outlayer[i] = labelout[i]
        else :
            # convex combination measures impact of layer
            outlayer[i] = tf.nn.relu( tf.add(  tf.scalar_mul( beta[i] , labelout[i]), tf.scalar_mul(1.0-beta[i], inlayer[i])))

        label_cost[i] = tf.reduce_sum ( tf.pow( tf.sub(outlayer[i],label_distance),2))
        label_optimizer[i] = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(label_cost[i], var_list=vars_Wb)
        full_label_optimizer[i] = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(label_cost[i])