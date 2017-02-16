"""
Simple tester for the small_vgg19_trainable
"""
import tensorflow as tf

import sys as sys
import time as time

import numpy as np
import skimage.io as io
import skimage.transform as skt
import os
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

from functools import reduce
import scipy.ndimage

class Vgg19:
    """
    A trainable version VGG19.
    """

    def __init__(self, vgg19_npy_path=None, trainable=True):
        if vgg19_npy_path is not None:
            self.data_dict = np.load(vgg19_npy_path, encoding='latin1').item()
        else:
            self.data_dict = None

        self.var_dict = {}
        self.trainable = trainable
        self.lcounts = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


    def build(self, rgb, index, train_mode=None):
        """
        load variable from npy to build the VGG

        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 255]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        """

        rgb_scaled = rgb * .004

        assert rgb.get_shape().as_list()[1:] == [64, 64, 3]
        red, green, blue = tf.split(3, 3, rgb_scaled)
        bgr = tf.concat(3, [blue, green, red])
        #bgr = tf.concat(3, [
        #    blue - VGG_MEAN[0],
        #    green - VGG_MEAN[1],
        #    red - VGG_MEAN[2],
        #])

        self.conv1_1, self.counts_1 = self.conv_layer(bgr, 3, 16, "conv1_1")
        self.lcounts1 = tf.reduce_sum(self.counts_1)
        self.variable_summaries(tf.to_float(tf.reduce_sum(self.lcounts1)),'lz1')
      
        self.conv1_2, self.counts_2 = self.conv_layer(self.conv1_1, 16, 16, "conv1_2")
        self.lcounts2 = tf.reduce_sum(self.counts_2)
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1,self.counts_3 = self.conv_layer(self.pool1, 16, 32, "conv2_1")
        self.lcounts3 = tf.reduce_sum(self.counts_3)
        self.conv2_2,self.counts_4 = self.conv_layer(self.conv2_1, 32, 32, "conv2_2")
        self.lcounts4 = tf.reduce_sum(self.counts_4)
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1,self.counts_5 = self.conv_layer(self.pool2, 32, 64, "conv3_1")
        self.lcounts5 = tf.reduce_sum(self.counts_5)
        self.conv3_2,self.counts_6 = self.conv_layer(self.conv3_1, 64, 64, "conv3_2")
        self.lcounts6 = tf.reduce_sum(self.counts_6)
        self.conv3_3,self.counts_7 = self.conv_layer(self.conv3_2, 64, 64, "conv3_3")
        self.lcounts7 = tf.reduce_sum(self.counts_7)

        self.conv3_4,self.counts_8 = self.conv_layer(self.conv3_3, 64, 64, "conv3_4")
        self.lcounts8 = tf.reduce_sum(self.counts_8)
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1,self.counts_9 = self.conv_layer(self.pool3, 64, 128, "conv4_1")
        self.lcounts9 = tf.reduce_sum(self.counts_9)
        self.conv4_2,self.counts_10 = self.conv_layer(self.conv4_1, 128, 128, "conv4_2")
        self.lcounts10 = tf.reduce_sum(self.counts_10)
        self.conv4_3,self.counts_11 = self.conv_layer(self.conv4_2, 128, 128, "conv4_3")
        self.lcounts11 = tf.reduce_sum(self.counts_11)
        self.conv4_4,self.counts_12 = self.conv_layer(self.conv4_3, 128, 128, "conv4_4")
        self.lcounts12 = tf.reduce_sum(self.counts_12)
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1,self.counts_13 = self.conv_layer(self.pool4, 128, 128, "conv5_1")
        self.lcounts13 = tf.reduce_sum(self.counts_13)
        self.conv5_2,self.counts_14 = self.conv_layer(self.conv5_1, 128, 128, "conv5_2")
        self.lcounts14 = tf.reduce_sum(self.counts_14)
        self.conv5_3,self.counts_15 = self.conv_layer(self.conv5_2, 128, 128, "conv5_3")
        self.lcounts15 = tf.reduce_sum(self.counts_15)
        self.conv5_4,self.counts_16 = self.conv_layer(self.conv5_3, 128, 128, "conv5_4")
        self.lcounts16 = tf.reduce_sum(self.counts_16)
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 512, 256, "fc6")  # 512 = ((64 / (2 ** 5)) ** 2) * 128
        self.relu6 = tf.nn.relu(self.fc6)
        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, 0.5), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, 0.5)
        self.zeros17 = tf.less_equal(self.relu6,tf.zeros_like(self.relu6))
        self.counts17 = tf.reduce_sum(tf.to_int32(self.zeros17))
        self.lcounts17 = tf.reduce_sum(self.counts17)

        self.fc7 = self.fc_layer(self.relu6, 256, 256, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, 0.5), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, 0.5)
        self.zeros18 = tf.less_equal(self.relu7,tf.zeros_like(self.relu7))
        self.counts18 = tf.reduce_sum(tf.to_int32(self.zeros18))
        self.lcounts18 = tf.reduce_sum(self.counts18)

        self.fc8 = self.fc_layer(self.relu7, 256 , 200, "fc8")

        self.prob = tf.nn.softmax(self.fc8, name="prob")

        self.data_dict = None
   



    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)
            
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            zeros = tf.less_equal(relu,tf.zeros_like(relu))
            counts = tf.reduce_sum(tf.to_int32(zeros))
            return relu,counts

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            self.variable_summaries(weights, name)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)

            return fc

    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0001, 0.0002)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        self.variable_summaries(filters, name)
        initial_value = tf.truncated_normal([out_channels], 0.0,0.0)
        biases = self.get_varb(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_fc_var(self, in_size, out_size, name):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0001, 0.0002)
        weights = self.get_var(initial_value, name, 0, name + "_weights")

        initial_value = tf.truncated_normal([out_size], 0.0,0.0)
        biases = self.get_varb(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        shap = initial_value.get_shape().as_list()
        #print("shape:",shap)
        if self.trainable:
            var = tf.get_variable(name=var_name, shape=shap, initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=True,seed=1717,dtype=tf.float32), trainable=True)
            #print(self, var_name, var.get_shape().as_list())
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

#        print  self, var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var


    def get_varb(self, initial_value, name, idx, var_name):
        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        shap = initial_value.get_shape().as_list()
        #print("shape:",shap)
        if self.trainable:
            var = tf.get_variable(name=var_name, initializer=initial_value, trainable=True)
            #print(self, var_name, var.get_shape().as_list())
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var

#        print  self, var_name, var.get_shape().as_list()
        assert var.get_shape() == initial_value.get_shape()

        return var


    def variable_summaries(self, var, name):
#      return
      """Attach a lot of summaries to a Tensor."""
      with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)
        tf.summary.scalar('max/' + name, tf.reduce_max(var))
        tf.summary.scalar('min/' + name, tf.reduce_min(var))
        tf.summary.histogram(name, var)


    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in self.var_dict.items():
            var_out = sess.run(var)
            if not data_dict.has_key(name):
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print("file saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in self.var_dict.values():
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count



dataset_path      = "tiny-imagenet-200/train/"

test_set_size = 5
IMAGE_HEIGHT  = 64
IMAGE_WIDTH   = 64
NUM_CHANNELS  = 3
BATCH_SIZE    = 128
CLASSES       = 200
IMAGES_PER    = 500
EPOCHS        = 100

def read_training_data():
    X_data = []
    label_data = []
    image_directories = os.listdir('tiny-imagenet-200/train/')
    for d in image_directories:
        image_filenames = os.listdir(dataset_path + d + '/images')
        for fname in image_filenames:
            X = scipy.ndimage.imread(dataset_path + d + '/images/'+fname, mode='RGB')
            label = fname.split('_')[0]
            X_data.append(X)
            label_data.append(label)
    X_data = np.stack(X_data, axis=0)
    labels_unique = np.unique(label_data)
    y_unique = range(len(labels_unique))
    label_y_map = dict(zip(labels_unique, y_unique))
    y_data = [label_y_map[label] for label in label_data]
    y_data = np.asarray(y_data)
    assert X_data.shape[1] == 64
    assert X_data.shape[2] == 64
    assert X_data.shape[3] == 3
    return X_data, y_data


def read_label_file(file):
  filepaths = []
  labels = []
  findex = []

  i = 0
  j = 0
  for dir in os.listdir(file):
      for file in os.listdir(dataset_path+dir+'/images/'):
          filepaths.append(dataset_path+dir+'/images/'+file)
          findex.append(j)
          labels.append(i)
          j += 1
      i += 1
      if i==CLASSES:
         return filepaths, labels, findex
  return filepaths, labels, findex

# reading labels and file path
train_filepaths, train_labels, train_indexes = read_label_file(dataset_path)

# convert file strings and labels into constant tensors
all_images = ops.convert_to_tensor(train_filepaths, dtype=dtypes.string)
all_labels = ops.convert_to_tensor(train_labels, dtype=dtypes.int32)
all_indexes = ops.convert_to_tensor(train_indexes, dtype=dtypes.int32)


# create a partition vector
partitions = [0] * len(train_filepaths)
partitions[:test_set_size] = [1] * test_set_size
np.random.shuffle(partitions)

# partition our data into a test and train set according to our partition vector
#train_images = tf.dynamic_partition(all_images, partitions, 1)
#train_labels  = tf.dynamic_partition(all_labels, partitions, 1)
train_images = all_images
train_labels = all_labels
train_indexes = all_indexes

images = tf.placeholder(tf.float32, [BATCH_SIZE, 64, 64, 3])
true_out = tf.placeholder(tf.float32, [BATCH_SIZE,200])


# create input queues
train_input_queue = tf.train.slice_input_producer(
                                    [train_images, train_labels, train_indexes],
                                    shuffle=True)

# process path and string tensor into an image and a label
file_content = tf.read_file(train_input_queue[0])
train_image = tf.image.decode_jpeg(file_content, channels=NUM_CHANNELS)
train_label = tf.one_hot(train_input_queue[1],200,on_value=1,off_value=0)
train_index = train_input_queue[2]

# define tensor shape
train_image.set_shape([IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS])
train_label.set_shape([200])

train_image_float = tf.to_float(train_image)
train_index_float = tf.to_float(train_index)
train_label_float = tf.to_float(train_label)

# collect batches of images before processing
train_image_batch, train_label_batch, train_index_batch = tf.train.batch(
                                    [train_image_float, train_label_float, train_index_float]
                                    ,batch_size=BATCH_SIZE
                                    )


with tf.Session() as sess:

    vgg = Vgg19()
    vgg.build(train_image_batch,train_index_batch)
    print('total vgg variables = {}'.format(vgg.get_var_count()))

    # simple training
    cross_ent = tf.nn.softmax_cross_entropy_with_logits(vgg.fc8,train_label_batch)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(vgg.fc8,1),tf.argmax(train_label_batch,1)),tf.float32))
    opt_train = tf.train.AdamOptimizer(learning_rate = 0.001)
    loss = tf.reduce_mean(cross_ent)
    opt = opt_train.minimize(loss)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter( './train/'+time.asctime(),sess.graph)
    #test_writer = tf.summary.FileWriter('./test')
    sess.run(tf.global_variables_initializer())

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    k = 0
    for j in range(EPOCHS):
       total_accuracy = 0
       total_loss = 0
       for offset in range(0, CLASSES*IMAGES_PER, BATCH_SIZE):
          summary,_ = sess.run([merged,opt])
          #sess.run(opt)
          #train_writer.add_summary(summary,k)
          k = k + 1
          acc = accuracy.eval()
          loss     = cross_ent.eval()
          print('shape of loss = {}'.format(loss.shape))
          total_accuracy += acc
          total_loss += loss

       print("epoch: {}, acc: {} , loss {}".format(j,total_accuracy/(CLASSES*IMAGES_PER), total_loss/(CLASSES*IMAGES_PER)))



    # stop our queue threads and properly close the session
    coord.request_stop()
    coord.join(threads)
    sess.close()


