import tensorflow as tf
from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle

import numpy as np
from sklearn.model_selection import train_test_split
import pickle

dataset_path      = "tiny-imagenet-200/train/"

BATCH_SIZE    = 128
EPOCHS        = 100

with open('train.p', 'rb') as f:
    data = pickle.load(f)
print('features shape = {}'.format(data['features'].shape))
print('labels shape = {}'.format(data['labels'].shape))


features_all = data['features']
labels_all = data['labels']

CLASSES = len(np.unique(labels_all))
print("Total classes = {}".format(CLASSES))

features_all, labels_all = shuffle(features_all, labels_all)

a = -0.5
b = 0.5

min_f = np.min(features_all)
max_f = np.max(features_all)

features_all = a + (b-a) * (features_all - min_f) / (max_f - min_f)

features_all, labels_all = features_all[:10000], labels_all[:10000]

X_train, X_val, y_train, y_val = train_test_split(features_all, labels_all, test_size=0.33, random_state=0)

features = tf.placeholder(tf.float32, (None, 64, 64, 3))
labels = tf.placeholder(tf.int64, None)

one_hot_labels = tf.one_hot(labels, CLASSES)


conv1_1W = tf.Variable(tf.truncated_normal(shape=(3, 3, 3, 64), stddev = 1e-2))
conv1_1b = tf.Variable(tf.zeros(64))
conv1_1   = tf.nn.conv2d(features, conv1_1W, strides=[1, 1, 1, 1], padding='SAME') + conv1_1b
conv1_1 = tf.nn.relu(conv1_1)

conv1_2W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 64), stddev = 1e-2))
conv1_2b = tf.Variable(tf.zeros(64))
conv1_2   = tf.nn.conv2d(conv1_1, conv1_2W, strides=[1, 1, 1, 1], padding='SAME') + conv1_2b
conv1_2 = tf.nn.relu(conv1_2)

pool1 = tf.nn.max_pool(conv1_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

conv2_1W = tf.Variable(tf.truncated_normal(shape=(3, 3, 64, 128), stddev = 1e-2))
conv2_1b = tf.Variable(tf.zeros(128))
conv2_1   = tf.nn.conv2d(pool1, conv2_1W, strides=[1, 1, 1, 1], padding='SAME') + conv2_1b
conv2_1 = tf.nn.relu(conv2_1)

conv2_2W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 128), stddev = 1e-2))
conv2_2b = tf.Variable(tf.zeros(128))
conv2_2   = tf.nn.conv2d(conv2_1, conv2_2W, strides=[1, 1, 1, 1], padding='SAME') + conv2_2b
conv2_2 = tf.nn.relu(conv2_2)

pool2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


##
conv3_1W = tf.Variable(tf.truncated_normal(shape=(3, 3, 128, 256), stddev = 1e-2))
conv3_1b = tf.Variable(tf.zeros(256))
conv3_1   = tf.nn.conv2d(pool2, conv3_1W, strides=[1, 1, 1, 1], padding='SAME') + conv3_1b
conv3_1 = tf.nn.relu(conv3_1)

conv3_2W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 256), stddev = 1e-2))
conv3_2b = tf.Variable(tf.zeros(256))
conv3_2   = tf.nn.conv2d(conv3_1, conv3_2W, strides=[1, 1, 1, 1], padding='SAME') + conv3_2b
conv3_2 = tf.nn.relu(conv3_2)

conv3_3W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 256), stddev = 1e-2))
conv3_3b = tf.Variable(tf.zeros(256))
conv3_3   = tf.nn.conv2d(conv3_2, conv3_3W, strides=[1, 1, 1, 1], padding='SAME') + conv3_3b
conv3_3 = tf.nn.relu(conv3_3)

pool3 = tf.nn.max_pool(conv3_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')



conv4_1W = tf.Variable(tf.truncated_normal(shape=(3, 3, 256, 512), stddev = 1e-2))
conv4_1b = tf.Variable(tf.zeros(512))
conv4_1   = tf.nn.conv2d(pool3, conv4_1W, strides=[1, 1, 1, 1], padding='SAME') + conv4_1b
conv4_1 = tf.nn.relu(conv4_1)

conv4_2W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), stddev = 1e-2))
conv4_2b = tf.Variable(tf.zeros(512))
conv4_2   = tf.nn.conv2d(conv4_1, conv4_2W, strides=[1, 1, 1, 1], padding='SAME') + conv4_2b
conv4_2 = tf.nn.relu(conv4_2)

conv4_3W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), stddev = 1e-2))
conv4_3b = tf.Variable(tf.zeros(512))
conv4_3   = tf.nn.conv2d(conv4_2, conv4_3W, strides=[1, 1, 1, 1], padding='SAME') + conv4_3b
conv4_3 = tf.nn.relu(conv4_3)

pool4 = tf.nn.max_pool(conv4_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')


conv5_1W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), stddev = 1e-2))
conv5_1b = tf.Variable(tf.zeros(512))
conv5_1   = tf.nn.conv2d(pool4, conv5_1W, strides=[1, 1, 1, 1], padding='SAME') + conv5_1b
conv5_1 = tf.nn.relu(conv5_1)

conv5_2W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), stddev = 1e-2))
conv5_2b = tf.Variable(tf.zeros(512))
conv5_2   = tf.nn.conv2d(conv5_1, conv5_2W, strides=[1, 1, 1, 1], padding='SAME') + conv5_2b
conv5_2 = tf.nn.relu(conv5_2)

conv5_3W = tf.Variable(tf.truncated_normal(shape=(3, 3, 512, 512), stddev = 1e-2))
conv5_3b = tf.Variable(tf.zeros(512))
conv5_3   = tf.nn.conv2d(conv5_2, conv5_3W, strides=[1, 1, 1, 1], padding='SAME') + conv5_3b
conv5_3 = tf.nn.relu(conv5_3)

pool5 = tf.nn.max_pool(conv5_3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
#
fc0   = flatten(pool5)
    
fc6_W = tf.Variable(tf.truncated_normal(shape=(2048, 4096), stddev = 1e-2))
fc6_b = tf.Variable(tf.zeros(4096))
fc6   = tf.matmul(fc0, fc6_W) + fc6_b

fc6    = tf.nn.relu(fc6)

fc7_W = tf.Variable(tf.truncated_normal(shape=(4096, 4096), stddev = 1e-2))
fc7_b = tf.Variable(tf.zeros(4096))
fc7   = tf.matmul(fc6, fc7_W) + fc7_b

fc7    = tf.nn.relu(fc7)

fc8_W = tf.Variable(tf.truncated_normal(shape=(4096, CLASSES), stddev = 1e-2))
fc8_b = tf.Variable(tf.zeros(CLASSES))
fc8   = tf.matmul(fc7, fc8_W) + fc8_b

logits    = tf.nn.relu(fc8)


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits, one_hot_labels)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)
training_operation = optimizer.minimize(loss_operation)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_labels, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()


def evaluate(X_data, y_data):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss = 0
    sess = tf.get_default_session()
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy = sess.run(accuracy_operation, feed_dict={features: batch_x, labels: batch_y})
        l = sess.run(loss_operation, feed_dict={features: batch_x, labels: batch_y})
        total_accuracy += (accuracy * len(batch_x))
        total_loss += l
    return total_accuracy / num_examples, total_loss / num_examples

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    num_examples = len(X_train)
    
    print("Training... size = {}".format(num_examples))
    print()
    for i in range(EPOCHS):
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, BATCH_SIZE):
            end = offset + BATCH_SIZE
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation, feed_dict={features: batch_x, labels: batch_y})
            
        validation_accuracy, loss = evaluate(X_val, y_val)
        print("EPOCH {} ...".format(i+1))
        print("Validation Accuracy = {:.3f}, loss = {}".format(validation_accuracy, loss))
        print()
        
    saver.save(sess, 'vgg-net')
    print("Model vgg saved")    

