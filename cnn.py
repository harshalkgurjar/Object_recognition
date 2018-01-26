# P-22 : hkgurjar - manush - ppfirake - sndesai
#code run congfig:  ARC cluster with Titanx

#data import

import os
import os.path
import tensorflow as tf
from PIL import Image
from PIL import ImageFilter
import time
import tensorflow as tf
import time
import numpy as np
from scipy import ndimage
from scipy import misc
import random
import glob
images_path = 'data/'
output_num = 9

# data read functions
# Read multiple categories images
def read_multiple_categories_images(data_indices, object_categories, size):
    input_data = []
    for index, category in zip(data_indices, object_categories):
        data_list = read_single_category_image(index[0], index[1], category, size)
        for data in data_list:
            input_data.append(data)
    return input_data

def read_single_category_image(first_index, last_index, object_categories, size=64, g=True):
    my_images_path = images_path + object_categories + "/"
    extension = "*.jpg"
    path = my_images_path
    directory = os.path.join(path, extension)
    file_list = glob.glob(directory)
    output_images = []
    for file in file_list[first_index - 1:last_index]:
        read_image = misc.imread(file, flatten=True)
        read_image = misc.imresize(read_image, [size, size])
        read_image = np.asarray(read_image)
        temporary_image = np.reshape(read_image, size * size)
        output_images.append(temporary_image)
        temporary_image = ndimage.gaussian_filter(read_image, sigma=3)
        temporary_image = np.reshape(temporary_image, size * size)
        output_images.append(temporary_image)
        read_image = ndimage.rotate(read_image, 90)
        temporary_image = np.reshape(read_image, size * size)
        output_images.append(temporary_image)
        temporary_image = ndimage.gaussian_filter(read_image, sigma=3)
        temporary_image = np.reshape(temporary_image, size * size)
        output_images.append(temporary_image)
        read_image = ndimage.rotate(read_image, 90)
        temporary_image = np.reshape(read_image, size * size)
        output_images.append(temporary_image)
        temporary_image = ndimage.gaussian_filter(read_image, sigma=3)
        temporary_image = np.reshape(temporary_image, size * size)
        output_images.append(temporary_image)
        read_image = ndimage.rotate(read_image, 90)
        temporary_image = np.reshape(read_image, size * size)
        output_images.append(temporary_image)
        temporary_image = ndimage.gaussian_filter(read_image, sigma=3)
        temporary_image = np.reshape(temporary_image, size * size)
        output_images.append(temporary_image)
    output_images = np.asarray(output_images)
    return output_images

# cnn code
# defining data size

tf.flags.DEFINE_integer("batch_size", 64, "batch size for training")
tf.flags.DEFINE_integer("eval_batch_size", 8, "batch size for evaluation")

image_categories = [d for d in os.listdir(images_path) if os.path.isdir(os.path.join(images_path, d))]
list_of_training_data_indices=[ ]
list_of_testing_data_indices=[ ]

for category_name in image_categories:
    list = os.listdir(images_path + category_name + "/")
    number_of_files = len(list)
    list_of_training_data_indices.append([1, int(number_of_files * 0.7)])
    list_of_testing_data_indices.append([int(number_of_files * 0.7) + 1, number_of_files])

print(list_of_training_data_indices)
print(list_of_testing_data_indices)

#cnn parameters
SIZE = 28
ALL_TRAINING_DATA = 0
BATCH = 80
DATA_AUGMENT = 8
for train in list_of_training_data_indices:
    ALL_TRAINING_DATA = ALL_TRAINING_DATA + (train[1] - train[0] + 1)
TRAIN_INDICES = range(DATA_AUGMENT * ALL_TRAINING_DATA)

# ---------------------- read data and input to cnn --------------------------------#

training_data = read_multiple_categories_images(list_of_training_data_indices, image_categories, SIZE)
temp_index = 0
training_label = []
for train in list_of_training_data_indices:
    d = train[1] - train[0] + 1
    x=[]
    for category_name in image_categories:
        x.append(0)
    x[temp_index] = 1
    for test in range(DATA_AUGMENT * d):
        training_label.append(x)
    temp_index = temp_index + 1
training_data = np.asarray(training_data)
training_label = np.asarray(training_label)
testing_data = read_multiple_categories_images(list_of_testing_data_indices, image_categories, SIZE)
print("Training images from data = ", len(training_data))
testing_label = []
temp_index = 0
for train in list_of_testing_data_indices:
    d = train[1] - train[0] + 1
    x=[]
    for category_name in image_categories:
        x.append(0)
    x[temp_index] = 1
    for test in range(DATA_AUGMENT * d):
        testing_label.append(x)
    temp_index = temp_index + 1

testing_data = np.asarray(testing_data)
testing_label = np.asarray(testing_label)
print("Testing images from data = ", len(testing_data))
#define tensorflow configurations for gpu computations
config = tf.ConfigProto(log_device_placement=False)
config.gpu_options.allow_growth = True
sess = tf.InteractiveSession(config=config)
x = tf.placeholder(tf.float32, shape=[None, SIZE * SIZE])
y = tf.placeholder(tf.float32, shape=[None, output_num])
testing_accuracy = []
training_accuracy = []
#define weights for cnn
def weight(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
#read data and input to cnn
def limit_training(list_of_training):
    if len(list_of_training) < 3:
        return True
    elif (list_of_training[-1] > list_of_training[-2]) or (list_of_training[-1] > list_of_training[-3]):
        return True
    else:
        return False
def convolutional_2d(x, W, s=1):  # s -> strides
    return tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')

#pooling layer
def max_pool(x, size=2, stride=2):
    return tf.nn.max_pool(x, ksize=[1, size, size, 1], strides=[1, stride, stride, 1], padding='SAME')

#Convolutional layer : 1
weight_convolutional_1 = weight([5, 5, 1, 64])
b_convolutional_1 = bias([64])
x_image = tf.reshape(x, [-1, SIZE, SIZE, 1])
h_convolutional_1 = tf.nn.relu(convolutional_2d(x_image, weight_convolutional_1) + b_convolutional_1)
h_pool1 = max_pool(h_convolutional_1)

#Convolutional layer : 2
weight_convolutional_2 = weight([5, 5, 64, 128])  
b_convolutional_2 = bias([128])
h_convolutional_2 = tf.nn.relu(convolutional_2d(h_pool1, weight_convolutional_2) + b_convolutional_2)
h_pool2 = max_pool(h_convolutional_2)

#Layer : 1d
weight_fc_1 = weight([7 * 7 * 128, 1024])
b_fc_1 = bias([1024])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 128])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, weight_fc_1) + b_fc_1)

#Layer : 2d
weight_fc_2 = weight([1024, 1024])
b_fc_2 = bias([1024])
h_fc1_transpose = tf.reshape(h_fc1, [-1, 1024])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_transpose, weight_fc_2) + b_fc_2)

#dropout layer
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc2, keep_prob)

#Layer : Output
weight_fc_2 = weight([1024, output_num])  # 4 output class
b_fc_2 = bias([output_num])
y_conv = tf.matmul(h_fc1_drop, weight_fc_2) + b_fc_2

#train
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y))
training_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)  # changed from 4 to 5
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())

for train in range(10000):
    if train % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: training_data, y: training_label, keep_prob: 1.0})
        print("step " + str(train))
        print("Training accuracy for this step : " + str(train_accuracy))
        print(" -- ")
        training_accuracy.append(train_accuracy)
        test_accuracy = accuracy.eval(feed_dict={x: testing_data, y: testing_label, keep_prob: 1.0})
        testing_accuracy.append(test_accuracy)
        print("step " + str(train))
        print("Testing accuracy for this step : " + str(test_accuracy))
        print(" -- ")
        if not limit_training(testing_accuracy):
            break
    random.shuffle(TRAIN_INDICES)
    for test in range(0, len(TRAIN_INDICES), BATCH):
        training_step.run(feed_dict={x: training_data[test:test + BATCH - 1], y: training_label[test:test + BATCH - 1], keep_prob: 0.5})

#calculating accuracy : test

for train, test in zip(training_accuracy, testing_accuracy):
    print("Training accuracy for this step : " + train)
    print("Testing accuracy for this step : " + test)
print("Maximum Testing accuracy", max(testing_accuracy))
print("Minimum Testing accuracy", min(testing_accuracy))

