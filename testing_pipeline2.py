from __future__ import division,print_function, absolute_import
import numpy as np
import tensorflow as tf 

import skimage
from skimage import io
import tflearn
from tflearn.data_utils import shuffle, to_categorical
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation()
img_aug.add_random_blur()

# Convolutional network building
network = input_data(shape=[None, 256, 256, 3],
                     data_preprocessing=img_prep, data_augmentation=img_aug)
network = conv_2d(network, 32, 3, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 16, 7, activation='relu')
network = max_pool_2d(network, 2)
network = conv_2d(network, 8, 18, activation='relu')
network = max_pool_2d(network, 2)
network = fully_connected(network, 256, activation='relu')
network = dropout(network, 0.2)
network = fully_connected(network, 4, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)

# Train using classifier
model = tflearn.DNN(network, tensorboard_dir = "logs/" , tensorboard_verbose=1)
model.load("models/run4.tflearn")


DATAPATH = "/Users/surajnair/Downloads/wfgO4XffLJBKS4ZlxQorLE05Xo0MmwjV/"
DATAPATH3 = "/Users/surajnair/Downloads/"

with open(DATAPATH + "test.txt") as f:
    dt = f.readlines()
for fl in dt:
    try:
        fl_sp = fl.split()
        # print fl_sp
        name = fl_sp[0]
        patient = int(fl_sp[1][1:-1])
        node = int(fl_sp[2][1:-1])
        if patient >= 100 and ((np.mean(io.imread(DATAPATH3 + name)) < 255) and (np.mean(io.imread(DATAPATH3 + name)) > 0)):
            # print(np.array([io.imread(DATAPATH3 + name)]))
            y = model.predict(np.array([io.imread(DATAPATH3 + name)]).astype("float"))
            f = open('predictions/run5raw.csv', 'a')
            f.write(str(patient)+"," + str(node) + "," + str(np.argmax(y[0])) + "\n")
            
    except IndexError:
        continue
