from __future__ import division, print_function, absolute_import
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

DATAPATH = "/Users/surajnair/Downloads/8Z455Ls09xG9w3esy18qnFOGrCzu0umj/"
DATAPATH2 = "/Users/surajnair/Downloads/LeySk8h49gbExZSvxpQXitwfWkYbgzA8/"
DATAPATH3 = "/Users/surajnair/Downloads/"


Xtrain, Ytrain, Xtest, Ytest = [], [], [], []

with open(DATAPATH + "train.txt") as f:
    dt = f.readlines()
for fl in dt:
    fl_sp = fl.split()
    name = fl_sp[0]
    Xtrain.append(io.imread(DATAPATH3 + name))
    y = np.zeros((4,))
    y[int(fl_sp[1])] = 1
    Ytrain.append(y)

currlen = len(Xtrain)
    
with open(DATAPATH2 + "train.txt") as f:
    dt = f.readlines()
for fl in dt:
    fl_sp = fl.split()
    name = fl_sp[0]
    Xtrain.append(io.imread(DATAPATH3+ name))
    y = np.zeros((4,))
    if float(fl_sp[1]) == 0:
        y[0] = 1
    else:
        y[int(fl_sp[1])] = 1
    Ytrain.append(y)
    if len(Xtrain) >= 2*currlen:
        break
    
with open(DATAPATH + "test.txt") as f:
    dt = f.readlines()
for fl in dt:
    fl_sp = fl.split()
    name = fl_sp[0]
    Xtest.append(io.imread(DATAPATH3+ name))
    y = np.zeros((4,))
    y[int(fl_sp[1])] = 1
    Ytest.append(y)
    
with open(DATAPATH2 + "test.txt") as f:
    dt = f.readlines()
for fl in dt:
    fl_sp = fl.split()
    name = fl_sp[0]
    Xtest.append(io.imread(DATAPATH3 + name))
    y = np.zeros((4,))
    if float(fl_sp[1]) == 0:
        y[0] = 1
    else:
        y[int(fl_sp[1])] = 1
    Ytest.append(y)
#     break


Xtrain = np.array(Xtrain).astype("float") 
Ytrain = np.array(Ytrain).astype("float")
Xtest = np.array(Xtest).astype("float")
Ytest = np.array(Ytest).astype("float")
print(Xtrain.shape)
print(Xtest.shape)
print(Ytrain.shape)
print(Ytest.shape)


Xtrain, Ytrain = shuffle(Xtrain, Ytrain)
Xtest, Ytest = shuffle(Xtest, Ytest)

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
model.fit(Xtrain, Ytrain, n_epoch=10, shuffle=True, validation_set=(Xtest, Ytest),
          show_metric=True, batch_size=96)

model.save("models/run4.tflearn")
