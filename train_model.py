from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist
from tensorflow.keras.optimizers import SGD
from keras.utils import np_utils
from tensorflow.keras import backend as K
import numpy as np
import argparse
import cv2
from architecture import Network

((trainData, trainLabels), (testData, testLabels)) = mnist.load_data()
# if we are using "channels first" ordering, then reshape the
# design matrix such that the matrix is:
# num_samples x depth x rows x columns
if K.image_data_format() == "channels_first":
    trainData = trainData.reshape((trainData.shape[0], 1, 28, 28))
    testData = testData.reshape((testData.shape[0], 1, 28, 28))
# otherwise, we are using "channels last" ordering, so the design
# matrix shape should be: num_samples x rows x columns x depth
else:
    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))
# scaling the data to the range of [0, 1]
trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0


trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

# initialiinge the optimizer and model
opt = SGD(lr=0.01)
model = Network.build(numChannels=1, imgRows=28, imgCols=28,numClasses=10)
model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])

# Training the model
model.fit(trainData, trainLabels, batch_size=128, epochs=20,verbose=1)

# Saving the model
model.save(r'trained_model.h5')