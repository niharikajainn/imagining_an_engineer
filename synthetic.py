#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from keras import applications
from keras.models import Model

from keras.preprocessing.image import load_img

def evaluate_model(model):
    import PIL.Image as image 
    folder = os.getcwd()+"/doILookLikeAnEngineer/"
    x = np.array([np.array(image.open(folder+fname)) for fname in os.listdir(folder)])
    model.load_weights('models/gan_augmented_30_epochs.h5')
    prediction = model.predict(x)
    profs = np.round(prediction)
    print(prediction[0][0])
    result = profs[0][0]
    if result:
        os.system("figlet Engineer!")
    else:
        os.system("figlet Not an engineer")
    return result

def main():

	# dimensions of our images.
	img_width, img_height = 64, 64

	train_data_dir = 'aug_train_data'
	validation_data_dir = 'data/validation/'

	# used to rescale the pixel values from [0, 255] to [0, 1] interval
	datagen = ImageDataGenerator(rescale=1./255)
	batch_size = 64
	val_batch_size = 2

	# automatically retrieve images and their classes for train and validation sets
	train_generator = datagen.flow_from_directory(
		train_data_dir,
		target_size=(img_width, img_height),
		batch_size=batch_size,
		class_mode='binary')

	validation_generator = datagen.flow_from_directory(
		validation_data_dir,
		target_size=(img_width, img_height),
		batch_size=val_batch_size,
		class_mode='binary')

	# a simple stack of 3 convolution layers with a ReLU activation and followed by max-pooling layers.
	model = Sequential()
	model.add(Convolution2D(32, (3, 3), input_shape=(img_width, img_height,3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(32, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Convolution2D(64, (3, 3)))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))

	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation('relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))

	model.compile(loss='binary_crossentropy',
		      optimizer='rmsprop',
		      metrics=['accuracy'])

	result = evaluate_model(model)
	return result

