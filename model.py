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
    folder = os.getcwd()+"/data/niharika_test/male/"
    x = np.array([np.array(image.open(folder+fname)) for fname in os.listdir(folder)])
    model.load_weights('models/augmented_30_epochs.h5')

    print(np.ones(x.shape[0]))
    num_profs = x.shape[0]
    profs = np.count_nonzero(np.trunc(model.predict(x)))
    print("Accuracy is {}".format(profs/num_profs))
    exit()


# dimensions of our images.
img_width, img_height = 64, 64

train_data_dir = 'data/train'
validation_data_dir = 'data/test'

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

#evaluate_model(model)

epochs = 25
train_samples = 12000+12000
validation_samples = 1000+1000

model.fit_generator(
        train_generator,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples// val_batch_size,)
#About 60 seconds an epoch when using CPU

model.save_weights('models/basic_cnn_30_epochs.h5')
model.evaluate_generator(validation_generator, validation_samples)


# By applying random transformation to our train set, we artificially enhance our dataset with new unseen images.
# This will hopefully reduce overfitting and allows better generalization capability for our network.

train_datagen_augmented = ImageDataGenerator(
        rescale=1./255,        # normalize pixel values to [0,1]
        shear_range=0.2,       # randomly applies shearing transformation
        zoom_range=0.2,        # randomly applies shearing transformation
        horizontal_flip=True)  # randomly flip the images

# same code as before
train_generator_augmented = train_datagen_augmented.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

model.fit_generator(
        train_generator_augmented,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_samples // val_batch_size,)


model.save_weights('models/augmented_30_epochs.h5')

#### Evaluating on validation set
score = model.evaluate_generator(validation_generator, validation_samples)
print("Accuracy = {}".format(score))
