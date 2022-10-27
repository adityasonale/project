import pandas as pd
import numpy as np
import seaborn as sns
import os

from keras.preprocessing.image import ImageDataGenerator
from keras_preprocessing.image import load_img, img_to_array
from keras.layers import Input, Conv2D, Flatten, BatchNormalization, Activation, MaxPooling2D, Dropout, Dense
from keras.models import Sequential
from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

#from matplotlib import pyplot as plt


# Displaying the images

picture_size = 48

folder_path_train = 'D:\Datasets\images\\train'
folder_path_test = 'D:\Datasets\images\\validation'

# expression = input()


# plt.figure(figsize=(12,12))
# for i in range(1, 10, 1):
#     plt.subplot(3, 3, i)

#     img = load_img(folder_path_train + expression + "\\" + os.listdir(folder_path_train + expression)[i], target_size= (picture_size, picture_size))

#     plt.imshow(img)

# plt.show()



# Making the training and validation data

batch_size = 128

total_classes = 7

datagen_train = ImageDataGenerator()
datagen_test = ImageDataGenerator()

train_set = datagen_train.flow_from_directory(folder_path_train, 
                                              target_size=(picture_size, picture_size), 
                                              color_mode= "grayscale", 
                                              batch_size=batch_size,
                                              class_mode='categorical', 
                                              shuffle=True)


test_set = datagen_test.flow_from_directory(folder_path_test, 
                                              target_size=(picture_size, picture_size), 
                                              color_mode= "grayscale", 
                                              batch_size=batch_size,
                                              class_mode='categorical', 
                                              shuffle=False)  # try true




# Model Building

model = Sequential()

# Layer 1

model.add(Conv2D(64,(3,3),padding='same', input_shape=(48,48,1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=(0.25)))


# Layer 2

model.add(Conv2D(128,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))

# Layer 3

model.add(Conv2D(512,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))

# Layer 4

model.add(Conv2D(512,(3,3),padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(rate=0.25))

# Flatten Layer 

model.add(Flatten())

# Fully Connected Layer 1

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation(activation='relu'))
model.add(Dropout(0.25))

# Fully Connected Layer 2

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))


model.add(Dense(total_classes,activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()


checkpoint = ModelCheckpoint("D:\\vs code\\.vscode\\python\DeepLearning\\models\\emotionDetection.h5", 
                              monitor='val_loss',
                              verbose=1,
                              save_best_only= True,
                              mode='max')


earlyStopping = EarlyStopping(monitor='val_loss', # Stop training when a monitored metric has stopped improving
                               min_delta=0,
                               patience=3,
                               verbose=1,
                               restore_best_weights=True)


reduceLRonPlateau = ReduceLROnPlateau(monitor='val_loss',
                                        factor=0.2,
                                        patience=3,
                                        verbose=1,
                                        min_delta=0.0001)

callbacks_list = [earlyStopping, checkpoint, reduceLRonPlateau]

epoch = 10
MODEL = model.fit(          train_set,
                            steps_per_epoch=(train_set.n)//(train_set.batch_size),
                            epochs=epoch,
                            validation_data= test_set,
                            validation_steps=(test_set.n)//(test_set.batch_size),
                            callbacks= callbacks_list)

