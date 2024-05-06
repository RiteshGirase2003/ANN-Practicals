# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1hrvvNHeMPmWUcN6NFSrrGCYk1NtvrVhn
"""

import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

(X_train,y_train), (X_test,y_test) = datasets.cifar10.load_data()
X_train.shape

y_train = y_train.reshape(-1,)

classes = ['airplane', 'automobile','bird', 'cat', 'deer','dog','frog','horse','ship','truck']

def plot_sample(X,y,index):
  plt.figure(figsize = (15,2))
  plt.imshow(X[index])
  plt.xlabel(classes[y[index]])

plot_sample(X_train,y_train,1)

X_train = X_train/255 # to normalize and make it from 0 to 1
X_test = X_test /255

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3,3), activation ='relu', input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),


    layers.Conv2D(filters=64, kernel_size=(3,3), activation ='relu'),
    layers.MaxPooling2D((2,2)),


    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(64,activation = 'relu'),
    # layers.Dense(1000,activation = 'relu'),
    layers.Dense(10,activation = 'softmax')

])

cnn.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs = 10)

cnn.evaluate(X_test,y_test)

y_test = y_test.reshape(-1,)

y_pred = cnn.predict(X_test)

y_classes = [np.argmax(element) for element in y_pred]

# print('Classification Report : \n',classification_report(y_test,y_classes))

plot_sample(X_test,y_test,5)
classes[y_classes[5]]
