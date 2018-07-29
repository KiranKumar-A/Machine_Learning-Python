# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 16:03:53 2018

@author: 1018090
"""
import tensorflow as tf
from tensorflow import keras
from keras.datasets import fashion_mnist
import gzip
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
sess = tf.InteractiveSession()
from keras.layers import Input, Convolution2D, MaxPooling2D, Flatten
from keras import optimizers
from keras.initializers import Constant
from numpy.random import seed
seed(1)

labeled_images = pd.read_csv('G:\\DATA_SCIENCE\\SENTDEX\\Akhil_work\\fashionmnist\\fashion-mnist_test.csv')
images = labeled_images.iloc[0:10000,1:]
labels = labeled_images.iloc[0:10000,:1]
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images.shape

i=6
img=train_images.iloc[i].as_matrix()
img=img.reshape((28,28))
plt.imshow(img,cmap='gray')
plt.title(train_labels.iloc[i,0])

train_images = train_images / 255.0
test_images = test_images / 255.0

plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    img=train_images.iloc[i].as_matrix()
    img=img.reshape((28,28))
    plt.imshow(img,cmap='gray')
    plt.title(train_labels.iloc[i,0])
    plt.xlabel(class_names[train_labels.iloc[i,0]])
    
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(256, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])
    
model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
model.fit(train_images.values.flatten().reshape(8000,28,28), train_labels.values.ravel(), epochs=10) 

test=train_images.values.flatten().reshape(8000,28,28)
test.shape

test_loss, test_acc = model.evaluate(test_images.values.flatten().reshape(2000,28,28), test_labels.values.ravel())

print('Test accuracy:', test_acc)

predictions = model.predict(test_images.values.flatten().reshape(2000,28,28))

predictions[0]



np.argmax(predictions[1])

test_labels.iloc[1]

plt.figure(figsize=(10,10))
for i in range(36):
    plt.subplot(6,6,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid('off')
    img=test_images.iloc[i].as_matrix()
    img=img.reshape((28,28))
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels.iloc[i]
    if predicted_label == true_label[0]:
      color = 'green'
    else:
      color = 'red'
    plt.xlabel("{} ({})".format(class_names[predicted_label], 
                                  class_names[true_label[0]]),
                                  color=color)
    
    
