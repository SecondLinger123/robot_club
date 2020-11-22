import tensorflow as tf
import numpy as np
import cv2
import os

#import mnist handwrite data
mnist = tf.keras.datasets.mnist

#load image array
(x_train, y_train),(x_test, y_test) = mnist.load_data()

#make imgs directory
os.makedirs("imgs", exist_ok=True)

#load 100 mnist images
for i in range(1,100):
 #make empty image array
 x = np.zeros((28, 28), dtype=np.uint16)
 #write image
 cv2.imwrite("imgs/test"+str(i)+".png", x_test[i])
