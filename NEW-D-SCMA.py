import tensorflow as tf
import random
import numpy as np
import time
from pylab import *
# import matplotlib.pyplot as plt
import shutil
import os

#network1
model_1_r2 = tf.keras.Sequential()
model_1_r2.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(400,4),
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model_1_r2.add(tf.keras.layers.Dense(32, activation='relu',
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model_1_r2.add(tf.keras.layers.Dense(32, activation='relu',
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model_1_r2.add(tf.keras.layers.Dense(32, activation='relu',
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model_1_r2.add(tf.keras.layers.Dense(32, activation='relu',
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model_1_r2.add(tf.keras.layers.Dense(2, activation='relu',
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))


model_1_r3 = tf.keras.Sequential()
model_1_r3.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(400,4),
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model_1_r3.add(tf.keras.layers.Dense(32, activation='relu',
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model_1_r3.add(tf.keras.layers.Dense(32, activation='relu',
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model_1_r3.add(tf.keras.layers.Dense(32, activation='relu',
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model_1_r3.add(tf.keras.layers.Dense(32, activation='relu',
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model_1_r3.add(tf.keras.layers.Dense(2, activation='relu',
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))

model_1_r6 = tf.keras.Sequential()
model_1_r6.add(tf.keras.layers.Dense(32, activation='relu', input_shape=(400,4),
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model_1_r6.add(tf.keras.layers.Dense(32, activation='relu',
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model_1_r6.add(tf.keras.layers.Dense(32, activation='relu',
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model_1_r6.add(tf.keras.layers.Dense(32, activation='relu',
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model_1_r6.add(tf.keras.layers.Dense(32, activation='relu',
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))
model_1_r6.add(tf.keras.layers.Dense(2, activation='relu',
                                    kernel_initializer='glorot_uniform',bias_initializer='zeros'))

#network1 end
