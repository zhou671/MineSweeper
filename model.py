import tensorflow as tf
import numpy as np

class Residual(tf.keras.Model):
    def __init__(self, out_size):
        self.conv1 = tf.keras.layers.Conv2D(out_size, (3, 3), 1, "same")
        self.norm1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(out_size, (3, 3), 1, "same")
        self.norm2 = tf.keras.layers.BatchNormalization()

    def call(self, x):
        fx = tf.nn.relu(self.norm1(self.conv1(x)))
        fx = self.norm2(self.conv2(fx))

        return tf.nn.relu(fx + x)

class Model(tf.keras.Model):
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.batch_size = 256

        self.optimizer = tf.keras.optimizers.Adam()

        self.lifting = tf.keras.layers.Dense(32)
        self.res1 = Residual(64)
        self.res2 = Residual(128)
        self.dense = tf.keras.layers.Dense(width * height)

    def call(self, ):
