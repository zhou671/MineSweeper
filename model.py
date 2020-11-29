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

        self.lifting = tf.keras.layers.Dense(64)
        self.res1 = Residual(128)
        self.res2 = Residual(256)
        self.dense = tf.keras.layers.Dense(width * height)
        

    def call(self, input, encoded_grids): 
        """
        input: a one-hot vector with depth of 10
        """
        features = self.lifting(input) + encoded_grids
        # features.shape = [batch_size, 6, 6, 64]
        features = self.res1(features)
        # features.shape = [batch_size, 6, 6, 128]
        features = tf.nn.max_pool(features, (2, 2), (2, 2), "VALID")
        # features.shape = [batch_size, 3, 3, 128]
        features = self.res2(features)
        # features.shape = [batch_size, 3, 3, 256]
        features = tf.nn.max_pool(features, (3, 3), (1, 1), "VALID")
        # features.shape = [batch_size, 1, 1, 256]
        features = tf.reshape(features, (-1, 256))
        # features.shape = [batch_size, 256]
        features = self.dense(features)
        # features.shape = [batch_size, 36]
        probs = tf.nn.softmax(featrues)

        return probs