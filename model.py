import tensorflow as tf
import numpy as np


class Residual(tf.keras.Model):
    def __init__(self, out_size):
        super(Residual, self).__init__()
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
        super(Model, self).__init__()
        self.width = width
        self.height = height
        self.batch_size = 256

        self.optimizer = tf.keras.optimizers.Adam(lr=0.001)

        self.lifting = tf.keras.layers.Dense(64)
        self.dense128 = tf.keras.layers.Dense(128)
        self.dense256 = tf.keras.layers.Dense(256)

        self.res1 = Residual(128)
        self.res2 = Residual(256)
        self.dense = tf.keras.Sequential([
            tf.keras.layers.Dense(512),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(512),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(width * height),
            tf.keras.layers.ReLU(),
        ])
        # self.dense = tf.keras.layers.Dense(width * height)

    def call(self, input, encoded_grids):
        """
        input: a one-hot vector with depth of 10
        """
        features = self.lifting(input) + encoded_grids
        # features.shape = [batch_size, 6, 6, 64]
        features = self.dense128(features)
        features = self.res1(features)
        # features.shape = [batch_size, 6, 6, 128]
        features = tf.nn.max_pool(features, (2, 2), (2, 2), "VALID")
        # features.shape = [batch_size, 3, 3, 128]
        features = self.dense256(features)
        features = self.res2(features)
        # features.shape = [batch_size, 3, 3, 256]
        features = tf.nn.max_pool(features, (3, 3), (1, 1), "VALID")
        # features.shape = [batch_size, 1, 1, 256]
        features = tf.reshape(features, (-1, 256))
        # features.shape = [batch_size, 256]
        features = self.dense(features)
        # features.shape = [batch_size, 36]
        return features

    def loss(self, probs, answer):
        labels = tf.reshape(answer, shape=(answer.shape[0], -1))
        labels = tf.cast(labels, tf.float32)
        accuracy = tf.reduce_mean(tf.cast((probs > 0) == tf.cast(labels, tf.bool), tf.float32))
        output_argmax = tf.argmax(probs, axis=1)
        pre_label = tf.gather(labels, output_argmax, axis=1)
        ac = tf.reduce_mean(tf.cast(pre_label, tf.float32))
        return tf.nn.sigmoid_cross_entropy_with_logits(labels, probs), accuracy, ac
