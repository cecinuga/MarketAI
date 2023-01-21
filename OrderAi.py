import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

housing = fetch_california_housing()

class MLPLinearRegressor(object):
    def __init__(self, lr=0.001, n_layers=3, n_neurons=3):
        self.lr = 0.001
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.initialized = False

    

    def initialize(self, X, y):
        n_samples, n_features = X.shape
        normal_initiliazer = tf.random_normal_initializer(seed=0, mean=0.0, stddev=1.0)
        XN = tf.keras.utils.normalize(X)
        self.X = tf.constant(XN, name="X", dtype=tf.float64)
        self.y = tf.constant(np.array(y).reshape(-1, 1), name="y", dtype=tf.float64)
        self.weights = tf.Variable(normal_initiliazer((self.n_layers, self.n_neurons, n_samples, n_features), dtype=tf.float64), name="weights", dtype=tf.float64)
        self.bias = tf.Variable(np.zeros(self.n_layers), name="bias", dtype=tf.float64)
        self.activation = tf.TensorArray(tf.float64, size=self.n_layers, dynamic_size=True, clear_after_read=False)
        for l in range(self.n_layers):
            if l-1 == 0:
                self.activation.write(l-1, [tf.math.sigmoid(tf.tensordot(X, self.weights[l-1][n], 1)) for n in range(self.n_neurons)]).mark_used()
            else:
                self.activation.write(l, [tf.math.sigmoid(tf.tensordot(self.weights[l-1], self.weights[l][n], 1)) for n in range(self.n_neurons)]).mark_used()


        self.initialized = True

    def train(self):
        if self.initialized:
            pass
            
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.15)

model = MLPLinearRegressor()
model.initialize(X_train, y_train)
model.train()