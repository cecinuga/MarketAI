import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()
print(tf.config.list_physical_devices('CPU'))
housing = fetch_california_housing()

class MLPLinearRegressor(object):
    def __init__(self, lr=0.001, n_layers=3, n_neurons=3):
        self.lr = 0.001
        self.n_layers = n_layers
        self.n_neurons = n_neurons
        self.initialized = False

    

    def initialize(self, X, y):
        n_samples, n_features = X.shape
        n_output = y.shape[0]
        normal_initiliazer = tf.random_normal_initializer(seed=0, mean=0.0, stddev=1.0)
        XN = tf.keras.utils.normalize(X)
        self.X = tf.constant(XN, name="X", dtype=tf.double)
        self.y = tf.constant(np.array(y).reshape(-1, 1), name="y", dtype=tf.double)
        self.f_weights = tf.Variable(normal_initiliazer([self.n_neurons, n_samples, n_features], dtype=tf.double), name="f_weights", dtype=tf.double)
        self.h_weights = tf.Variable(normal_initiliazer([self.n_layers, self.n_neurons, n_output], dtype=tf.double), name="h_weights", dtype=tf.double)
        self.bias = tf.Variable(np.zeros(self.n_layers), name="bias", dtype=tf.double)
        self.activation = {}

        self.activation[0] = [tf.math.sigmoid(tf.matmul(X, tf.transpose(self.f_weights[n])) + self.bias[0]) for n in range(self.n_neurons)]
        for l in range(1,self.n_layers):
            self.activation[l] = [tf.math.sigmoid(tf.matmul(self.activation[l-1][n], tf.transpose(self.h_weights[l][n])) + self.bias[l]) for n in range(self.n_neurons)]

        self.initialized = True

    def train(self):
        if self.initialized:
            pass
            
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.9)

model = MLPLinearRegressor()
model.initialize(X_train, y_train)
model.train()