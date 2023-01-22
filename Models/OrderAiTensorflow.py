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
        self.activation = {}
        normal_initiliazer = tf.random_normal_initializer(seed=0, mean=0.0, stddev=1.0)
        XN = tf.keras.utils.normalize(X)
        self.X = tf.constant(XN, name="X", dtype=tf.double)
        self.y = tf.constant(np.array(y).reshape(-1, 1), name="y", dtype=tf.double)

        self.f_weights = tf.Variable(normal_initiliazer([self.n_neurons, n_samples, n_features], dtype=tf.double), name="f_weights", dtype=tf.double, trainable=True )
        self.h_weights = tf.Variable(normal_initiliazer([self.n_layers, self.n_neurons, n_samples, n_output], dtype=tf.double), name="h_weights", dtype=tf.double, trainable=True )
        self.bias = tf.Variable([0.01 for l in range(self.n_layers)], name="bias", dtype=tf.double, trainable=True )
        
        self.activation = tf.TensorArray(tf.double, size=0, dynamic_size=True, clear_after_read=False, tensor_array_name="Activation_Layers")
        self.activation.write(0, [tf.math.sigmoid(tf.matmul(X, tf.transpose(self.f_weights[n])) + self.bias[0]) for n in range(self.n_neurons)]).mark_used()
        for l in range(1,self.n_layers-1):
            self.activation.write(l, [tf.math.sigmoid(tf.matmul(self.activation.read(l-1)[n], self.h_weights[l][n]) + self.bias[l]) for n in range(self.n_neurons)])
        self.activation.write(self.n_layers-1, [tf.math.sigmoid(tf.matmul(self.activation.read(self.n_layers-2)[n], self.h_weights[self.n_layers-1][n]) + self.bias[self.n_layers-1]) for n in range(self.n_neurons)])


        mean_weights = tf.reduce_mean(self.h_weights[self.n_layers-1], axis=0)
        mean_activation = tf.reduce_mean(self.activation.read(self.n_layers-1), axis=1)

        self.predicted = tf.reduce_max(tf.math.sigmoid(tf.matmul(mean_activation, mean_weights) + tf.reshape(self.bias[self.n_layers-1], [-1, 1])), axis=0)
        self.loss = lambda : 1/2 * tf.reduce_sum( tf.square((tf.reshape(self.predicted, [-1, 1]) - self.y)))
        #self.loss = lambda : tf.losses.mean_squared_error(self.predicted, y)
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(self.lr, 5, 0.85, name="Learning_Rate")
        #self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list=[self.f_weights, self.h_weights, self.bias], name="Optimizer")
        self.optimizer = tf.optimizers.Adam(self.learning_rate).minimize(self.loss, var_list=[self.f_weights, self.h_weights, self.bias], name="Optimizer")

        self.corrects = tf.equal(self.predicted, y)
        self.accuracy = tf.reduce_mean(tf.cast(self.corrects, tf.float32))

        self.initialized = True

    def train(self):
        if self.initialized:
            pass
            
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target, test_size=0.9)

model = MLPLinearRegressor()
model.initialize(X_train, y_train)
model.train()