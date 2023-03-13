import tensorflow as tf
import sys
sys.path.insert(0, 'C:\\Users\\Utente\\Desktop\\Dev\\Progetti\\OrderAi\\lib\\')
from preprocessing import normalize_dataset

class DenseLayer(tf.Module):
    def __init__(self, n_neurons, input_dim, activation=tf.identity, dropout=0.0):
        self.n_neurons = n_neurons
        self.input_dim = input_dim
        self.builded = False
        self.activation = activation
        self.dropout = dropout
    
    @tf.function
    def build(self):
        if not self.builded:
            self.W_n = tf.Variable(tf.cast(tf.random.uniform(shape=(self.n_neurons, self.input_dim)), dtype=tf.float32), dtype=tf.float32, trainable=True)
            self.bias = tf.Variable(tf.cast(tf.ones(shape=(self.n_neurons)), dtype=tf.float32), dtype=tf.float32, trainable=True)
            self.vars = [self.W_n, self.bias]
            self.builded = True
            
    @tf.function(reduce_retracing=True)
    def __call__(self, X):
        print("------DenseLayer------")
        hypotesis = tf.transpose(tf.add(tf.transpose(tf.multiply(self.W_n, X)), self.bias))
        output = tf.nn.dropout(self.activation(hypotesis), self.dropout)
        
        print("[#] output: {0}".format(output.shape))
        print("\n")
        return output