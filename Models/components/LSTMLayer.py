import tensorflow as tf
from sklearn import preprocessing
import sys
sys.path.insert(0, 'C:\\Users\\Utente\\Desktop\\Dev\\Progetti\\OrderAi\\lib\\')

class LSTMLayer(tf.Module):
    def __init__(self, units, input_dim, output_dim):
        self.units = units
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.builded = False   

    @tf.function
    def build(self):
        if not self.builded:
            self.W_hx = tf.Variable(tf.cast(preprocessing.normalize(tf.random.normal(shape=(self.units, self.input_dim)),axis=1), dtype=tf.float32), trainable=True, dtype=tf.float32)
            self.W_hh = tf.Variable(tf.cast(preprocessing.normalize(tf.random.normal(shape=(self.units, self.units)),axis=1), dtype=tf.float32), trainable=True, dtype=tf.float32)
            self.W_hy = tf.Variable(tf.cast(preprocessing.normalize(tf.random.normal(shape=(self.input_dim, self.units)),axis=1), dtype=tf.float32), trainable=True, dtype=tf.float32)
            self.h = tf.Variable(tf.cast(tf.zeros([self.units, 1]), dtype=tf.float32), dtype=tf.float32)
            self.vars = [self.W_hx, self.W_hh, self.W_hy]
            self.builded = True

    @tf.function(reduce_retracing=True)
    def __call__(self, x):
        print("------LSTMLayer------")
        print("[#] W_hx: {0}, W_hh: {1}, W_hy: {2}, X: {3}".format(self.W_hx.shape, self.W_hh.shape, self.W_hy.shape, x.shape))
        updated_input = tf.matmul(self.W_hx, x, transpose_b=True)
        updated_memory = tf.matmul(self.W_hh, self.h)
        print("[#] updated_input: {0}, updated_memory: {1}".format(updated_input.shape, updated_memory.shape))
        self.h = tf.math.tanh(
            tf.add(updated_memory, updated_input), 
        )
        output = tf.transpose(tf.nn.sigmoid(tf.matmul(self.W_hy, self.h)))
        print("[#] h: {0}, output: {1}".format(self.h.shape, output.shape))
        print("\n")
        return output, self.h
    