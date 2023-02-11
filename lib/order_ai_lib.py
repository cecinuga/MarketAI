import tensorflow as tf
from sklearn.model_selection import KFold

def normalize_dataset(X):
    return tf.keras.utils.normalize(X)


