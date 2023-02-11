import tensorflow as tf
from sklearn.model_selection import KFold

def normalize_dataset(X):
    return tf.keras.utils.normalize(X)

def make_dataset(X_data,y_data,k):
    def gen():
        for train_index, test_index in KFold(k).split(X_data):
            X_train, X_test = X_data[train_index], X_data[test_index]
            XN_train, XN_test = normalize_dataset(X_data[train_index]), normalize_dataset(X_data[test_index])
            y_train, y_test = y_data[train_index], y_data[test_index]
            yield X_train,XN_train,y_train,X_test,XN_test,y_test

    return tf.data.Dataset.from_generator(gen, (tf.double,tf.double,tf.double,tf.double,tf.double,tf.double))
    
def write_stats(samples, features, epochs, lr, batch_size, cross_k):
    with open('../data/stats.txt', 'w') as:
        pass
