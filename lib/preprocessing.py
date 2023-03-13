import csv
import numpy as np
from pandas import DataFrame
import tensorflow as tf
from scipy import stats
from sklearn.model_selection import KFold

def fill_dataset(dataframe: DataFrame):
    for column in dataframe:
        if dataframe[column].dtype != 'object':
            dataframe[column] = dataframe[column].fillna(dataframe[column].mean())
    return dataframe

def normalize_dataset(X):
    return tf.divide(
        X - tf.reduce_min(X, axis=0),
        tf.reduce_max(X, axis=0) - tf.reduce_min(X, axis=0)
    ) 

def normalize(X):
    return tf.keras.utils.normalize(X)

def denormalize(x):
    return normalize_dataset(x) * stdev(x) + tf.reduce_mean(x)

def stdev(x):
    return tf.square(tf.divide(tf.reduce_sum((x - tf.divide(tf.reduce_sum(x), len(x)))**2), len(x)-1))

def remove_outliers(X, threshold=7):
    z = np.abs(stats.zscore(X))
    return X[(z<threshold).all(axis=1)][:, 0:-1], X[(z<threshold).all(axis=1)][: ,-1]

def make_dataset(X_data,y_data,k):
    X_data, y_data = remove_outliers(np.concatenate([X_data, y_data], axis=1))
    def gen():
        for train_index, test_index in KFold(k).split(X_data):
            X_train, X_test = X_data[train_index], X_data[test_index]
            XN_train, XN_test = normalize_dataset(X_data[train_index]), normalize_dataset(X_data[test_index])
            y_train, y_test = y_data[train_index], y_data[test_index]
            yield X_train,XN_train,y_train,X_test,XN_test,y_test

    return tf.data.Dataset.from_generator(gen, (tf.double,tf.double,tf.double,tf.double,tf.double,tf.double))
