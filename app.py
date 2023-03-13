import sys
sys.path.insert(0, 'C:\\Users\\Utente\\Desktop\\Dev\\Progetti\\OrderAi\\lib\\')
sys.path.insert(1, 'C:\\Users\\Utente\\Desktop\\Dev\\Progetti\\OrderAi\\Models\\')
sys.path.insert(2, 'C:\\Users\\Utente\\Desktop\\Dev\\Progetti\\OrderAi\\Models\\components')
import os
import pandas as pd
import matplotlib.pyplot as plt
from Network import MarketAI
from DenseLayer import DenseLayer
from LSTMLayer import LSTMLayer
from datetime import datetime
from graphics import Graphics
from preprocessing import *
from termcolor import colored, cprint

def prepare_data():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
    stocks = pd.read_csv("./data/datasets/all_stocks_5yr.csv")
    stocks = pd.DataFrame(stocks)
    stocks.drop(["Name", "date"], axis=1, inplace=True)
    rnnstocks = stocks.values[0:100, :]
    rnnstocks = pd.DataFrame(rnnstocks)
    rnnstocks = fill_dataset(rnnstocks)
    rnnstocks = pd.DataFrame(rnnstocks)
    rnnstocks.columns = stocks.columns

    rnnstocks["target"] = rnnstocks["close"]
    n_features = len(rnnstocks.values[0])
    n_rows = 10
    n_ciclics = 10
    data = np.array(rnnstocks.values).reshape(n_ciclics, n_rows, n_features)
    X_test = data[:, 0:-1, :]
    Y_test = data[:, -1, :]

    n_rows = 20
    n_ciclics = 5
    data = np.array(rnnstocks.values).reshape(n_ciclics, n_rows, n_features)
    X_train = data[:, 0:-1, :]
    Y_train = data[:, -1, :]
    return X_train, X_test

def plot(model):
    fig, ax = plt.subplots(2, 4, figsize=(20, 8))
    ax[0, 0].plot([e for e in range(len(model.loss_history))], model.loss_history, label="Loss")
    ax[0, 0].set_ylabel('Loss')
    ax[0, 0].set_xlabel('Epochs')
    ax[0, 0].legend()

    ax[0, 2].plot([e for e in range(len(model.X_train))], model.X_train[:, -1], label="Train Price", color='red')
    ax[0, 2].set_ylabel('Train Price')
    ax[0, 2].set_xlabel('Time')
    ax[0, 2].legend()

    ax[0, 3].plot([e for e in range(len(model.X_train))], model.predicted_train[:, -1], label="Predicted Train Price", color='green')
    ax[0, 3].set_ylabel('Predicted Train Price')
    ax[0, 3].set_xlabel('Time')
    ax[0, 3].legend()

    ax[1, 0].plot([e for e in range(len(model.X_test))], model.X_test[:, -1], label="Test Price", color='red')
    ax[1, 0].set_ylabel('Tested Price')
    ax[1, 0].set_xlabel('Time')
    ax[1, 0].legend()

    ax[1, 1].plot([e for e in range(len(model.X_test))], model.predicted_test[:, -1], label="Predicted Test Price", color='green')
    ax[1, 1].set_ylabel('Predicted Tested Price')
    ax[1, 1].set_xlabel('Time')
    ax[1, 1].legend()

    ax[0, 1].plot([e for e in range(len(model.r2_history))], model.r2_history, label="R2 Score", color='green')
    ax[0, 1].set_ylabel('R2 Score')
    ax[0, 1].set_xlabel('Time')
    ax[0, 1].legend()
    plt.show()

def main():
    X_train, X_test = prepare_data()
    n_ciclic, n_rows, n_features = X_train.shape
    n_neurons = 1
    epochs = 5
    lr = 0.001
    l1, l2 = 0.1, 0.1
    model = MarketAI([
        DenseLayer(n_neurons, n_features, activation=tf.nn.relu, dropout=0.3),
        LSTMLayer(n_neurons, n_features, 1),
        DenseLayer(n_neurons, n_features, activation=tf.nn.relu, dropout=0.3),
        DenseLayer(1, n_features, activation=tf.nn.relu),
    ], epochs, lr, l1, l2)
    model.build()
    model.load(X_train[0], X_test[2])
    model.train()
    model.predict()
    plot(model)


if __name__ == '__main__':
    main()
