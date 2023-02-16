import sys
sys.path.insert(0, 'C:\\Users\\Utente\\Desktop\\Dev\\Progetti\\OrderAi\\lib\\')
from stats import *

def run_model(data, model):
    k = 0
    for data in data.as_numpy_iterator():
        X_train, XN_train, y_train, X_test, XN_test, y_test = data
        model.run(X_train, X_test, XN_train, XN_test, tf.squeeze(y_train), tf.squeeze(y_test), k)
        write_stats(X_train.shape[0]+X_test.shape[0], X_train.shape[1], model.loss_history[-1], epochs, lr, batch_size, kk, model.r2_accuracy_tt[-1].numpy(), model.train_mse.result().numpy(), model.test_mse.result().numpy(), model.train_mae.result().numpy(), model.test_mae.result().numpy(), model.residual_tr[-1].numpy(), model.residual_tt[-1].numpy())
        k = k+1 
        break
    return X_train, X_test, XN_train, XN_test, tf.squeeze(y_train), tf.squeeze(y_test)
    