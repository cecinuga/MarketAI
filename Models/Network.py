import numpy as np
import tensorflow as tf

class MLPLinearRegressor(tf.Module):
    def __init__(self, layers, k, epochs=100, lr=0.01, batch_size=50):
        self.layers = layers
        self.k = k
        self.epochs = epochs
        self.batch_size = batch_size
        self.batch_counter = 0
        self.history_builded = False
        self.learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(lr, decay_steps=epochs, decay_rate=0.50, staircase=True)
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.train_mse = tf.keras.metrics.MeanSquaredError()
        self.train_mae = tf.keras.metrics.MeanAbsoluteError()
        self.train_accuracy = tf.keras.metrics.MeanSquaredLogarithmicError()
        self.test_mse = tf.keras.metrics.MeanSquaredError()
        self.test_mae = tf.keras.metrics.MeanAbsoluteError()
        self.test_accuracy = tf.keras.metrics.MeanSquaredLogarithmicError()
        self.regularizer = tf.keras.layers.ActivityRegularization()
        self.loss_history = [e for e in range(epochs*k+1)]
        self.bias_history = [e for e in range(epochs*k+1)]
        self.mae_train_error_history = [e for e in range(epochs*k+1)]
        self.mse_train_error_history = [e for e in range(epochs*k+1)]
        self.mae_test_error_history = [e for e in range(epochs*k+1)]
        self.mse_test_error_history = [e for e in range(epochs*k+1)]
        self.residual_tr = [e for e in range(epochs*k+1)]
        self.residual_tt = [e for e in range(epochs*k+1)]
        self.r2_accuracy_tr = [e for e in range(epochs*k+1)]
        self.r2_accuracy_tt = [e for e in range(epochs*k+1)]

    def reset_history_metrics(self):
        self.train_mse.reset_state()
        self.train_mae.reset_state()
        self.test_mse.reset_state()
        self.test_mae.reset_state()
    
    def update_states(self, loss, i, y, y_test):
        self.bias_history[i] = self.bias
        self.loss_history[i] = loss.numpy()
        self.mae_train_error_history[i] = self.train_mae.result().numpy()
        self.mse_train_error_history[i] = self.train_mse.result().numpy()
        self.mae_test_error_history[i] = self.test_mae.result().numpy()
        self.mse_test_error_history[i] = self.test_mse.result().numpy()
        self.r2_accuracy_tr[i] = self.r2(y, self.predicted_train)
        self.r2_accuracy_tt[i] = self.r2(y_test, self.predicted_test)
        self.residual_tr[i] = tf.reduce_mean(tf.subtract(y, self.predicted_train))
        self.residual_tt[i] = tf.reduce_mean(tf.subtract(y_test, self.predicted_test))

    def calc_metrics(self, y, y_test, e, k, loss):
        self.reset_history_metrics()
        self.train_mae.update_state(y, self.predicted_train)
        self.train_mse.update_state(y, self.predicted_train)
        self.test_mae.update_state(y_test, self.predicted_test)
        self.test_mse.update_state(y_test, self.predicted_test)
        self.update_states(loss, e+(k*self.epochs), y, y_test)

    def calc_history(self, e_, epochs, k):
        if not self.history_builded:
            self.train_history = [[i for i in range(self.predicted_train.shape[0])] for e in range(epochs*self.k+1)]
            self.test_history = [[i for i in range(self.predicted_test.shape[0])] for e in range(epochs*self.k+1)]
            self.history_builded = True
        self.train_history[e_] = self.predicted_train
        self.test_history[e_] = self.predicted_test
        

    def verify_batch(self, X, X_test):
        if self.batch_counter >= X:
            self.batch_counter = 0
        
    def r2(self, y, y_pred):
        return tf.subtract(
            tf.convert_to_tensor(1, dtype=tf.float64), 
            tf.divide(
                tf.reduce_sum(tf.square(tf.subtract(y, y_pred))),
                tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y_pred))))
            )
        )

    @tf.function
    def lasso(self):
        return tf.reduce_sum(tf.norm(self.layers[-1].weights))

    @tf.function
    def ridge(self):
        return tf.reduce_sum(tf.square(tf.norm(self.layers[-1].weights)))

    @tf.function(reduce_retracing=True)
    def loss(self, y, predicted):
        return tf.add(tf.add(tf.losses.MSE(y, predicted), self.lasso()), 0)
        
    """@tf.function(reduce_retracing=True)
    def elastic_loss(self, y, predicted):
        return 1/2(tf.matmul(tf.transpose(tf.subtract(predicted, y)), tf.subtract(predicted, y)))+1/2"""
    
    @tf.function
    def _predict(self, X, bias, compressed_weights, error):
        return tf.reduce_mean(tf.add(tf.add(tf.multiply(X, compressed_weights), bias), error), axis=1, name="Predict")
    
    def predict(self, x):
        return tf.reduce_mean(tf.add(tf.multiply(x, self.compressed_weights), self.bias))

    @tf.function(reduce_retracing=True)
    def _forward(self, X):
        for layer in self.layers:
            X = layer(X)
        return X

    @tf.function(reduce_retracing=True)
    def _backprop(self, X, XN, X_test, y, y_test):
        error_tr = 0
        error_tt = 0
        batch_xn = XN[self.batch_counter:self.batch_counter+self.batch_size, :]
        weights = self._forward(batch_xn)
        bias = self.layers[-1].bias

        compressed_weights = tf.reduce_sum(tf.nn.relu(tf.add(tf.multiply(tf.transpose(batch_xn), tf.reduce_mean(weights, axis=1)), bias)), axis=1)
        predicted_train = self._predict(X, bias, compressed_weights, error_tr)
        predicted_test = self._predict(X_test, bias, compressed_weights, error_tt)
        error_tr = tf.reduce_mean(tf.subtract(y, predicted_train))
        error_tt = tf.reduce_mean(tf.subtract(y_test, predicted_test))
        loss = tf.nn.scale_regularization_loss(self.loss(y, predicted_train))

        #intercept_tr = tf.add(tf.multiply(X, compressed_weights), bias)[:, 0]
        #intercept_tt = tf.add(tf.multiply(X_test, compressed_weights), bias)[:, 0]

        self.vars = [self.layers[0].weights, self.layers[1].weights, self.layers[2].weights,self.layers[3].weights,self.layers[4].weights, self.layers[0].bias, self.layers[1].bias, self.layers[2].bias,self.layers[3].bias, self.layers[4].bias]
        self.batch_counter = self.batch_counter + self.batch_size
        return loss, bias, compressed_weights, predicted_train, predicted_test, error_tt #intercept_tr, intercept_tt

    def run(self, X, X_test, XN, XN_test, y, y_test, k):
        for e in range(self.epochs+1):
            self.verify_batch(X.shape[0], X_test.shape[0])
            
            with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
                loss, self.bias, self.compressed_weights, self.predicted_train, self.predicted_test, self.error_tt = self._backprop(X, XN, X_test, y, y_test)
            
            grads = tape.gradient(loss, self.vars)  
            self.optimizer.apply_gradients(zip(grads, self.vars))   
            self.calc_metrics(y, y_test, e, k, loss)
            self.calc_history(e, self.epochs, k)
            #if e%50==0:
                #print("[{0}] Train Score: {1}, Test Score: {2} Loss: {3} Error: {4} Bias: {5} Weights: {6} ".format(e, np.round(self.r2_accuracy_tr[e], 3), np.round(self.r2_accuracy_tt[e], 3), np.round(loss.numpy(), 3), np.round(self.error_tt.numpy(), 3), np.round(self.bias.numpy(), 3), np.round(self.compressed_weights.numpy(), 3)))
            