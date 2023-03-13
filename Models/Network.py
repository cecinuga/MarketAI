import numpy as np
import tensorflow as tf
import sys
from sklearn import preprocessing

sys.path.insert(0, 'C:\\Users\\Utente\\Desktop\\Dev\\Progetti\\OrderAi\\lib\\')

class MarketAI(tf.Module):
    def __init__(self, layers, epochs=100, lr=0.1, l1=0.1, l2=0.1):
        super().__init__(name="MarketAI")
        self.epochs = epochs+1
        self.layers = layers
        self.l1 = tf.convert_to_tensor(l1, dtype=tf.float32)
        self.l2 = tf.convert_to_tensor(l2, dtype=tf.float32)
        self.Adam = tf.optimizers.Adam(lr, clipnorm=0.1, weight_decay=True)
        self.loss_history = [e for e in range(self.epochs)]
        self.r2_history = [e for e in range(self.epochs)]
        self.losses = [e for e in range(len(self.layers))]
        self.losses_test = [e for e in range(len(self.layers))]
        self._builded = False

    def _history(self, e, loss, y, y_pred):
        self.loss_history[e] = loss
        self.r2_history[e] = self._r2(y, y_pred)

    def load(self, X_train, X_test):
        self.X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        self.X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
        self.X_train_norm = tf.convert_to_tensor(tf.cast(preprocessing.normalize(X_train, axis=1), dtype=tf.float32))
        self.X_test_norm = tf.convert_to_tensor(tf.cast(preprocessing.normalize(X_test, axis=1), dtype=tf.float32))

    @tf.function
    def build(self):
        for l in range(len(self.layers)):
            self.layers[l].build()

    def _r2(self, y, y_pred):
        return tf.reduce_mean(tf.subtract(
            tf.convert_to_tensor(1, dtype=tf.float32), 
            tf.divide(
                tf.reduce_sum(tf.square(tf.subtract(y, y_pred))),
                tf.reduce_sum(tf.square(tf.subtract(y, tf.reduce_mean(y_pred))))
            )
        ))
    
    @tf.function
    def _lasso(self, weights):
        return tf.reduce_sum(tf.multiply(self.l1, tf.abs(weights)))

    @tf.function
    def _ridge(self, weights):
        return tf.reduce_sum(1/2*tf.multiply(self.l2, tf.square(weights)))

    @tf.function
    def _loss(self, y, predicted, weights):
        lasso_reg = tf.reduce_sum([self._lasso(weight) for weight in weights])
        ridge_reg = tf.reduce_sum([self._ridge(weight) for weight in weights])
        return tf.add(tf.add(tf.losses.MAE(y, predicted), lasso_reg), ridge_reg)    

    def _forward_lstm_layer(self, i, X_loss, output):
        outputs = []
        losses = []
        for x_loss, x in zip(X_loss, output):
            loss, output = self._forward_lstm(i, x, x_loss)
            losses.append(loss)
            outputs.append(output)
        loss = tf.reduce_sum(losses)
        return loss, outputs

    @tf.function
    def _forward_lstm(self, i, x, x_loss):
        output, h = self.layers[i](x)
        loss = self._loss(x_loss, output, [self.layers[i].W_hx, self.layers[i].W_hh, self.layers[i].W_hy])
        return loss, tf.convert_to_tensor(output, dtype=tf.float32)
    
    def _forward_dense_layer(self, i, X_loss, X_norm):
        outputs = []
        losses = []
        for x, x_loss in zip(X_loss, X_norm):
            loss, output = self._forward_dense(i, x, x_loss)
            losses.append(loss)
            outputs.append(output)
        loss = tf.reduce_sum(losses)
        return loss, tf.convert_to_tensor(outputs, dtype=tf.float32)

    @tf.function
    def _forward_dense(self, i, x, x_loss):
        output = self.layers[i](x)
        loss = self._loss(x_loss, output, [self.layers[i].W_n])
        return loss, output

    def predict(self):
        self.predicted_test, self.final_loss_test = self._forward(self.X_test, self.X_test_norm)
        
    def _forward(self, X, X_norm):
        losses = [e for e in range(len(self.layers))]
        losses[0], output = self._forward_dense_layer(0, X, X_norm)
        losses[1], output = self._forward_lstm_layer(1, X, output)
        losses[2], output = self._forward_dense_layer(2, X, output)
        losses[3], output = self._forward_dense_layer(3, X, output)
        predicted = tf.reduce_mean(tf.reshape(output, [output.shape[1], output.shape[0], output.shape[2]]), axis=0)
        final_loss = tf.reduce_sum(losses)
        return predicted, final_loss

    def _backprop(self, tape: tf.GradientTape):
        grads = tape.gradient(self.final_loss, self.variables) 
        self.Adam.apply_gradients(zip(grads, self.variables))

    def train(self):
        for e in range(self.epochs):
            with tf.GradientTape(watch_accessed_variables=True, persistent=True) as tape:
                self.predicted_train, self.final_loss = self._forward(self.X_train, self.X_train_norm)
            self._backprop(tape)

            self._history(e, self.final_loss, self.X_train, self.predicted_train)
            if e%100==0 or e == 0 or e == self.epochs-1:
                print("[{2}] TotalLoss: {0} R2: {1}".format(self.final_loss, self.r2_history[e], e))