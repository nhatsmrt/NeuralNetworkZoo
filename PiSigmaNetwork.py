import tensorflow as tf
import numpy as np

class PiSigmaNetwork:

    def __init__(self, inp_dim, n_hidden_units):
        tf.set_random_seed(1)
        self._X = tf.placeholder(dtype = tf.float32, shape = [None, inp_dim])
        self._W = tf.get_variable(name = "W", shape = [inp_dim, n_hidden_units],
                                  initializer = tf.contrib.layers.xavier_initializer())
        self._b = tf.get_variable(name = "b", shape = [n_hidden_units])

        self._sigma = tf.matmul(self._X, self._W) + self._b
        self._pi = tf.reduce_prod(self._sigma, axis = -1, keep_dims = True)

        self._op = tf.sigmoid(self._pi)

    def fit(self, X, y, n_epochs = 10):
        self._y = tf.placeholder(dtype = tf.float32, shape = [None, 1])
        self._mean_loss = -tf.reduce_mean(self._y * tf.log(self._op) + (1 - self._y) * tf.log(1 - self._op))
        self._optimizer = tf.train.AdamOptimizer()
        self._train_step = self._optimizer.minimize(self._mean_loss)

        self._init = tf.global_variables_initializer()
        self._sess = tf.Session()
        self._sess.run(self._init)
        # loss = self._sess.run(self._pi, feed_dict={self._X: X
        #                                             })
        # print(loss)
        for e in range(n_epochs):
            self._sess.run(self._train_step, feed_dict = {self._X: X,
                                                       self._y: y})

    def predict(self, X):
        prob = self._sess.run(self._op, feed_dict = {self._X: X})
        predictions = np.round(prob).reshape(X.shape[0])
        return predictions
