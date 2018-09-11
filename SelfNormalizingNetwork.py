import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

class SelfNormalizingNet:

    def __init__(self, inp_w = 28, inp_h = 28, n_class = 10, keep_prob = 0.9, use_gpu = False, seed = 1):
        tf.set_random_seed(seed)
        self._h = inp_h
        self._w = inp_w
        self._n_class = n_class

        self._X = tf.placeholder(dtype = tf.float32, shape = [None, inp_w, inp_h, 1])

        self._keep_prob = keep_prob
        self._use_gpu = use_gpu

        if use_gpu:
            with tf.device('/device:GPU:0'):
                self.create_network(inp_w, inp_h)
        else:
            with tf.device('/device:CPU:0'):
                self.create_network(inp_w, inp_h)



    def create_network(self, inp_w, inp_h):
        self._is_training = tf.placeholder(tf.bool)
        self._keep_prob_tensor = tf.placeholder(tf.float32)

        # Create network:
        self._conv1 = self.convolutional_layer(self._X, name = "conv1", strides = 1, padding = 'SAME', inp_dim = 28,
                                               kernel_size = 5, inp_channel = 1, op_channel = 32) # [batch_size, 28, 28, 32]
        self._conv2 = self.convolutional_layer(self._conv1, name = "conv2", strides = 2, padding = 'SAME', inp_dim = 28,
                                               kernel_size = 2, inp_channel = 32, op_channel = 32) # [batch_size, 15, 15, 32]
        self._conv3 = self.convolutional_layer(self._conv2, name = "conv3", strides = 1, padding = 'SAME', inp_dim = 15,
                                               kernel_size = 5, inp_channel = 32, op_channel = 64) # [batch_size, 15, 15, 64]
        self._conv4 = self.convolutional_layer(self._conv3, name = "conv4", strides = 2, padding = 'SAME', inp_dim = 15,
                                               kernel_size = 2, inp_channel = 64, op_channel = 64) # [batch_size, 7, 7, 64]


        self._conv4_flat = tf.reshape(self._conv4, shape = [-1, 3136])
        self._fc1 = self.feedforward_layer(self._conv4_flat, name = "fc1", n_inp = 3136, n_op = 1024)
        self._fc2 = self.feedforward_layer(self._fc1, name = "fc2", n_inp = 1024, n_op = 10, final_layer = True)
        self._op = tf.nn.softmax(self._fc2)

        # self._X_flat = tf.reshape(self._X, shape = [-1, 784])
        # self._fc1 = self.feedforward_layer(self._X_flat, name = "fc1", n_inp = 784, n_op = 1024)
        # self._fc2 = self.feedforward_layer(self._fc1, name = "fc2", n_inp = 1024, n_op = 512)
        # self._fc3 = self.feedforward_layer(self._fc2, name = "fc3", n_inp = 512, n_op = 256)
        # self._fc4 = self.feedforward_layer(self._fc3, name = "fc4", n_inp = 256, n_op = 128)
        # self._fc5 = self.feedforward_layer(self._fc4, name = "fc5", n_inp = 128, n_op = 64)
        # self._fc6 = self.feedforward_layer(self._fc5, name = "fc6", n_inp = 64, n_op = 32)
        # self._fc7 = self.feedforward_layer(self._fc6, name = "fc7", n_inp = 32, n_op = 16)
        # self._fc8 = self.feedforward_layer(self._fc7, name = "fc8", n_inp = 16, n_op = 10, final_layer = True)
        # self._op = tf.nn.softmax(self._fc8)





    # Define layers and modules:
    def convolutional_layer(self, x, name, inp_dim, inp_channel, op_channel, kernel_size=3, strides = 1, padding='VALID',
                            pad = 1, dropout=True, not_activated=False):
        if pad != 0 and padding == 'VALID':
            x_padded = tf.pad(x, self.create_pad(4, pad))
        else:
            x_padded = x

        # if padding == 'VALID':
        #     op_dim = tf.floor((inp_dim + 2 * pad - kernel_size) / strides) + 1
        # else:
        #     op_dim = inp_dim
        n_neurons = inp_dim * inp_dim * inp_channel

        W_conv = tf.get_variable("W_" + name, shape=[kernel_size, kernel_size, inp_channel, op_channel],
                                 initializer=tf.initializers.random_normal(mean = 0, stddev = 1 / np.sqrt(n_neurons)))
        b_conv = tf.get_variable("b_" + name, initializer=tf.zeros(op_channel))
        z_conv = tf.nn.conv2d(x_padded, W_conv, strides=[1, strides, strides, 1], padding=padding) + b_conv
        a_conv = tf.nn.selu(z_conv)

        if dropout:
            a_conv_dropout = tf.contrib.nn.alpha_dropout(a_conv, keep_prob = self._keep_prob_tensor)
            return a_conv_dropout
        if not_activated:
            return z_conv
        return a_conv

    def feedforward_layer(self, x, name, n_inp, n_op, final_layer = False):
        W = tf.get_variable(name = "W_" + name, shape = [n_inp, n_op],
                            initializer = tf.initializers.random_normal(mean = 0, stddev = np.sqrt(1 / n_inp)))
        b = tf.get_variable(name = "b_" + name, shape = [n_op])
        z = tf.matmul(x, W) + b

        if final_layer:
            return z
        else:
            a = tf.nn.selu(z)
            a_dropout = tf.contrib.nn.alpha_dropout(a, keep_prob = self._keep_prob_tensor)
            return a_dropout


    # Predict:
    def predict(self, X):
        ans = self._sess.run(self._op,
                             feed_dict={self._X: X, self._is_training: False, self._keep_prob_tensor: 1.0})
        return ans


    # Train:
    def fit(self, X, y, num_epoch = 1, batch_size = 16, weight_save_path=None, weight_load_path=None,
            plot_losses=False, print_every = 1):
        self._y = tf.placeholder(tf.float32, shape=[None, self._n_class])
        self._mean_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self._y, logits = self._fc2,))
        self._optimizer = tf.train.AdamOptimizer(beta2 = 0.99, epsilon = 0.01)
        # self._optimizer = tf.train.AdadeltaOptimizer()
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self._train_step = self._optimizer.minimize(self._mean_loss)
        self._sess = tf.Session()
        if weight_load_path is not None:
            loader = tf.train.Saver()
            loader.restore(sess=self._sess, save_path=weight_load_path)
            print("Weight loaded successfully")
        else:
            self._sess.run(tf.global_variables_initializer())
            # shape = tf.shape(self._conv1)
            # print(self._sess.run(shape, feed_dict = {self._X: [X[0]], self._keep_prob_tensor: 1}))
        if num_epoch > 0:
            print('Training Self-Normalizing Net for ' + str(num_epoch) + ' epochs')
            self.run_model(self._sess, self._op, self._mean_loss, X, y, num_epoch, batch_size, print_every,
                           self._train_step, weight_save_path=weight_save_path, plot_losses=plot_losses)

    # Adapt from Stanford's CS231n Assignment3
    def run_model(self, session, predict, loss_val, Xd, yd,
                  epochs=1, batch_size=1, print_every=1,
                  training=None, plot_losses=False, weight_save_path=None, patience=None):
        # have tensorflow compute accuracy
        correct_prediction = tf.cast(tf.equal(tf.argmax(self._op, axis = -1), tf.argmax(self._y, axis = -1)), tf.float32)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # Define saver:
        saver = tf.train.Saver()

        # shuffle indicies
        train_indicies = np.arange(Xd.shape[0])
        np.random.shuffle(train_indicies)

        training_now = training is not None

        # setting up variables we want to compute (and optimizing)
        # if we have a training function, add that to things we compute
        variables = [self._mean_loss, correct_prediction, accuracy]
        if training_now:
            variables[-1] = training
            self._keep_prob_passed = self._keep_prob
        else:
            self._keep_prob_passed = 1.0

        # counter
        iter_cnt = 0
        val_losses = []
        early_stopping_cnt = 0
        for e in range(epochs):
            # keep track of losses and accuracy
            correct = 0
            losses = []
            # make sure we iterate over the dataset once
            for i in range(int(math.ceil(Xd.shape[0] / batch_size))):
                # generate indicies for the batch
                start_idx = (i * batch_size) % Xd.shape[0]
                idx = train_indicies[start_idx:start_idx + batch_size]

                # create a feed dictionary for this batch
                # get batch size
                actual_batch_size = yd[idx].shape[0]

                if i < int(math.ceil(Xd.shape[0] / batch_size)) - 1:
                    feed_dict = {self._X: Xd[idx, :],
                                 self._y: yd[idx],
                                 self._is_training: training_now,
                                 self._keep_prob_tensor: self._keep_prob_passed}
                    # have tensorflow compute loss and correct predictions
                    # and (if given) perform a training step
                    loss, corr, _ = session.run(variables, feed_dict=feed_dict)

                    # aggregate performance stats
                    losses.append(loss * actual_batch_size)
                    correct += np.sum(corr)

                    # print every now and then
                    if training_now and (iter_cnt % print_every) == 0:
                        print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}" \
                              .format(iter_cnt, loss, np.sum(corr) / actual_batch_size))


                else:
                    feed_dict = {self._X: Xd[idx, :],
                                 self._y: yd[idx],
                                 self._is_training: False,
                                 self._keep_prob_tensor: 1.0}
                    val_loss = session.run(self._mean_loss, feed_dict=feed_dict)
                    print("Validation loss: " + str(val_loss))
                    val_losses.append(val_loss)
                    # if training_now and weight_save_path is not None:
                    if training_now and val_loss <= min(val_losses) and weight_save_path is not None:
                        save_path = saver.save(session, save_path=weight_save_path)
                        print("Model's weights saved at %s" % save_path)
                    if patience is not None:
                        if val_loss > min(val_losses):
                            early_stopping_cnt += 1
                        else:
                            early_stopping_cnt = 0
                        if early_stopping_cnt > patience:
                            print("Patience exceeded. Finish training")
                            return
                iter_cnt += 1
            total_correct = correct / Xd.shape[0]
            total_loss = np.sum(losses) / Xd.shape[0]
            print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}" \
                  .format(total_loss, total_correct, e + 1))
            if plot_losses:
                plt.plot(losses)
                plt.grid(True)
                plt.title('Epoch {} Loss'.format(e + 1))
                plt.xlabel('minibatch number')
                plt.ylabel('minibatch loss')
                plt.show()
        return total_loss, total_correct


    def create_pad(self, n, pad):
        pad_matrix = [[0, 0]]
        for i in range(n - 2):
            pad_matrix.append([pad, pad])
        pad_matrix.append([0, 0])
        return tf.constant(pad_matrix)

    def save_weights(self, weight_save_path):
        saver = tf.train.Saver()
        saver.save(sess=self._sess, save_path=weight_save_path)
        print("Weight saved successfully")


    def evaluate(self, X, y):
        self.run_model(self._sess, self._op, self._mean_loss, X, y, 1, 16)








