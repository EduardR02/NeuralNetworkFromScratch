import math
from math import log

import numpy as np
import scipy.special as sci
import mnist
from numba import jit


def progressBar(current, total, epoch, acc, loss, bar_length=20):
    percent = float(current) * 100 / (total - 1)
    arrow = '-' * int(percent / 100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    print("metrics[", "acc:", acc[0] / acc[1], "; loss:", loss[0] / loss[1], "]",
          "Epoch:", epoch + 1, ";", 'Progress: [%s%s] %d %%' % (arrow, spaces, percent), "out of", total, "  ", end='\r', )


def to_categorical(arr):
    input_shape = arr.shape
    arr = arr.ravel()
    num_classes = np.max(arr) + 1
    n = arr.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), arr] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


@jit('void(float64[:,:], float64, float64[:,:], float64[:,:], float64[:,:])', nopython=True)
def backprop_sigmoid_derivative(weight_mat, lr, error, this_layer_activation, prev_layer_activation):
    # the sigmoid function has at this point already been applied to this_layer
    weight_mat -= lr * np.dot((error * this_layer_activation * (1.0 - this_layer_activation)),
                              np.transpose(prev_layer_activation))


@jit('void(float64[:,:], float64, float64[:,:], float64[:,:], float64[:,:])', nopython=True)
def backprop_relu(weight_mat, lr, error, this_layer_activation, prev_layer_activation):
    # relu derivative has been applied to this_layer by this point
    weight_mat -= lr * np.dot((error * this_layer_activation), np.transpose(prev_layer_activation))


@jit('void(float64[:,:], float64, float64[:,:], float64[:,:])', nopython=True)
def backprop_softmax_derivative(weight_mat, lr, error, prev_layer_activation):
    # (softmax activation - target values)
    weight_mat -= lr * np.dot(error, np.transpose(prev_layer_activation))


@jit('void(float64[:,:], float64, float64[:,:])', nopython=True)
def backprop_bias_single_layer(bias_mat, lr,  error):
    bias_mat -= lr * error


def relu_derivative(k):
    x = k.copy()
    x[x < 0] = 0
    x[x > 0] = 1
    return x


def leaky_relu_derivative(k):
    x = k.copy()
    x[x <= 0] = 1e-6
    x[x > 0] = 1
    return x


def mean_squared_error_cost(target, prediction):
    return np.mean(np.square(target - prediction) * 0.5)


def back_prop_single_layer(weight_mat, lr, error, this_layer_activation, prev_layer_activation, activation ="relu"):
    if activation == "sigmoid":
        backprop_sigmoid_derivative(weight_mat, lr, error, this_layer_activation, prev_layer_activation)
    elif activation == "relu":
        backprop_relu(weight_mat, lr, error, relu_derivative(this_layer_activation), prev_layer_activation)
    elif activation == "softmax":
        backprop_softmax_derivative(weight_mat, lr, error, prev_layer_activation)
    else:
        raise Exception("Given activation function does not exist")


def cross_entropy_cost_func(predicted, target):
    # only one layer in value in target will be 1 and all others will be 0, therefore sum
    # is not necessary
    # very low epsilon so the inside of log is never 0
    return -log(predicted[np.argmax(target)] + 1e-8)


def shuffle_data(samples, labels):
    indices = np.arange(labels.shape[0])
    np.random.shuffle(indices)
    labels = labels[indices]
    samples = samples[indices]
    return samples, labels


def iterate_minibatches(samples, labels, batchsize):
    # https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
    for i in range(0, samples.shape[0] - batchsize + 1, batchsize):
        excerpt = slice(i, i + batchsize)
        yield samples[excerpt], labels[excerpt]


class NeuralNetwork:
    def __init__(self, input_nodes, *hidden, output_nodes, learning_rate, activation="relu",
                 last_layer_activation="softmax", bias=False, shuffle=True):
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.lr = learning_rate
        self.shuffle = shuffle
        self.all_nodes = [layer for layer in hidden]
        self.all_nodes.insert(0, input_nodes)
        self.all_nodes.append(output_nodes)
        # create weight matrices
        self.weight_matrices = []
        for i in range(len(self.all_nodes) - 1):
            self.weight_matrices.append(np.random.normal(0.0, pow(self.all_nodes[i], -0.5),
                                                         (self.all_nodes[i + 1], self.all_nodes[i])))
        if len(self.weight_matrices) == 1 or not bias:
            self.bias = False
        else:
            self.bias = True
            self.bias_matrices = []
            x = 0
            self.bias_last_layer = 0 if last_layer_activation == "softmax" else 1
            for i in range(len(self.weight_matrices) - 1):
                x, y = self.weight_matrices[i + 1].shape
                self.bias_matrices.append(np.atleast_2d(np.zeros(y)).transpose())
            # add last bias layer if not softmax
            if self.bias_last_layer:
                self.bias_matrices.append(np.atleast_2d(np.zeros(x)).transpose())
        # activation functions
        self.sigmoid_activation = lambda x: sci.expit(x)
        self.relu_forward = lambda x: np.maximum(0.0, x)
        self.softmax = lambda x: sci.softmax(x)
        # metrics
        self.accuracy = np.ones(2)
        self.loss = np.ones(2)
        # set activation function for all except last layer, because last is always softmax
        self.activation = activation
        self.last_layer_activation = last_layer_activation

    def cross_entropy_loss(self, prediction, target):
        self.loss[0] += cross_entropy_cost_func(prediction, target)
        self.loss[1] += 1

    def mean_squared_loss(self, prediction, target):
        self.loss[0] += mean_squared_error_cost(target, prediction)
        self.loss[1] += 1

    def apply_activation_function(self, mat1, activation):
        if activation == "sigmoid":
            return self.sigmoid_activation(mat1)
        if activation == "relu":
            return self.relu_forward(mat1)
        if activation == "softmax":
            return self.softmax(mat1)

    def get_errors(self, target, outputs):
        if self.last_layer_activation == "relu":
            errors = [outputs[-1] - target]
        elif self.last_layer_activation == "softmax":
            errors = [outputs[-1] - target]
        else:
            errors = [outputs[-1] - target]
        for i in range(1, len(self.weight_matrices) + 1):
            errors.insert(0, np.matmul(self.weight_matrices[-i].transpose(), errors[0]))
        return errors

    def backpropagation(self, errors, outputs):
        for i in range(1, len(self.weight_matrices) + 1):
            if i == 1:
                back_prop_single_layer(self.weight_matrices[-i], self.lr, errors[-i], outputs[-i], outputs[-(i + 1)], self.last_layer_activation)
            else:
                back_prop_single_layer(self.weight_matrices[-i], self.lr, errors[-i], outputs[-i], outputs[-(i + 1)], self.activation)
            if self.bias and i < len(self.bias_matrices) + 1 + self.bias_last_layer:
                backprop_bias_single_layer(self.bias_matrices[-i], self.lr, errors[-(i + 1)])

    def train_without_batches(self, inputs_list, target_list):
        target = np.atleast_2d(target_list).transpose()
        outputs = self.feed_forward(inputs_list)
        errors = self.get_errors(target, outputs)
        self.backpropagation(errors, outputs)
        # metrics
        self.get_accuracy(target, outputs[-1])
        self.cross_entropy_loss(outputs[-1], target)

    def train_minibatches(self, batches, current_epoch, batch_size, sample_amt):
        for k, batch in enumerate(batches):
            outputs_sum = [np.atleast_2d(np.zeros(self.all_nodes[i])).transpose() for i in range(len(self.all_nodes))]
            errors_sum = [np.atleast_2d(np.zeros(self.all_nodes[i])).transpose() for i in range(len(self.all_nodes))]
            samples_batch, target_batch = batch
            for i in range(target_batch.shape[0]):
                target = np.atleast_2d(target_batch[i]).transpose()
                outputs = self.feed_forward(samples_batch[i])
                errors = self.get_errors(target, outputs)
                for j in range(len(outputs)):
                    outputs_sum[j] += outputs[j]
                    errors_sum[j] += errors[j]
                # metrics, yes you could and maybe should do that for the whole batch at
                # once but rn im too lazy maybe later
                self.get_accuracy(target, outputs[-1])
                self.cross_entropy_loss(outputs[-1], target)

            for i in range(len(outputs_sum)):
                outputs_sum[i] = outputs_sum[i] / (1.0 * batch_size)
                errors_sum[i] = errors_sum[i] / (1.0 * batch_size)
            self.backpropagation(errors_sum, outputs_sum)
            progressBar(k, math.ceil(sample_amt / batch_size), current_epoch, self.accuracy, self.loss)

    def get_accuracy(self, target, final):
        if np.argmax(target) == np.argmax(final):
            self.accuracy[0] += 1
        self.accuracy[1] += 1

    def feed_forward(self, inputs_list):
        outputs = [np.atleast_2d(inputs_list).transpose()]
        for i in range(len(self.weight_matrices)):
            if self.bias and i < len(self.weight_matrices) - 1 + self.bias_last_layer:
                output = np.matmul(self.weight_matrices[i], outputs[i]) + self.bias_matrices[i]
            else:
                output = np.matmul(self.weight_matrices[i], outputs[i])
            if i == len(self.weight_matrices) - 1:
                output = self.apply_activation_function(output, self.last_layer_activation)
            else:
                output = self.apply_activation_function(output, self.activation)
            outputs.append(output)
        return outputs

    def predict(self, input_list):
        return self.feed_forward(input_list)[-1]

    def train_mnist(self, epochs, load=True, batch_size=1):
        if batch_size < 1:
            raise ValueError("Batch size cannot be smaller than 1, your batch size is:", batch_size)
        self.lr = self.lr * batch_size
        train_images = mnist.train_images()
        train_labels = to_categorical(mnist.train_labels())
        train_images = train_images.reshape((-1, 784))
        train_images = (train_images / 255)
        if batch_size > train_images.shape[0]:
            raise ValueError("Batch size cannot be larger than amount of training samples, amount of training samples "
                             "is:", train_images.shape[0], "batch size is:", batch_size)
        if load:
            self.load_weights()
            if self.bias: self.load_bias()
        for j in range(epochs):
            if self.shuffle:
                train_images, train_labels = shuffle_data(train_images, train_labels)
            if batch_size == 1:
                for i in range(len(train_images)):
                    self.train_without_batches(train_images[i], train_labels[i])
                    if i % 50 == 0 or i == len(train_images) - 1:
                        progressBar(i, len(train_images), j, self.accuracy, self.loss)
            else:
                x = iterate_minibatches(train_images, train_labels, batch_size)
                self.train_minibatches(x, j, batch_size, len(train_images))
            print("")
            self.accuracy = np.ones(2)
            self.loss = np.ones(2)
        self.save_weights()
        if self.bias: self.save_bias()

    def test_net(self):
        test_images = mnist.test_images()
        test_labels = to_categorical(mnist.test_labels())
        test_images = test_images.reshape((-1, 784))
        test_images = (test_images / 255)
        self.accuracy = np.ones(2)
        self.loss = np.ones(2)

        self.load_weights()
        if self.bias: self.load_bias()
        for i in range(len(test_labels)):
            x = self.predict(test_images[i])
            self.cross_entropy_loss(x, test_labels[i])
            self.get_accuracy(x, test_labels[i])
            if i % 200 == 0 or i == len(test_labels) - 1:
                progressBar(i, len(test_labels), 666, self.accuracy, self.loss)

    def load_weights(self):
        for i in range(len(self.weight_matrices)):
            self.weight_matrices[i] = np.load(f"weights_matrix_{i}.npy")

    def save_weights(self):
        for i in range(len(self.weight_matrices)):
            np.save(f"weights_matrix_{i}.npy", self.weight_matrices[i])

    def save_bias(self):
        for i in range(len(self.bias_matrices)):
            np.save(f"bias_matrix_{i}.npy", self.bias_matrices[i])

    def load_bias(self):
        for i in range(len(self.bias_matrices)):
            self.bias_matrices[i] = np.load(f"bias_matrix_{i}.npy")
