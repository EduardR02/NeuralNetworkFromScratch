import math
from dataset import progressBar
from math import log
import numpy as np
import scipy.special as sci
from numba import jit


@jit('void(float64[:,:], float64[:,:], float64[:,:], float64[:,:])', nopython=True)
def sigmoid_gradients(gradients, error, this_layer_actv, prev_layer_activation):
    # the sigmoid function has at this point already been applied to this_layer
    gradients += np.dot(error * this_layer_actv * (1.0 - this_layer_actv), np.transpose(prev_layer_activation))


@jit('void(float64[:,:], float64[:,:], float64[:,:], float64[:,:])', nopython=True)
def relu_gradients(gradients, error, this_layer_actv, prev_layer_activation):
    # relu derivative has been applied to this_layer by this point
    gradients += np.dot(error * this_layer_actv, np.transpose(prev_layer_activation))


@jit('void(float64[:,:], float64[:,:],  float64[:,:])', nopython=True)
def softmax_gradients(gradients, error, prev_layer_activation):
    # (softmax activation - target values)
    gradients += np.dot(error, np.transpose(prev_layer_activation))


@jit('void(float64[:,:], float64, float64[:,:])', nopython=True)
def backprop_bias_single_layer(bias_mat, lr,  error):
    bias_mat -= lr * error


def relu_derivative(k):
    x = k.copy()
    x[x < 0] = 0
    x[x > 0] = 1.0
    return x


def leaky_relu_derivative(k):
    x = k.copy()
    x[x <= 0] *= 0.01
    x[x > 0] = 1.0
    return x


def calc_gradients_single_layer(gradients, error, this_layer_actv, prev_layer_activation, activation ="relu"):
    if activation == "sigmoid":
        sigmoid_gradients(gradients, error, this_layer_actv, prev_layer_activation)
    elif activation == "relu":
        relu_gradients(gradients, error, relu_derivative(this_layer_actv), prev_layer_activation)
    elif activation == "softmax":
        softmax_gradients(gradients, error, prev_layer_activation)
    else:
        raise Exception("Given activation function does not exist")


def mean_squared_error_cost(prediction, target):
    return np.mean(np.square(target - prediction) * 0.5)


def cross_entropy_cost_func(predicted, target):
    # only one layer in value in target will be 1 and all others will be 0, therefore sum
    # is not necessary
    # very low epsilon so the inside of log is never 0
    return -log(predicted[np.argmax(target)] + 1e-8)


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
        if (len(self.weight_matrices) == 1 and last_layer_activation == "softmax") or not bias:
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
        self.error_func = "categorical_crossentropy" if last_layer_activation == "softmax" else "mean_squared"

    def log_loss(self, prediction, target):
        if self.error_func == "categorical_crossentropy":
            self.loss[0] += cross_entropy_cost_func(prediction, target)
        else:
            self.loss[0] += mean_squared_error_cost(prediction, target)
        self.loss[1] += 1

    def apply_activation_function(self, mat1, activation):
        if activation == "sigmoid":
            return self.sigmoid_activation(mat1)
        if activation == "relu":
            return self.relu_forward(mat1)
        if activation == "softmax":
            return self.softmax(mat1)

    def get_errors(self, target, outputs):
        errors = [outputs[-1] - target]
        for i in range(1, len(self.weight_matrices)):
            errors.insert(0, np.matmul(self.weight_matrices[-i].transpose(), errors[0]))
        return errors

    def get_gradients(self, errors, outputs):
        gradients = [np.zeros(i.shape) for i in self.weight_matrices]
        for i in range(len(gradients)):
            if i == len(self.weight_matrices) - 1: actv_func = self.last_layer_activation
            else: actv_func = self.activation
            calc_gradients_single_layer(gradients[i], errors[i], outputs[i + 1], outputs[i], actv_func)
        return gradients

    def update_weights_and_bias(self, gradients, errors):
        for i in range(len(self.weight_matrices)):
            self.weight_matrices[i] -= self.lr * gradients[i]
            if self.bias and 0 < i < len(self.bias_matrices) - 1 + self.bias_last_layer:
                backprop_bias_single_layer(self.bias_matrices[i], self.lr, errors[i])

    def train_without_batches(self, inputs_list, target_list):
        target = np.atleast_2d(target_list).transpose()
        outputs = self.feed_forward(inputs_list)
        errors = self.get_errors(target, outputs)
        gradients = self.get_gradients(errors, outputs)
        self.update_weights_and_bias(gradients, errors)
        # metrics
        self.get_accuracy(outputs[-1], target)
        self.log_loss(outputs[-1], target)

    def train_minibatches(self, batches, current_epoch, batch_size, sample_amt):
        for k, batch in enumerate(batches):
            errors_sum = [np.atleast_2d(np.zeros(self.all_nodes[i + 1])).transpose() for i in range(len(self.all_nodes) - 1)]
            gradients_sum = [np.zeros(i.shape) for i in self.weight_matrices]

            samples_batch, target_batch = batch
            # should be possible to parallelize on gpu but i don't know cuda
            for i in range(batch_size):
                target = np.atleast_2d(target_batch[i]).transpose()
                outputs = self.feed_forward(samples_batch[i])
                errors = self.get_errors(target, outputs)
                gradients = self.get_gradients(errors, outputs)
                for j in range(len(errors)):
                    gradients_sum[j] += gradients[j]
                    errors_sum[j] += errors[j]
                self.get_accuracy(outputs[-1], target)
                self.log_loss(outputs[-1], target)
            self.update_weights_and_bias(gradients_sum, errors_sum)
            progressBar(k, math.ceil(sample_amt / batch_size), current_epoch, self.accuracy, self.loss)

    def get_accuracy(self, final, target):
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

    def check_dead_relus(self):
        s = 0
        for weight_matrix in self.weight_matrices:
            s += len(np.where(weight_matrix == 0))
        print(s)

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
