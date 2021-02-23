import neural_net
import numpy as np
import random
import time

if __name__ == "__main__":
    model = neural_net.NeuralNetwork(784, 100, output_nodes=10, learning_rate=0.01, activation="relu",
                                     last_layer_activation="relu", bias=False)
    model.train_mnist(2, False, 1)
    model.test_net()
