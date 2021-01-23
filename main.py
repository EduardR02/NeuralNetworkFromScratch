import neural_net
import numpy as np
import random
import time

if __name__ == "__main__":
    model = neural_net.NeuralNetwork(784, 100, output_nodes=10, learning_rate=0.001, activation="relu", bias=True)
    model.train_mnist(5, False, 4)
    model.test_net()
