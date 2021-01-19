import neural_net
import numpy as np
import random
import time

if __name__ == "__main__":
    model = neural_net.NeuralNetwork(784, 200, 100, output_nodes=10, learning_rate=0.002, activation="relu", bias=True)
    model.train_mnist(10, False)
    model.test_net()
