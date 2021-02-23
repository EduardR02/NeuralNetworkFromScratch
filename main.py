import neural_net
import dataset

if __name__ == "__main__":
    model = neural_net.NeuralNetwork(784, 150, 100, output_nodes=10, learning_rate=0.002, activation="relu",
                                     last_layer_activation="softmax", bias=True)
    dataset.train_mnist(model, 50, True, 32)
    dataset.test_mnist(model)
