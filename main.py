import neural_net
import dataset

if __name__ == "__main__":
    model = neural_net.NeuralNetwork(3072, 400, output_nodes=10, learning_rate=0.0001, activation="relu",
                                     last_layer_activation="softmax", bias=True)
    dataset.train_mnist(model, 2, False, 32)
    dataset.test_mnist(model)
