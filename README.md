# NeuralNetworkFromScratch
This is a neural network from scratch with MNIST and CIFAR10 (no cnn yet, only dense) as the dataset for training

You can choose between sigmoid and relu and you can have multiple hidden layers. You can choose a different activation function for the last layer.
If you choose the last activation function to not be softmax and have enabled bias, the last layer will have bias, otherwise the input and output layers do not have bias.
If you train a model the layers will automatically be saved as a .npy file and can be loaded either for training or evaluating.
There are probably some mistakes as this is the first time I am writing a NN from scratch.

# Requirements
install mnist from the command line that's it
also the numba stuff is not necessary
