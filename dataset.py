import mnist
import numpy as np
from cifar10_web import cifar10


def progressBar(current, total, epoch, acc, loss, bar_length=20):
    percent = float(current) * 100 / (total - 1)
    arrow = '-' * int(percent / 100 * bar_length - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))

    print("metrics[", "acc:", acc[0] / acc[1], "; loss:", loss[0] / loss[1], "]",
          "Epoch:", epoch + 1, ";", 'Progress: [%s%s] %d %%' % (arrow, spaces, percent), "out of", total, "  ", end='\r',)


def iterate_minibatches(samples, labels, batchsize):
    # https://stackoverflow.com/questions/38157972/how-to-implement-mini-batch-gradient-descent-in-python
    for i in range(0, samples.shape[0] - batchsize + 1, batchsize):
        excerpt = slice(i, i + batchsize)
        yield samples[excerpt], labels[excerpt]


def shuffle_data(samples, labels):
    indices = np.arange(labels.shape[0])
    np.random.shuffle(indices)
    labels = labels[indices]
    samples = samples[indices]
    return samples, labels


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


def get_train_data_cifar10():
    # already preprocessed
    train_images, train_labels, test_images, test_labels = cifar10(path="C:/Users/Eduard/data/cifar10")
    train_images = train_images.astype("float64")
    return train_images, train_labels


def get_test_data_cifar10():
    train_images, train_labels, test_images, test_labels = cifar10(path="C:/Users/Eduard/data/cifar10")
    test_images = test_images.astype("float64")
    return test_images, test_labels


def get_train_data_mnist():
    train_images = mnist.train_images()
    train_labels = to_categorical(mnist.train_labels())
    train_images = train_images.reshape((-1, 784))
    train_images = (train_images / 255)
    return train_images, train_labels


def get_test_data_mnist():
    test_images = mnist.test_images()
    test_labels = to_categorical(mnist.test_labels())
    test_images = test_images.reshape((-1, 784))
    test_images = (test_images / 255)
    return test_images, test_labels


def train_mnist(model, epochs, load=True, batch_size=1):
    if batch_size < 1:
        raise ValueError("Batch size cannot be smaller than 1, your batch size is:", batch_size)
    model.lr /= batch_size
    train_images, train_labels = get_train_data_cifar10()
    if batch_size > train_images.shape[0]:
        raise ValueError("Batch size cannot be larger than amount of training samples, amount of training samples "
                         "is:", train_images.shape[0], "batch size is:", batch_size)
    if load:
        model.load_weights()
        if model.bias: model.load_bias()

    for j in range(epochs):
        if model.shuffle:
            train_images, train_labels = shuffle_data(train_images, train_labels)
            pass
        if batch_size == 1:
            for i in range(len(train_images)):
                model.train_without_batches(train_images[i], train_labels[i])
                if i % 50 == 0 or i == len(train_images) - 1:
                    progressBar(i, len(train_images), j, model.accuracy, model.loss)
        else:
            x = iterate_minibatches(train_images, train_labels, batch_size)
            model.train_minibatches(x, j, batch_size, len(train_images))
        print("")
        model.accuracy = np.ones(2)
        model.loss = np.ones(2)
    model.save_weights()
    if model.bias: model.save_bias()


def test_mnist(model):
    test_images, test_labels = get_test_data_cifar10()
    model.accuracy = np.ones(2)
    model.loss = np.ones(2)

    model.load_weights()
    if model.bias: model.load_bias()
    for i in range(len(test_labels)):
        x = model.predict(test_images[i])
        model.log_loss(x, test_labels[i])
        model.get_accuracy(x, test_labels[i])
        if i % 200 == 0 or i == len(test_labels) - 1:
            progressBar(i, len(test_labels), 666, model.accuracy, model.loss)
