from src.mnist import Generator as Gen
from src.enums import *


def main():
    path = 'data/mnist.pkl.gz'
    training_data, validation_data, test_data = Gen.load_data_wrapper(path)

    loss = Losses.CROSSENTROPY
    activation = Activations.RELU
    schedule = Schedules.EXP_RANGE.value
    epochs = 30
    batch_size = 20
    lr = schedule(0.03, epochs*len(training_data), epochs)
    net = Network([784, 30, 10], lr, cost=loss, activation=activation)
    net.sgd(training_data[:1000], epochs, batch_size, lmbda=2.0,
            evaluation_data=validation_data[:100],
            monitor_evaluation_accuracy=True)


if __name__ == '__main__':
    main()

