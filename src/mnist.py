""" This file has operations for load MNIST image data """

import os
import random
import pickle
import gzip
import numpy as np
from .network import Network


class Generator:
    """" Static class that has methods to load the MNIST image data """
    @staticmethod
    def load_data(path):
        """Return the MNIST data as a tuple containing the training data,
        the validation data, and the test data.

        The ``training_data`` is returned as a tuple with two entries.
        The first entry contains the actual training images.  This is a
        numpy ndarray with 50,000 entries.  Each entry is, in turn, a
        numpy ndarray with 784 values, representing the 28 * 28 = 784
        pixels in a single MNIST image.

        The second entry in the ``training_data`` tuple is a numpy ndarray
        containing 50,000 entries.  Those entries are just the digit
        values (0...9) for the corresponding images contained in the first
        entry of the tuple.

        The ``validation_data`` and ``test_data`` are similar, except
        each contains only 10,000 images.

        This is a nice data format, but for use in neural networks it's
        helpful to modify the format of the ``training_data`` a little.
        That's done in the wrapper function ``load_data_wrapper()``, see
        below.
        """
        with gzip.open(path, 'rb') as f:
            train, val, test = pickle.load(f, encoding='latin1')

        return train, val, test

    @staticmethod
    def load_data_wrapper(path):
        """Return a tuple containing ``(training_data, validation_data,
        test_data)``. Based on ``load_data``, but the format is more
        convenient for use in our implementation of neural networks.

        In particular, ``training_data`` is a list containing 50,000
        2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
        containing the input image.  ``y`` is a 10-dimensional
        numpy.ndarray representing the unit vector corresponding to the
        correct digit for ``x``.

        ``validation_data`` and ``test_data`` are lists containing 10,000
        2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
        numpy.ndarry containing the input image, and ``y`` is the
        corresponding classification, i.e., the digit values (integers)
        corresponding to ``x``.

        Obviously, this means we're using slightly different formats for
        the training data and the validation / test data.  These formats
        turn out to be the most convenient for use in our neural network
        code."""
        tr_d, va_d, te_d = Generator.load_data(path)
        training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
        training_results = [Network.vectorized_result(y) for y in tr_d[1]]
        training_data = list(zip(training_inputs, training_results))
        validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
        validation_data = list(zip(validation_inputs, va_d[1]))
        test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
        test_data = list(zip(test_inputs, te_d[1]))
        return training_data, validation_data, test_data

    @staticmethod
    def expand_data():
        print("Expanding the MNIST training set")

        if os.path.exists("data/mnist_expanded.pkl.gz"):
            print("The expanded training set already exists.")
        else:
            with gzip.open("data/mnist.pkl.gz", 'rb') as f:
                training_data, validation_data, test_data = pickle.load(f)
            expanded_training_pairs = []
            j = 0  # counter
            for x, y in zip(training_data[0], training_data[1]):
                expanded_training_pairs.append((x, y))
                image = np.reshape(x, (-1, 28))
                j += 1
                if j % 1000 == 0:
                    print("Expanding image number", j)
                # iterate over data telling us the details of how to
                # do the displacement
                for d, axis, index_position, index in [(1, 0, "first", 0),
                                                       (-1, 0, "first", 27),
                                                       (1, 1, "last", 0),
                                                       (-1, 1, "last", 27)]:
                    new_img = np.roll(image, d, axis)
                    if index_position == "first":
                        new_img[index, :] = np.zeros(28)
                    else:
                        new_img[:, index] = np.zeros(28)
                    expanded_training_pairs.append((
                        np.reshape(new_img, 784), y))
            random.shuffle(expanded_training_pairs)
            expanded_training_data = [
                list(d) for d in zip(*expanded_training_pairs)]
            print("Saving expanded data. This may take a few minutes.")
            with gzip.open("data/mnist_expanded.pkl.gz", "w") as f:
                pickle.dump((
                    expanded_training_data, validation_data, test_data), f)
