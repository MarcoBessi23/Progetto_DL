import numpy as np
import array
import gzip
import os
import struct


def mnist():
    #base_url = "http://yann.lecun.com/exdb/mnist/"

    def parse_labels(filename):
        with gzip.open(filename, "rb") as fh:
            magic, num_data = struct.unpack(">II", fh.read(8))
            return np.array(array.array("B", fh.read()), dtype=np.uint8)

    def parse_images(filename):
        with gzip.open(filename, "rb") as fh:
            magic, num_data, rows, cols = struct.unpack(">IIII", fh.read(16))
            return np.array(array.array("B", fh.read()), dtype=np.uint8).reshape(num_data, rows, cols)

    train_images = parse_images("/home/marco/Documenti/data/MNIST/raw/train-images-idx3-ubyte.gz")
    train_labels = parse_labels("/home/marco/Documenti/data/MNIST/raw/train-labels-idx1-ubyte.gz")
    test_images = parse_images("/home/marco/Documenti/data/MNIST/raw/t10k-images-idx3-ubyte.gz")
    test_labels = parse_labels("/home/marco/Documenti/data/MNIST/raw/t10k-labels-idx1-ubyte.gz")

    return train_images, train_labels, test_images, test_labels



def load_mnist():
    partial_flatten = lambda x: np.reshape(x, (x.shape[0], np.prod(x.shape[1:])))
    one_hot = lambda x, k: np.array(x[:, None] == np.arange(k)[None, :], dtype=int)
    train_images, train_labels, test_images, test_labels = mnist()
    train_images = partial_flatten(train_images) / 255.0
    test_images = partial_flatten(test_images) / 255.0
    train_labels = one_hot(train_labels, 10)
    test_labels = one_hot(test_labels, 10)
    N_data = train_images.shape[0]
    #aggiungo normalizzazione 
    train_mean = np.mean(train_images, axis=0)
    train_images = train_images - train_mean
    test_images = test_images - train_mean

    return N_data, train_images, train_labels, test_images, test_labels

