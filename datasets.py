import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils import shuffle
import pandas as pd

from sklearn.datasets import fetch_lfw_people

def build_lfw():
    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.2)
    X = lfw_people.data
    X = X.astype(np.float32) / 255.
    return X, lfw_people.images.shape[1:]

from sklearn.datasets import load_digits
def build_digits():
    digits = load_digits()
    X = digits.data
    X = X.astype(np.float32) / 16.
    return X, digits.images.shape[1:]

from lasagnekit.datasets.mnist import MNIST
def build_mnist():
    data = MNIST('all')
    data.load()
    X = data.X
    X = X.astype(np.float32)
    return X, (28, 28)

from lasagnekit.datasets.cifar10 import Cifar10
def build_cifar10():
    data = Cifar10('train')
    data.load()
    X = data.X
    X = X.astype(np.float32)
    return X, (3, 32, 32)

from lasagnekit.datasets.textures import Textures
def build_textures():
    data = Textures()
    data.load()
    X = data.X
    X = X.astype(np.float32) / 255.
    return X, data.img_dim

datasets = dict(
    lfw=build_lfw,
    digits=build_digits,
    mnist=build_mnist,
    cifar10=build_cifar10,
    textures=build_textures
)
