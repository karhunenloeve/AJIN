#!/usr/bin/env python

import typing
import numpy as np
import math
import matplotlib.pyplot as plt

from random import randint
from typing import Callable
from keras import backend as K
from keras.layers import Input, Dense, Conv2D
from keras.layers import MaxPooling2D, UpSampling2D, Lambda
from keras.layers import multiply, add, LeakyReLU
from keras.layers import BatchNormalization, Reshape, concatenate
from keras.layers import Flatten, Cropping2D, Concatenate
from keras.models import Model
from keras.losses import kullback_leibler_divergence
from keras.datasets import mnist, cifar10, cifar100, boston_housing
from keras.callbacks import TensorBoard
from keras.backend import slice


def take_out_element(k: tuple, r) -> tuple:
    """
        **A function taking out specific values.**
        
        + param **k**: tuple object to be processed, type `tuple`.
        + param **r**: value to be removed, type `int, float, string, None`.
        + return **k2**: cropped tuple object, type `tuple`.
    """
    k2 = list(k)
    k2.remove(r)
    return tuple(k2)


def primeFactors(n):
    """
        **A function that returns the prime factors of an integer.**
        
        + param **n**: an integer, type `int`.
        + return **factors**: a list of prime factors, type `list`.
    """
    factors = []
    # Print the number of two's that divide n.
    while n % 2 == 0:
        factors.append(n / 2)

    # n must be odd at this point
    # so a skip of 2 ( i = i + 2) can be used.
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        # while i divides n, append i ad divide n.
        while n % i == 0:
            factors.append(n / i)

    # Condition if n is a prime.
    # number greater than 2 .
    if n > 2:
        return [n]
    else:
        return factors


def load_data_keras(
    dimensions: tuple, factor: float = 255.0, dataset: str = "mnist"
) -> tuple:
    """
        **A utility function to load datasets.**

        This functions helps to load particular datasets ready for a processing with convolutional
        or dense autoencoders. It depends on the specified shape (the input dimensions). This functions
        is for validation purpose and works for keras datasets only.
        Supported datasets are `mnist` (default), `cifar10`, `cifar100` and `boston_housing`.
        The shapes: `mnist (28,28,1)`, `cifar10 (32,32,3)`, `cifar100 (32,32,3)`

        + param **dimensions**: dimension of the data, type `tuple`.
        + param **factor**: division factor, default is `255`, type `float`.
        + param **dataset**: keras dataset, default is `mnist`,type `str`.
        + return **X_train, X_test, input_image**: , type `tuple`.
    """
    input_image = Input(shape=dimensions)

    # Loading the data and dividing the data into training and testing sets.
    if dataset == "mnist":
        (X_train, _), (X_test, _) = mnist.load_data()
    elif dataset == "cifar10":
        (X_train, _), (X_test, _) = cifar10.load_data()
    elif dataset == "cifar100":
        (X_train, _), (X_test, _) = cifar100.load_data()
    elif dataset == "boston_housing":
        (X_train, _), (X_test, _) = boston_housing.load_data()

    else:
        print("You choose a not configured dataset. Please select a valid option.")
        return None

    # Cleaning and reshaping the data as required by the model.
    X_train = X_train.astype("float64") / factor
    X_train = np.reshape(
        X_train, (len(X_train), dimensions[0], dimensions[1], dimensions[2])
    )
    X_test = X_test.astype("float64") / factor
    X_test = np.reshape(
        X_test, (len(X_test), dimensions[0], dimensions[1], dimensions[2])
    )

    return (X_train, X_test, input_image)


def add_gaussian_noise(
    data: np.ndarray, noise_factor: float = 0.5, mean: float = 0.0, std: float = 1.0
) -> np.ndarray:
    """
        **A utility function to add gaussian noise to data.**

        The purpose of this functions is validating certain models under gaussian noise.
        The noise can be added changing the mean, standard deviation and the amount of
        noisy points added.

        + param **noise_factor**: amount of noise in percent, type `float`.
        + param **data**: dataset, type `np.ndarray`.
        + param **mean**: mean, type `float`.
        + param **std**: standard deviation, type `float`.
        + return **x_train_noisy**: noisy data, type `np.ndarray`.
    """
    x_train_noisy = data + noise_factor * np.random.normal(
        loc=mean, scale=std, size=data.shape
    )
    x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
    return x_train_noisy


def crop_tensor(dimension: int, start: int, end: int) -> Callable:
    """
        **A utility function cropping a tensor along a given dimension.**

        The purpose of this function is to be used for multivariate cropping and to serve
        as a procedure for the invertible autoencoders, which need a cropping to make the
        matrices trivially invertible, as can be seen in the `Real NVP` architecture.
        This procedure works up to dimension `4`.

        + param **dimension**: the dimension of cropping, type `int`.
        + param **start**: starting index for cropping, type `int`.
        + param **end**: ending index for cropping, type `int`.
        + return **Lambda(func)**: Lambda function on the tensor, type `Callable`.
    """
    try:

        def func(x):
            if dimension == 0:
                return x[start:end]
            if dimension == 1:
                return x[:, start:end]
            if dimension == 2:
                return x[:, :, start:end]
            if dimension == 3:
                return x[:, :, :, start:end]
            if dimension == 4:
                return x[:, :, :, :, start:end]

        return Lambda(func)
    except IndexError:
        print("Sorry, your index is out of range.")


def convolutional_group(
    _input: np.ndarray,
    filterNumber: int,
    alpha: float = 5.5,
    kernelSize: tuple = (2, 2),
    kernelInitializer: str = "uniform",
    padding: str = "same",
    useBias: bool = True,
    biasInitializer: str = "zeros",
):
    """
        **This group can be extended for deep learning models and is a sequence of convolutional layers.**

        The convolutions is a `2D`-convolution and uses a `LeakyRelu` activation function. After the activation
        function batch-normalization is performed on default, to take care of the covariate shift. As default
        the padding is set to same, to avoid difficulties with convolution.

        + param **_input**: data from previous convolutional layer, type `np.ndarray`.
        + param **filterNumber**: multiple of the filters per layer, type `int`.
        + param **alpha**: parameter for `LeakyRelu` activation function, default `5.5`, type `float`.
        + param **kernelSize**: size of the `2D` kernel, default `(2,2)`, type `tuple`.
        + param **kernelInitializer**: keras kernel initializer, default `uniform`, type `str`.
        + param **padding**: padding for convolution, default `same`, type `str`.
        + param **useBias**: whether or not to use the bias term throughout the network, type `bool`.
        + param **biasInitializer**: initializing distribution of the bias values, type `str`.
        + return **data**: processed data by neural layers, type `np.ndarray`.
    """
    _conv = Conv2D(
        filterNumber,
        kernelSize,
        kernel_initializer=kernelInitializer,
        use_bias=useBias,
        bias_initializer=biasInitializer,
        padding=padding,
    )(_input)
    _activ = LeakyReLU(alpha=alpha)(_conv)
    _norm = BatchNormalization()(_activ)
    return _norm


def loop_group(
    group: Callable,
    groupLayers: int,
    element: np.ndarray,
    filterNumber: int,
    kernelSize: tuple,
    useBias: bool = True,
    kernelInitializer: str = "uniform",
    biasInitializer: str = "zeros",
) -> np.ndarray:
    """
        **This callable is a loop over a group specification.**

        The neural embeddings ends always with dimension `1` in the color channel. For other
        specifications use the parameter `colorChannel`. The function operates on every keras
        group of layers using the same parameter set as `2D` convolution.

        + param **group**: a callable that sets up the neural architecture, type `Callable`.
        + param **groupLayers**: depth of the neural network, type `int`.
        + param **element**: data, type `np.ndarray`.
        + param **filterNumber**: number of filters as exponential of `2`, type `int`.
        + param **kernelSize**: size of the kernels, type `tuple`.
        + return **data**: processed data by neural network, type `np.ndarray`.
        + param **useBias**: whether or not to use the bias term throughout the network, type `bool`.
        + param **biasInitializer**: initializing distribution of the bias values, type `str`.
    """
    data = element
    for i in range(1, groupLayers + 1):
        try:
            if i == 1:
                data = group(
                    data,
                    filterNumber=filterNumber ** groupLayers,
                    kernelSize=kernelSize,
                    kernelInitializer=kernelInitializer,
                    useBias=useBias,
                )
            else:
                data = group(
                    data,
                    filterNumber=int((filterNumber ** groupLayers) / (i ** 2)),
                    kernelSize=kernelSize,
                    kernelInitializer=kernelInitializer,
                    useBias=useBias,
                )
        except TypeError:
            exit("TypeError on your convolutional layer size.")
    return data


def invertible_layer(
    data: np.ndarray,
    alpha: float = 5.5,
    kernelSize: tuple = (2, 2),
    kernelInitializer: str = "uniform",
    groupLayers: int = 6,
    filterNumber: int = 2,
    croppingFactor: int = 4,
    useBias: bool = True,
    biasInitializer: str = "zeros",
) -> np.ndarray:
    """
        **Returns an invertible neural network layer.**

        This neural network layer learns invertible subspaces, parameterized by higher dimensional
        functions with a trivial invertibility. The higher dimensional functions are also neural
        subnetworks, trained during learning process.

        + param **data**: data from previous convolutional layer, type `np.ndarray`.
        + param **alpha**: parameter for `LeakyRelu` activation function, default `5.5`, type `float`.
        + param **groupLayers**: depth of the neural network, type `int`.
        + param **kernelSize**: size of the kernels, type `tuple`.
        + param **filterNumber**: multiple of the filters per layer, type `int`.
        + param **croppingFactor**: should be a multiple of the strides length, type `int`.
        + param **useBias**: whether or not to use the bias term throughout the network, type `bool`.
        + param **biasInitializer**: initializing distribution of the bias values, type `str`.
        + return **data**: processed data, type `np.ndarray`.
    """
    data_shape = take_out_element(K.int_shape(data), None)
    cropOne = int(data_shape[0] / croppingFactor)
    cropTwo = int(data_shape[1] / croppingFactor)
    crop = (cropOne, cropTwo)
    partVectorOne = Cropping2D(cropping=((0, 0), crop))(data)
    partVectorTwo = Cropping2D(cropping=(crop, (0, 0)))(data)

    # Storing the shapes for dynamic adaptation of parameters.
    partVectorOneShape = take_out_element(K.int_shape(partVectorOne), None)
    partVectorTwoShape = take_out_element(K.int_shape(partVectorTwo), None)

    # Compute the dimension reduction caused by the convolutional layer.
    sizeReductionPerDimension = []
    for i in range(0, len(kernelSize)):
        sizeReductionPerDimension.append((kernelSize[i] - 1) * groupLayers)

    # First function for invertibility.
    dataDimension = np.prod(np.array(partVectorOneShape))
    dataReduced = dataDimension - np.prod(np.array(sizeReductionPerDimension))
    firstGroup = loop_group(
        group=convolutional_group,
        groupLayers=groupLayers,
        element=partVectorTwo,
        filterNumber=filterNumber,
        kernelSize=kernelSize,
        kernelInitializer=kernelInitializer,
        useBias=useBias,
        biasInitializer=biasInitializer,
    )
    firstMultiplication = multiply(
        [partVectorOne, Reshape(partVectorOneShape)(firstGroup)]
    )
    firstAddition = add([firstMultiplication, Reshape(partVectorOneShape)(firstGroup)])

    # Inverse process of learning.
    secondGroup = loop_group(
        group=convolutional_group,
        groupLayers=groupLayers,
        element=firstAddition,
        filterNumber=filterNumber,
        kernelSize=kernelSize,
        kernelInitializer=kernelInitializer,
        useBias=useBias,
        biasInitializer=biasInitializer,
    )
    secondMultiplication = multiply(
        [partVectorTwo, Reshape(partVectorTwoShape)(firstGroup)]
    )
    secondAddition = add(
        [secondMultiplication, Reshape(partVectorTwoShape)(secondGroup)]
    )

    # Storing the shapes for dynamic adaptation of parameters.
    outOneShape = take_out_element(K.int_shape(firstAddition), None)
    outTwoShape = take_out_element(K.int_shape(secondAddition), None)

    decoded_layer = Concatenate(axis=2)(
        [firstAddition, Reshape(target_shape=outOneShape)(secondAddition)]
    )
    decoded_layer = Reshape(data_shape)(decoded_layer)

    return decoded_layer


def invertible_subspace_dimension2(units: int):

    """
        **A helper function converting dimensions into 2D convolution shapes.**

        This functions works only for quadratic dimension size. It reshapes the data
        according to an embedding with the same dimension, represented by a `2D` array.

        + param **units**: , type `int`.
        + return **embedding**: , type `tuple`.
    """
    embedding = (int(math.sqrt(units)), int(math.sqrt(units)), 1)
    return embedding


def invertible_subspace_autoencoder(
    data: np.ndarray,
    units: int,
    invertibleLayers: int,
    alpha: float = 5.5,
    kernelSize: tuple = (2, 2),
    kernelInitializer: str = "uniform",
    groupLayers: int = 6,
    filterNumber: int = 2,
    useBias: bool = True,
    biasInitializer: str = "zeros",
):
    """
        **A function returning an invertible autoencoder model.**

        This model works only with a quadratic number as units. The convolutional embedding
        dimension in `2D` is determined, for the quadratic matrix, as the square root of the
        respective dimension of the dense layer. This module is for testing purposes and not
        meant to be part of a productive environment.

        + param **data**: data, type `np.ndarray`.
        + param **units**: projection dim. into lower dim. by dense layer, type `int`.
        + param **invertibleLayers**: amout of invertible layers in the middle of the network, type `int`.
        + param **alpha**: parameter for `LeakyRelu` activation function, default `5.5`, type `float`.
        + param **kernelSize**: size of the kernels, type `tuple`.
        + param **kernelInitializer**: initializing distribution of the kernel values, type `str`.
        + param **groupLayers**: depth of the neural network, type `int`.
        + param **filterNumber**: multiple of the filters per layer, type `int`.
        + param **useBias**: whether or not to use the bias term throughout the network, type `bool`.
        + param **biasInitializer**: initializing distribution of the bias values, type `str`.
        + param **filterNumber**: an integer factor for each convolutional layer, type `int`.
        + return **output**: an output layer for keras neural networks, type `np.ndarray`.
    """
    firstLayerFlattened = Flatten()(data)
    dataDimension = take_out_element(K.int_shape(firstLayerFlattened), None)[0]
    data_shape = take_out_element(K.int_shape(data), None)
    firstLayer = Dense(
        units=units,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    )(firstLayerFlattened)
    firstActivation = LeakyReLU(alpha=alpha)(firstLayer)
    firstNorm = BatchNormalization()(firstActivation)

    firstShape = take_out_element(K.int_shape(firstLayerFlattened), None)

    shape = invertible_subspace_dimension2(units)
    reshapedLayer = Reshape((shape))(firstNorm)

    for i in range(0, invertibleLayers):
        reshapedLayer = add(
            [
                reshapedLayer,
                invertible_layer(
                    data=reshapedLayer,
                    alpha=alpha,
                    kernelSize=kernelSize,
                    kernelInitializer=kernelInitializer,
                    groupLayers=groupLayers,
                    filterNumber=filterNumber,
                    useBias=useBias,
                    biasInitializer=biasInitializer,
                ),
            ]
        )

    lastLayerFlattened = Flatten()(reshapedLayer)
    lastLayer = Dense(
        units=units,
        use_bias=True,
        kernel_initializer="glorot_uniform",
        bias_initializer="zeros",
    )(lastLayerFlattened)
    lastActivation = LeakyReLU(alpha=alpha)(lastLayer)
    lastNorm = BatchNormalization()(lastActivation)
    output = Reshape(data_shape)(lastNorm)
    return output