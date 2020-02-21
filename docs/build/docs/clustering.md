# clustering.imageAutoencode

## take_out_element
```python
take_out_element(k: tuple, r) -> tuple
```

**A function taking out specific values.**

+ param **k**: tuple object to be processed, type `tuple`.
+ param **r**: value to be removed, type `int, float, string, None`.
+ return **k2**: cropped tuple object, type `tuple`.

## primeFactors
```python
primeFactors(n)
```

**A function that returns the prime factors of an integer.**

+ param **n**: an integer, type `int`.
+ return **factors**: a list of prime factors, type `list`.

## load_data_keras
```python
load_data_keras(dimensions: tuple, factor: float = 255.0, dataset: str = 'mnist') -> tuple
```

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

## add_gaussian_noise
```python
add_gaussian_noise(data: numpy.ndarray, noise_factor: float = 0.5, mean: float = 0.0, std: float = 1.0) -> numpy.ndarray
```

**A utility function to add gaussian noise to data.**

The purpose of this functions is validating certain models under gaussian noise.
The noise can be added changing the mean, standard deviation and the amount of
noisy points added.

+ param **noise_factor**: amount of noise in percent, type `float`.
+ param **data**: dataset, type `np.ndarray`.
+ param **mean**: mean, type `float`.
+ param **std**: standard deviation, type `float`.
+ return **x_train_noisy**: noisy data, type `np.ndarray`.

## crop_tensor
```python
crop_tensor(dimension: int, start: int, end: int) -> Callable
```

**A utility function cropping a tensor along a given dimension.**

The purpose of this function is to be used for multivariate cropping and to serve
as a procedure for the invertible autoencoders, which need a cropping to make the
matrices trivially invertible, as can be seen in the `Real NVP` architecture.
This procedure works up to dimension `4`.

+ param **dimension**: the dimension of cropping, type `int`.
+ param **start**: starting index for cropping, type `int`.
+ param **end**: ending index for cropping, type `int`.
+ return **Lambda(func)**: Lambda function on the tensor, type `Callable`.

## convolutional_group
```python
convolutional_group(_input: numpy.ndarray, filterNumber: int, alpha: float = 5.5, kernelSize: tuple = (2, 2), kernelInitializer: str = 'uniform', padding: str = 'same', useBias: bool = True, biasInitializer: str = 'zeros')
```

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

## loop_group
```python
loop_group(group: Callable, groupLayers: int, element: numpy.ndarray, filterNumber: int, kernelSize: tuple, useBias: bool = True, kernelInitializer: str = 'uniform', biasInitializer: str = 'zeros') -> numpy.ndarray
```

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

## invertible_layer
```python
invertible_layer(data: numpy.ndarray, alpha: float = 5.5, kernelSize: tuple = (2, 2), kernelInitializer: str = 'uniform', groupLayers: int = 6, filterNumber: int = 2, croppingFactor: int = 4, useBias: bool = True, biasInitializer: str = 'zeros') -> numpy.ndarray
```

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

## invertible_subspace_dimension2
```python
invertible_subspace_dimension2(units: int)
```

**A helper function converting dimensions into 2D convolution shapes.**

This functions works only for quadratic dimension size. It reshapes the data
according to an embedding with the same dimension, represented by a `2D` array.

+ param **units**: , type `int`.
+ return **embedding**: , type `tuple`.

## invertible_subspace_autoencoder
```python
invertible_subspace_autoencoder(data: numpy.ndarray, units: int, invertibleLayers: int, alpha: float = 5.5, kernelSize: tuple = (2, 2), kernelInitializer: str = 'uniform', groupLayers: int = 6, filterNumber: int = 2, useBias: bool = True, biasInitializer: str = 'zeros')
```

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

# clustering.imageTransform

## gramian_angular_field
```python
gramian_angular_field(timeseries: numpy.ndarray, upper_bound: float = 1.0, lower_bound: float = -1.0) -> tuple
```

**Compute the Gramian Angular Field of a time series.**

The Gramian Angular Field is a bijective transformation of time series data into an image of dimension `n+1`.
Inserting an `n`-dimensional time series gives an `(n x n)`-dimensional array with the corresponding encoded
time series data.

+ param **timeseries**: time series data, type `np.ndarray`.
+ param **upper_bound**: upper bound for scaling, type `float`.
+ param **lower_bound**: lower bound for scaling, type `float`.
+ return **tuple**: (GAF, phi, r, scaled-series), type `tuple`.

