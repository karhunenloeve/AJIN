# Ajin
[![License](https://img.shields.io/:license-mit-blue.svg)](https://badges.mit-license.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/python/black)

1. [homologicalSampling]

## homologicalSampling

### sample_dsphere
```python
sample_dsphere(dimension: int, amount: int, radius: float = 1) -> numpy.ndarray
```

**Create uniform random sampling of a d-sphere.**

This algorithm generates a certain set of normally distributed random variables.
Since the multivariate normal distribution of `(x1, ..., xn)` is rotationally symmetrical about the
origin, data can be generated on a sphere. The computation time for this algorithm is `O(n * d)`,
with `n` being the number of samples and `d` the number of dimensions.

+ param **dimension**: as dimension of the embedding space, type `int`.
+ param **amount**: amount of sample points, type `float`.
+ param **radius**: radius of the d-sphere, type `float`.
+ return **np.ndarray**: data points, type `np.ndarray`.

### sample_dball
```python
sample_dball(dimension: int, amount: int, radius: float = 1) -> numpy.ndarray
```

**Sample from a d-ball by drop of coordinates.**

Similar to the sphere, values are randomly assigned to each dimension dimension from a certain interval
evenly distributed. Since the radius can be determined via the norm of the boundary points, these
is also the parameter for the maximum radius. Note that there will no points be sampled on the boundary itself.
The computation time for this algorithm is `O(n * d)`, with `n` being the number of samples
and `d` the number of dimensions.

+ param **dimension**: as dimension of the embedding space, type `int`.
+ param **amount**: amount of sample points, type `float`.
+ param **radius**: radius of the d-sphere, type `float`.
+ return **np.ndarray**: data points, type `np.ndarray`.

### sample_dtorus_cursed
```python
sample_dtorus_cursed(dimension: int, amount: int, radii: list) -> numpy.ndarray
```

**Sample from a d-torus by rejection.**

The function is named cursed, because the curse of dimensionality leads to an exponential grouth in time.
The samples are drawn and then rejected if the lie on the algebraic variety of the torus. Unfortunately
the curse of dimensionality makes the computation time exponential in the number of dimensions. Therefore
this is just a prototype for low dimensional sampling

+ param **dimension**: as dimension of the embedding space, type `int`.
+ param **amount**: amount of sample points, type `float`.
+ param **radii**: radii of the torical spheres, type `list`.
+ return **np.ndarray**: data points, type `np.ndarray`.

### sample_torus
```python
sample_torus(dimension: int, amount: int, radii: list) -> numpy.ndarray
```

**Sample from a d-torus.**

The function is named cursed, because the curse of dimensionality leads to an exponential grouth in time.
The samples are drawn and then rejected if the lie on the algebraic variety of the torus. Unfortunately
the curse of dimensionality makes the computation time exponential in the number of dimensions. Therefore
this is just a prototype for low dimensional sampling

+ param **dimension**: as dimension of the embedding space, type `int`.
+ param **amount**: amount of sample points, type `float`.
+ param **radii**: radii of the torical spheres, type `list`.
+ return **list**: data points, type `list`.

## persistenceHomology

### persistent_homology
```python
persistent_homology(data: numpy.ndarray, plot: bool = False, tikzplot: bool = False, maxEdgeLength: int = 42, maxDimension: int = 10, maxAlphaSquare: float = 1000000000000.0, homologyCoeffField: int = 2, minPersistence: float = 0, filtration: str = ['alphaComplex', 'vietorisRips', 'tangential'])
```

**Create persistence diagram.**

This function computes the persistent homology of a dataset upon a filtration of a chosen
simplicial complex. It can be used for plotting or scientific displaying of persistent homology classes.

+ param **data**: data, type `np.ndarray`.
+ param **plot**: whether or not to plot the persistence diagram using matplotlib, type `bool`.
+ param **tikzplot**: whether or not to create a tikz file from persistent homology, type `bool`.
+ param **maxEdgeLength**: maximal edge length of simplicial complex, type `int`.
+ param **maxDimension**: maximal dimension of simplicial complex, type `int`.
+ param **maxAlphaSquare**: alpha square value for Delaunay complex, type `float`.
+ param **homologyCoeffField**: integers, cyclic moduli integers, rationals enumerated, type `int`.
+ param **minPersistence**: minimal persistence of homology class, type `float`.
+ param **filtration**: the used filtration to calculate persistent homology, type `str`.
+ return **np.ndarray**: data points, type `np.ndarray`.

## tda.persistenceLandscapes

### concatenate_landscapes
```python
concatenate_landscapes(persLandscape1: numpy.ndarray, persLandscape2: numpy.ndarray, resolution: int) -> list
```

**This function concatenates the persistence landscapes according to homology groups.**

The computation of homology groups requires a certain resolution for each homology class.
According to this resolution the direct sum of persistence landscapes has to be concatenated
in a correct manner, such that the persistent homology can be plotted according to the `n`-dimensional
persistent homology groups.

+ param **persLandscape1**: persistence landscape, type `np.ndarray`.
+ param **persLandscape2**: persistence landscape, type `np.ndarray`.
+ return **concatenatedLandscape**: direct sum of persistence landscapes, type `list`.

### compute_persistence_landscape
```python
compute_persistence_landscape(data: numpy.ndarray, res: int = 1000, persistenceIntervals: int = 1, maxAlphaSquare: float = 1000000000000.0, filtration: str = ['alphaComplex', 'vietorisRips', 'tangential'], maxDimensions: int = 10, edgeLength: float = 0.1, plot: bool = False, smoothen: bool = False, sigma: int = 3) -> numpy.ndarray
```

**A function for computing persistence landscapes for 2D images.**

This function computes the filtration of a 2D image dataset, the simplicial complex,
the persistent homology and then returns the persistence landscape as array. It takes
the resolution of the landscape as parameter, the maximum size for `alphaSquare` and
options for certain filtrations.

+ param **data**: data set, type `np.ndarray`.
+ param **res**: resolution, default is `1000`, type `int`.
+ param **persistenceIntervals**: interval for persistent homology, default is `1e12`,type `float`.
+ param **maxAlphaSquare**: max. parameter for delaunay expansion, type `float`.
+ param **filtration**: alphaComplex, vietorisRips, cech, delaunay, tangential, type `str`.
+ param **maxDimensions**: only needed for VietorisRips, type `int`.
+ param **edgeLength**: only needed for VietorisRips, type `float`.
+ param **plot**: whether or not to plot, type `bool`.
+ param **smoothen**: whether or not to smoothen the landscapes, type `bool`.
+ param **sigma**: smoothing factor for gaussian mixtures, type `int`.
+ return **landscapeTransformed**: persistence landscape, type `np.ndarray`.

### compute_mean_persistence_landscapes
```python
compute_mean_persistence_landscapes(data: numpy.ndarray, resolution: int = 1000, persistenceIntervals: int = 1, maxAlphaSquare: float = 1000000000000.0, filtration: str = ['alphaComplex', 'vietorisRips', 'tangential'], maxDimensions: int = 10, edgeLength: float = 0.1, plot: bool = False, tikzplot: bool = False, name: str = 'persistenceLandscape', smoothen: bool = False, sigma: int = 2) -> numpy.ndarray
```

**This function computes mean persistence diagrams over 2D datasets.**

The functions shows a progress bar of the processed data and takes the direct
sum of the persistence modules to get a summary of the landscapes of the various
samples. Further it can be decided whether or not to smoothen the persistence
landscape by gaussian filter. A plot can be created with `matplotlib` or as
another option for scientific reporting with `tikzplotlib`, or both.

Information: The color scheme has 5 colors defined. Thus 5 homology groups can be
displayed in different colors.

+ param **data**: data set, type `np.ndarray`.
+ param **resolution**: resolution of persistent homology per group, type `int`.
+ param **persistenceIntervals**: intervals for persistence classes, type `int`.
+ param **maxAlphaSquare**: max. parameter for Delaunay expansion, type `float`.
+ param **filtration**: `alphaComplex`, `vietorisRips` or `tangential`, type `str`.
+ param **maxDimensions**: maximal dimension of simplices, type `int`.
+ param **edgeLength**: length of simplex edge, type `float`.
+ param **plot**: whether or not to plot, type `bool`.
+ param **tikzplot**: whether or not to plot as tikz-picture, type `bool`.
+ param **name**: name of the file to be saved, type `str`.
+ param **smoothen**: whether or not to smoothen the landscapes, type `bool`.
+ param **sigma**: smoothing factor for gaussian mixtures, type `int`.
+ return **meanPersistenceLandscape**: mean persistence landscape, type `np.ndarray`.

## persistenceStatistics

### hausd_interval
```python
hausd_interval(data: numpy.ndarray, confidenceLevel: float = 0.95, subsampleSize: int = -1, subsampleNumber: int = 1000, pairwiseDist: bool = False, leafSize: int = 2, ncores: int = 2) -> float
```

**Computation of Hausdorff distance based confidence values.**

Measures the confidence between two persistent features, wether they are drawn from
a distribution fitting the underlying manifold of the data. This function is based on
the Hausdorff distance between the points.

+ param **data**: a data set, type `np.ndarray`.
+ param **confidenceLevel**: confidence level, default `0.95`, type `float`.
+ param **subsampleSize**: size of each subsample, type `int`.
+ param **subsampleNumber**: number of subsamples, type `int`.
+ param **pairwiseDist**: if `true`, a symmetric `nxn`-matrix is generated out of the data, type `bool`.
+ param **leafSize**: leaf size for KDTree, type `int`.
+ param **ncores**: number of cores for parallel computing, type `int`.
+ return **confidence**: the confidence to be a persistent homology class, type `float`.

### truncated_simplex_tree
```python
truncated_simplex_tree(simplexTree: numpy.ndarray, int_trunc: int = 100) -> tuple
```

**This function return a truncated simplex tree.**

A sparse representation of the persistence diagram in the form of a truncated
persistence tree. Speeds up computation on large scale data sets.

+ param **simplexTree**: simplex tree, type `np.ndarray`.
+ param **int_trunc**: number of persistent interval kept per dimension, default is `100`, type `int`.
+ return **simplexTreeTruncatedPersistence**: truncated simplex tree, type `np.ndarray`.


## imageAutoencode

### take_out_element
```python
take_out_element(k: tuple, r) -> tuple
```

**A function taking out specific values.**

+ param **k**: tuple object to be processed, type `tuple`.
+ param **r**: value to be removed, type `int, float, string, None`.
+ return **k2**: cropped tuple object, type `tuple`.

### primeFactors
```python
primeFactors(n)
```

**A function that returns the prime factors of an integer.**

+ param **n**: an integer, type `int`.
+ return **factors**: a list of prime factors, type `list`.

### load_data_keras
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

### add_gaussian_noise
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

### crop_tensor
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

### convolutional_group
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

### loop_group
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

### invertible_layer
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

### invertible_subspace_dimension2
```python
invertible_subspace_dimension2(units: int)
```

**A helper function converting dimensions into 2D convolution shapes.**

This functions works only for quadratic dimension size. It reshapes the data
according to an embedding with the same dimension, represented by a `2D` array.

+ param **units**: , type `int`.
+ return **embedding**: , type `tuple`.

### invertible_subspace_autoencoder
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

## imageTransform

### gramian_angular_field
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


## auxiliary

### ***aux_iterative_descent***
```python
def aux_iterative_descent(data:numpy.ndarray, function:Callable) -> numpy.ndarray
```

**Iterative process an `np.ndarray` of shape `(m,n)`.**

This function processes an `np.ndarray` filled columnwise with time series data. We consider an `(m,n)`-dimensional
array and perform the callable over the `m`th row of the dataset. Our result is an `np.ndarray` with dimension `(m,l)`.
This function treats the row vectors as time series. Therefore the time series must be ordered by the first index `m`.

+ param **data**: multidimensional data, type `np.ndarray`.
+ param **function**: callable, type `Callable`.
+ return **proc_data**: all kind of processed data.

### ***aux_recursive_descent***
```python
def aux_recursive_descent(data:numpy.ndarray, function:Callable)
```

**Recursivly process an `np.ndarray` until the last dimension.**

This function applies a callable to the very last dimension of a numpy multidimensional array. It is foreseen
for time series processing expecially in combination with the function `ts_gaf_transform`.

+ param **data**: multidimensional data, type `np.ndarray`.
+ param **function**: callable, type `Callable`.
+ return **function(data)**: all kind of processed data.


## Requirements
For some dependencies of our library we need to have installed the `gcc` compiler. Please install `gcc` using one of the following commands for the linux distributions *Arch, Solus4* or *Ubuntu*, or another package manager suitable for your operating system and/or distribution:
```bash
## Archlinux
sudo pacman -S gcc

## Solus4
sudo eopkg install gcc
## These are the requirements to run gcc for Solus4
sudo eopkg install -c system.devel

## Ubuntu
sudo apt update
sudo apt install build-essential
sudo apt-get install python3-dev
sudo apt-get install manpages-dev
gcc --version
```
Some packages are way easier to install using Anaconda. For the installation on several linux distributions please follow [this link](https://docs.anaconda.com/anaconda/install/linux/). Further the installation of our clustering prototype requires some python packages to be installed. We provide a requirements file, but here is a complete list for manual installation using `pip3` and `python 3`:
```bash
pip3 install pandas
pip3 install sklearn
pip3 install tadasets
pip3 install tensorflow
pip3 install keras
pip3 install matplotlib
pip3 install tikzplotlib ## Converts matplotlib to tikz pictures.
pip3 install hdbscan ## Works only with gcc installed.
```
We'll need the Gudhi Python library. A good way to install Gudhi is to utilize the Anaconda environment. This installation requires a non default channel called `conda-forge`. You can use one of the following commands:
```
## Install Gudhi, easiest installation with Anaconda.
## Gudhi is a library to compute persistent homology.
conda install -c conda-forge gudhi
conda install -c conda-forge/label/cf201901 gudhi 
```
**Information**: Some of the somewhat older Gudhi versions have bugs for the Python bindings. Therefore, please use at least Gudhi in version `3.1.1`, which is also available [here](https://github.com/GUDHI/gudhi-devel/releases/tag/tags/gudhi-release-3.1.1) for download. Instructions for the `C++` installation of Gudhi on your system can be found [here](https://gudhi.inria.fr/doc/latest/installation.html).