# tda

# tda.homologicalSampling

## sample_dsphere
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

## sample_dball
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

## sample_dtorus_cursed
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

## sample_torus
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

# tda.persistenceHomology

## persistent_homology
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

# tda.persistenceLandscapes

## concatenate_landscapes
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

## compute_persistence_landscape
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

## compute_mean_persistence_landscapes
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

# tda.persistenceStatistics

## hausd_interval
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

## truncated_simplex_tree
```python
truncated_simplex_tree(simplexTree: numpy.ndarray, int_trunc: int = 100) -> tuple
```

**This function return a truncated simplex tree.**

A sparse representation of the persistence diagram in the form of a truncated
persistence tree. Speeds up computation on large scale data sets.

+ param **simplexTree**: simplex tree, type `np.ndarray`.
+ param **int_trunc**: number of persistent interval kept per dimension, default is `100`, type `int`.
+ return **simplexTreeTruncatedPersistence**: truncated simplex tree, type `np.ndarray`.
