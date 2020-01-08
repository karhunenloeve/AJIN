# homological_sampling

## sample_dsphere
```python
def sample_dsphere(dimension: int, amount: int, radius: float = 1) -> np.ndarray
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
def sample_dball(dimension: int, amount: int, radius: float = 1) -> np.ndarray
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
def sample_dtorus_cursed(dimension: int, amount: int, radii: list) -> np.ndarray
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
