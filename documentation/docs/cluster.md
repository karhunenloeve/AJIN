# compute_imagets

## ts_gaf_transform
```python
ts_gaf_transform(timeseries:numpy.ndarray, upper_bound:float=1.0, lower_bound:float=-1.0) -> tuple
```

**Compute the Gramian Angular Field of a time series.**

The Gramian Angular Field is a bijective transformation of time series data into an image of dimension `n+1`.
Inserting an `n`-dimensional time series gives an `(n x n)`-dimensional array with the corresponding encoded
time series data.

+ param **timeseries**: time series data, type `np.ndarray`.
+ param **upper_bound**: upper bound for scaling, type `float`.
+ param **lower_bound**: lower bound for scaling, type `float`.
+ return **tuple**: (GAF, phi, r, scaled-series), type `tuple`.

