Help on module compute_imagets:

NAME
    compute_imagets

FUNCTIONS
    transform(timeseries: numpy.ndarray)
        Compute the Gramian Angular Field of a time series.
        
        The Gramian Angular Field is a bijective transformation of time series data into an image of dimension `n+1`.
        Inserting an `n`-dimensional time series gives an `(n x n)`-dimensional array with the corresponding encoded
        time series data.
        
        Parameter | Usage | Data type
        --- | --- | ---
        `timeseries` | `n`-dimensional time series data | NumPy ndarray

FILE
    /home/lume/Dokumente/Ajin/cluster/compute_imagets.py


