import math
import typing
import numpy as np


def transform(timeseries: np.ndarray):
        """  
        **Compute the Gramian Angular Field of a time series.**  
        The Gramian Angular Field is a bijective transformation of time series data into an image of dimension `n+1`.
        Inserting an `n`-dimensional time series gives an `(n x n)`-dimensional array with the corresponding encoded
        time series data.  
        **:param timeseries:** *np.ndarray of time series data.*
        **:return :**
        """
        # Min-Max scaling
        min_ = np.amin(serie)
        max_ = np.amax(serie)
        scaled_serie = (2*serie - max_ - min_)/(max_ - min_)

        # Floating point inaccuracy!
        scaled_serie = np.where(scaled_serie >= 1., 1., scaled_serie)
        scaled_serie = np.where(scaled_serie <= -1., -1., scaled_serie)

        # Polar encoding
        phi = np.arccos(scaled_serie)

        # Note! The computation of r is not necessary
        r = np.linspace(0, 1, len(scaled_serie))

        # GAF Computation (every term of the matrix)
        gaf = tabulate(phi, phi, cos_sum)

        return(gaf, phi, r, scaled_serie)