import os
import sys
import inspect
import math

current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import numpy as np
import auxiliary as aux
from typing import Callable


def ts_gaf_transform(
    timeseries: np.ndarray, upper_bound: float = 1.0, lower_bound: float = -1.0
) -> tuple:
    """
        **Compute the Gramian Angular Field of a time series.**

        The Gramian Angular Field is a bijective transformation of time series data into an image of dimension `n+1`.
        Inserting an `n`-dimensional time series gives an `(n x n)`-dimensional array with the corresponding encoded
        time series data.

        + param **timeseries**: time series data, type `np.ndarray`.
        + param **upper_bound**: upper bound for scaling, type `float`.
        + param **lower_bound**: lower bound for scaling, type `float`.
        + return **tuple**: (GAF, phi, r, scaled-series), type `tuple`.
    """

    # Min-Max scaling
    min_ = np.amin(serie)
    max_ = np.amax(serie)
    scaled_serie = (2 * serie - max_ - min_) / (max_ - min_)

    # Floating point inaccuracy!
    scaled_serie = np.where(scaled_serie >= 1.0, 1.0, scaled_serie)
    scaled_serie = np.where(scaled_serie <= -1.0, -1.0, scaled_serie)

    # Polar encoding
    phi = np.arccos(scaled_serie)

    # Note! The computation of r is not necessary
    r = np.linspace(0, 1, len(scaled_serie))

    # GAF Computation (every term of the matrix)
    gaf = tabulate(phi, phi, cos_sum)

    return (gaf, phi, r, scaled_serie)
