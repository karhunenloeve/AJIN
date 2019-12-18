import math
import numpy as np

from typing import Callable


def ts_iterative_descent(data: np.ndarray, function: Callable) -> np.ndarray:
    """
        **Iterative process an `np.ndarray` of shape `(m,n)`.**

        This function processes an `np.ndarray` filled columnwise with time series data. We consider an `(m,n)`-dimensional
        array and perform the callable over the `m`th row of the dataset. Our result is an `np.ndarray` with dimension `(m,l)`.
        This function treats the row vectors as time series. Therefore the time series must be ordered by the first index `m`.

        + param **data**: multidimensional data, type `np.ndarray`.
        + param **function**: callable, type `Callable`.
        + return **proc_data**: all kind of processed data.
        """
    try:
        proc_data = []
        for i in range(0, data.shape[0]):
            proc_data.append(function(data[i]))
        return np.array(proc_data)
    except TypeError:
        print("Wrong size of np.ndarray.")


def ts_recursive_descent(data: np.ndarray, function: Callable):
    """
        **Recursivly process an `np.ndarray` until the last dimension.**

        This function applies a callable to the very last dimension of a numpy multidimensional array. It is foreseen
        for time series processing expecially in combination with the function `ts_gaf_transform`.

        + param **data**: multidimensional data, type `np.ndarray`.
        + param **function**: callable, type `Callable`.
        + return **function(data)**: all kind of processed data.
        """
    if len(data.shape) == 1:
        return function(data)
    for i in range(0, data.shape[0]):
        return ts_recursive_descent(data[i], function)
