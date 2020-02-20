#!/usr/bin/env python

import matplotlib.pyplot as plot
import tikzplotlib as tikz
import gudhi as gd
import numpy as np

from typing import *

def persistent_homology(
    data: np.ndarray,
    display: bool = False,
    tikzplot: bool = False,
    maxEdgeLength: int = 42,
    maxDimension: int = 5,
    maxAlphaSquare: float = 1e12,
    homologyCoeffField: int = 3,
    minPersistence: float = 0,
    filtration: str = ["alphaComplex", "vietorisRips", "tangential"],
):
    """
        **Create uniform random sampling of a d-sphere.**

        This algorithm generates a certain set of normally distributed random variables.
        Since the multivariate normal distribution of `(x1, ..., xn)` is rotationally symmetrical about the
        origin, data can be generated on a sphere. The computation time for this algorithm is `O(n * d)`,
        with `n` being the number of samples and `d` the number of dimensions.

        + param **data**: data, type `np.ndarray`.
        + param **display**: whether or not to plot the persistence diagram using matplotlib, type `bool`.
        + param **tikzplot**: whether or not to create a tikz file from persistent homology, type `bool`.
        + param **maxEdgeLength**: maximal edge length of simplicial complex, type `int`.
        + param **maxDimension**: maximal dimension of simplicial complex, type `int`.
        + param **maxAlphaSquare**: alpha square value for Delaunay complex, type `float`.
        + param **homologyCoeffField**: integers, cyclic moduli integers, rationals enumerated, type `int`.
        + param **minPersistence**: minimal persistence of homology class, type `float`.
        + param **filtration**: the used filtration to calculate persistent homology, type `str`.
        + return **np.ndarray**: data points, type `np.ndarray`.
    """
    dataShape = data.shape
    elementSize = len(data[0].flatten())
    reshapedData = data[0].reshape((int(elementSize / 2), 2))

    if filtration == "vietorisRips":
        simComplex = gd.RipsComplex(
            points=reshapedData, max_edge_length=maxEdgeLength
        ).create_simplex_tree(max_dimension=maxDimension)
    elif filtration == "alphaComplex":
        simComplex = gd.AlphaComplex(points=reshapedData).create_simplex_tree(
            max_alpha_square=maxAlphaSquare
        )
    elif filtration == "tangential":
        simComplex = gd.AlphaComplex(
            points=reshapedData, intrinsic_dimension=len(data.shape) - 1
        ).compute_tangential_complex()

    persistenceDiagram = simComplex.persistence(
        homology_coeff_field=homologyCoeffField, min_persistence=minPersistence
    )

    if display == True:
        gd.plot_persistence_diagram(persistenceDiagram)
        plot.show()
    elif tikzplot == True:
        gd.plot_persistence_diagram(persistenceDiagram)
        plot.title("Persistence landscape.")
        tikz.save("persistentHomology_" + filtration + ".tex")

    return persistenceDiagram
