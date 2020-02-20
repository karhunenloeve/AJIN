#!/usr/bin/env python

import matplotlib.pyplot as plt
import tikzplotlib as tikz
import gudhi as gd
import numpy as np

from typing import *

def persistent_homology(
    data: np.ndarray,
    plot: bool = False,
    tikzplot: bool = False,
    maxEdgeLength: int = 42,
    maxDimension: int = 10,
    maxAlphaSquare: float = 1e12,
    homologyCoeffField: int = 2,
    minPersistence: float = 0,
    filtration: str = ["alphaComplex", "vietorisRips", "tangential"],
):
    """
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

    if plot == True:
        gd.plot_persistence_diagram(persistenceDiagram)
        plt.show()
    elif tikzplot == True:
        gd.plot_persistence_diagram(persistenceDiagram)
        plt.title("Persistence landscape.")
        tikz.save("persistentHomology_" + filtration + ".tex")

    return persistenceDiagram