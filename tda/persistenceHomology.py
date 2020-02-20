#!/usr/bin/env python

import matplotlib.pyplot as plot
import tikzplotlib as tikz
import gudhi as gd
import numpy as np
import typing

from keras.datasets import cifar10, cifar100, fashion_mnist


def persistent_homology(
    data: np.ndarray,
    display: bool = False,
    maxEdgeLength: int = 42,
    maxDimension: int = 5,
    maxAlphaSquare: float = 1e12,
    homologyCoeffField: int = 3,
    minPersistence: int = 0,
    filtration: str = ["alphaComplex", "vietorisRips", "tangential"],
):

    dataShape = data.shape
    elementSize = len(data[0].flatten())
    reshapedData = data[0].reshape((int(elementSize / 2), 2))

    if filtration == "vietorisRips":
        simComplex = gd.RipsComplex(
            points=reshapedData, max_edge_length=maxEdgeLength
        ).create_simplex_tree(max_dimension=maxDimension)
    elif filtration == "alphaComplex":
        print("Hello world")
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
        tikzplotlib.save("persistentHomology_" + filtration + ".tex")

    return persistenceDiagram


(X, y_train), (x_test, y_test) = cifar100.load_data()
persistent_homology(X[:1000], filtration="alphaComplex", display=True)
