#!/usr/bin/python
"""
" @section DESCRIPTION
" Different basis functions that can be used for adding an intensity dimension
"""

import numpy as np
from operator import mul

n_bases = 5


def scaled_original_basis(x):
    """ Scales the original basis to the interval [0, 1]

    :param x:
    :return x_basis:
    """

    x_basis = x.copy()
    x_basis -= x_basis.min()  # min value 0.0
    if x_basis.max() != 0:
        x_basis /= x_basis.max()  # max value 1.0

    if len(x_basis.shape) == 2:
        x_basis = x_basis.reshape(list(x_basis.shape) + [1])

    return x_basis


def negated_scaled_original_basis(x):
    """ Negates and scales the original basis to the interval [0, 1]

    :param x:
    :return x_basis:
    """

    x_basis = x.copy()
    x_basis *= -1
    x_basis -= x_basis.min()  # min value 0.0
    x_basis /= x_basis.max()  # max value 1.0

    if len(x_basis.shape) == 2:
        x_basis = x_basis.reshape(list(x_basis.shape) + [1])

    return x_basis


def binary_basis(x):
    """ Binary basis (low/high)

    :param x:
    :return x_basis:
    """

    # Find the correct shapes
    input_dims = [dim for dim in x.shape if dim > 1]
    new_input_dims = input_dims + [2]
    n_elements = reduce(mul, new_input_dims)

    x_org = x.reshape(input_dims)
    x_mean = x_org.mean()
    x_basis = np.zeros(n_elements).reshape(new_input_dims)
    x_basis[x_org < x_mean , 0] = 1.  # low
    x_basis[x_org > x_mean, 1] = 1.  # high

    return x_basis


def pyramid_basis(x):
    """ Remaps the input array onto a set of pyramid shaped basis functions

    Args:
        x: input array
    Returns:
        x_basis:
    Raises:

    """

    tops = np.linspace(0, 1, n_bases)
    slope = tops[1] - tops[0]

    x_copy = x.copy()
    x_copy -= x_copy.min()  # min value 0.0
    if x_copy.max() != 0:
        x_copy /= x_copy.max()  # max value 1.0

    # Determine dimensions
    new_input_dims = list(x.shape) + [len(tops)]
    n_elements = reduce(mul, new_input_dims)

    x_basis = np.zeros(n_elements).reshape(new_input_dims)
    for basis_fun in range(len(tops)):
        basis_tmp = np.abs(x_copy - tops[basis_fun])
        basis_tmp = 1 - basis_tmp/slope
        basis_tmp[basis_tmp<0] = 0

        if basis_fun == 0:
            basis_tmp[x_copy < tops[basis_fun]] = 1
        elif basis_fun == (len(tops)-1):
            basis_tmp[x_copy > tops[basis_fun]] = 1
        x_basis[:, :, basis_fun] = basis_tmp.reshape(new_input_dims[:-1])

    x_basis = x_basis.reshape(new_input_dims)

    return x_basis


def radial_basis_fun(x):
    """ Remaps the input array onto a set of Gaussian basis functions

    Args:
        x: input array
    Returns:
        x_basis:
    Raises:

    """
    means = np.linspace(0, 1, n_bases)
    std = (means[1] - means[0]) / 2

    x_copy = x.copy()
    x_copy -= x_copy.min()  # min value 0.0
    if x_copy.max() != 0:
        x_copy /= x_copy.max()  # max value 1.0

    # Determine dimensions
    new_input_dims = list(x.shape) + [len(means)]
    n_elements = reduce(mul, new_input_dims)

    x_basis = np.zeros(n_elements).reshape(new_input_dims)
    for basis_fun in range(len(means)):
        basis_tmp = np.exp(- (x_copy - means[basis_fun]) ** 2
                       / 2 / (std ** 2))
        if basis_fun == 0:
            basis_tmp[x_copy < means[basis_fun]] = 1
        elif basis_fun == (len(means) - 1):
            basis_tmp[x_copy > means[basis_fun]] = 1
        x_basis[:, :, basis_fun] = basis_tmp.reshape(new_input_dims[:-1])

    x_basis = x_basis.reshape(new_input_dims)

    return x_basis


def binned_basis_fun(x):
    """ Bins the input array into non-overlapping bins

    Args:
        x: input array
    Returns:
        x_basis:
    Raises:

    """

    edges = np.linspace(0, 1, n_bases + 1)

    x_copy = x.copy()
    x_copy -= x_copy.min()  # min value 0.0
    if x_copy.max() != 0:
        x_copy /= x_copy.max()  # max value 1.0

    # Determine dimensions
    new_input_dims = list(x.shape) + [len(edges)-1]
    n_elements = reduce(mul, new_input_dims)

    x_basis = np.zeros(n_elements).reshape(new_input_dims)
    x_binned = np.digitize(x_copy, edges)
    x_basis[x_binned == 0, 0] = 1
    for basis_fun in range(1, len(edges)):
        x_basis[x_binned == basis_fun, basis_fun-1] = 1
    x_basis[x_binned == len(edges), len(edges)-2] = 1

    x_basis = x_basis.reshape(new_input_dims)

    return x_basis