#!/usr/bin/python
"""
" @section DESCRIPTION
" Helper functions for training and evaluating RF models
"""

import os
import numpy as np
import cPickle as pickle
from scipy.io import loadmat
from scipy.linalg import toeplitz
from sklearn.neighbors import kneighbors_graph
from numpy.lib import stride_tricks
from operator import mul
from cython.rf_cython import cross_corr_c, cf_mat_der_c


def drange(start, stop, step):
    r = start
    while r < stop:
        yield r
        r += step


def add_fake_dimension(org_ndarray, time_win_size):
    """ Rolls a time window over a vector and extract the window content

    Stride_tricks only affect the shape and strides in the array interface.
    The memory footprint is therefore equal for both org_ndarray and
    fake_ndarray.

    Important!!!
    The time dimension in X must be along the first dimension (axis=0)

    Args:
        org_ndarray: vector to roll the window over
        time_win_size: window size in vector elements (time dimension)
    Returns:
        fake_ndarray:
    Raises:

    """
    n_element = org_ndarray.size
    element_size = org_ndarray.itemsize
    input_dims = org_ndarray.shape
    stride_length = 1
    for dims in input_dims[1:]:
        stride_length *= dims

    org_1darray = org_ndarray.ravel()

    shape = (n_element/stride_length - time_win_size + 1,
             time_win_size*stride_length)
    strides = (stride_length*element_size, element_size)
    fake_2darray = stride_tricks.as_strided(org_1darray,
                                              shape=shape,
                                              strides=strides)

    new_shape = [shape[0], time_win_size]
    for dims in input_dims[1:]:
        new_shape.append(dims)

    fake_ndarray = fake_2darray.reshape(new_shape)
    return fake_ndarray


def gaussian_field(shape, origin):
    """ Generates a multi-dimensional Gaussian field

    :param shape:
    :param origin:
    :return:
    """
    cov_inv = np.diag(np.ones(3))
    # cov_inv = np.diag([10. / shape[1], 10. / shape[0], 10. / shape[2]])
    dim0, dim1, dim2 = np.meshgrid(np.arange(shape[1]) - shape[1] / 2 - origin[1],
                                   np.arange(shape[0]) - shape[0] / 2 - origin[0],
                                   np.arange(shape[2]) - shape[2] / 2 - origin[2])
    x = np.vstack([dim0.ravel(), dim1.ravel(), dim2.ravel()])
    tmp = (x * np.dot(cov_inv, x)).sum(axis=0)
    field = np.exp(-0.5 * tmp).reshape(shape)
    field /= field.max()

    return field


def smooth_reg_l(shape):
    """ Smooth regularization using a n-D discrete Laplace operator

    :param shape:
    :return reg_l:
    """

    shape = [dim for dim in shape if dim > 1]

    if len(shape) == 1:
        row = np.concatenate([[-2, 1], np.zeros(shape[0] - 2)])
        reg_l = toeplitz(row)
        reg_l[0, :] = 0
        reg_l[-1, :] = 0
    else:
        if len(shape) == 2:
            dim0, dim1 = np.meshgrid(range(shape[1]), range(shape[0]))
            dim = np.vstack([dim0.ravel(), dim1.ravel()])
        elif len(shape) == 3:
            dim0, dim1, dim2 = np.meshgrid(range(shape[1]),
                                           range(shape[0]),
                                           range(shape[2]))
            dim = np.vstack([dim0.ravel(), dim1.ravel(), dim2.ravel()])

        con_mat = kneighbors_graph(dim.T, 6, mode='distance').toarray()
        con_mat[con_mat > 1] = 0
        connections_per_node = con_mat.sum(axis=0)
        con_mat[con_mat == 1] = -1
        con_mat[np.diag_indices_from(con_mat)] = connections_per_node
        reg_l = con_mat

    return reg_l


def field_part_der(x_nd, field, part_idx):
    """ Field part derivative in multilinear (separable) models

    :param x_nd:
    :param field:
    :param part_idx:
    :return part_der:
    """

    n_parts = len(field.parts)

    # Create the outer product between non-part_idx parts
    cross_idx = range(part_idx) + \
                range(part_idx + 1, n_parts)
    part_cross = outer_product(field.parts, cross_idx)

    # Sum up contributions along other dimensions
    x_axes = range(1, part_idx + 1) + \
             range(part_idx + 2, 2 + part_cross.ndim)
    field_axes = range(len(part_cross.shape))
    part_der = np.tensordot(x_nd, part_cross, axes=(x_axes, field_axes))

    return part_der


def sta_and_stc(x_2d, y):
    """ Calculate the STA and the STC

    Args:
        x_2d: input array (assumed to have zero mean)
        y: output array
    Returns:
        sta:
        stc:
    Raise

    """

    # Select the spike triggered ensemble
    x_2d_ste = x_2d[y.ravel() > 0, :]

    # STA
    yx_2d_ste = x_2d_ste * y[y > 0, None]
    sta = np.sum(yx_2d_ste, axis=0) / y.sum()

    # STC
    # Remove the STA
    x_2d_ste -= sta
    yx_2d_ste = x_2d_ste * y[y > 0, None]
    stc = np.dot(yx_2d_ste.T, x_2d_ste) / (y.sum()-1)

    return sta, stc


def get_insignificant_basis(x, y, rf_shape):

    # Make a 2D matrix
    x_nd = add_fake_dimension(x, rf_shape[0])
    x_nd_full = x_nd.copy()
    n_samples = x_nd_full.shape[0]
    rf_size = reduce(mul, rf_shape)
    x_2d = x_nd_full.reshape(n_samples, rf_size)

    # Mean center and whiten
    x_2d -= x_2d.mean(axis=0)
    x_2d /= x_2d.std(axis=0)

    _, stc = sta_and_stc(x_2d, y)

    eig_val, eig_vec = np.linalg.eig(stc)
    sort_idxs = np.argsort(eig_val)

    n_zero_val = (np.abs(eig_val) < 1e-10).sum()
    middle_idx = (sort_idxs.size - n_zero_val) / 2 + n_zero_val

    # insignificant_basis = np.real(eig_vec[:, sort_idxs[middle_idx]])
    # rf = insignificant_basis.reshape(rf_shape)
    # return rf

    rfs = []
    for i in range(-2, 3, 1):
        insignificant_basis = np.real(eig_vec[:, sort_idxs[middle_idx + i]])
        rfs.append(insignificant_basis.reshape(rf_shape))

    return rfs


def scale_params(params):

    for cf_id in range(len(params.cfs)):

        scale_factor = 1 / params.cfs[cf_id].bias
        params.rfs[0].field[params.context_map == cf_id] /= scale_factor
        params.cfs[cf_id].field *= scale_factor
        params.cfs[cf_id].bias *= scale_factor

    return params


def outer_product(parts, cross_idx=[]):
    """ Calculates an outer product between 1 to 3 vectors

    Args:
        parts: list with vectors
        cross_idx: indices indicating which vectors to multiply
    Returns:
        part_cross
    Raise
        Exception if more than three parts
    """
    # If part_cross is empty we use all vecotrs
    if len(cross_idx) == 0:
        cross_idx = range(len(parts))

    # Outer product between selected vectors
    if len(cross_idx) == 1:
        part_cross = parts[cross_idx[0]]
    elif len(cross_idx) == 2:
        if parts[cross_idx[0]].ndim == parts[cross_idx[1]].ndim:
            part_cross = np.outer(parts[cross_idx[0]], parts[cross_idx[1]])
        else:
            part_cross = parts[cross_idx[0]][:, np.newaxis, np.newaxis] * \
                         parts[cross_idx[1]]
    elif len(cross_idx) == 3:
        part_cross = parts[cross_idx[0]][:, np.newaxis, np.newaxis] * \
                     np.outer(parts[cross_idx[1]], parts[cross_idx[2]])
    else:
        raise Exception("Can only handle max 3 parts")

    return part_cross


def inner_product(x_nd, rfs):
    """ Calculates the inner product between between multidimensional arrays

    This function calculates a generalized multidimensional euclidean inner
    product using numpy.tensordot as numpy.dot can't handle multidimensional
    matrices. The inner product is calculated for each provided receptive field
    and stored columnwise in the matrix inner_product

    Args:
        x_nd: multidimensional input array
        rfs: list with receptive fields
    Returns:
        inner_product_nd:
    Raise
    """

    # Stores the inner product from each receptive field in separate columns
    inner_product_nd = np.empty([x_nd.shape[0], len(rfs)])
    for rf, rf_idx in zip(rfs, range(len(rfs))):
        # Inner product
        x_axes = range(1, len(x_nd.shape))
        rf_axes = range(len(rf.shape))
        inner_product_nd[:, rf_idx] = np.tensordot(x_nd,
                                                   rf.field,
                                                   axes=(x_axes, rf_axes))
        # Check whether this is a quadratic filter
        if hasattr(rf, 'qn_square') and rf.qn_square:
            inner_product_nd[:, rf_idx] *= \
                rf.qn_lambda * inner_product_nd[:, rf_idx]
        # Add the filter's bias term
        inner_product_nd[:, rf_idx] += rfs[rf_idx].bias

    return inner_product_nd


def cross_corr(x, rf):
    """ Calculates the cross-correlation between x and rf

        Computes the cross-correlation between x and rf without the need to
        create a large input matrix by adding a fake dimension.

        The function is a python wrapper for the cython function:
        cross_corr_c()

        Args:
            x: input array
            rf: receptive field
        Returns:
            z: similarity score
        Raise
        """

    win_size = rf.field.size
    stride = reduce(mul, x.shape[1:])
    n_vals = x.shape[0] - rf.shape[0] + 1

    z = np.empty(n_vals)
    z[:] = cross_corr_c(x.ravel(), rf.field.ravel(), n_vals, stride, win_size)
    # z += rf.bias

    return z


def cf_mat_der(x, e, rf):

    win_size = rf.field.size
    stride = reduce(mul, x.shape[1:])
    n_vals = x.shape[0] - rf.shape[0] + 1

    cf_der_sum = np.zeros(win_size)
    cf_der_sum[:] = cf_mat_der_c(x.ravel(), e.ravel(), rf.field.ravel(), n_vals, stride, win_size)
    cf_der_sum = cf_der_sum / n_vals

    return cf_der_sum


def z_dist(z, y, n_bins):
    """Approximates the similarity score distributions P(z) and P(z|spike)

    IMPORTANT!
    This function ONLY uses the first two receptive fields in the LN-model

    Args:
        z: similarity score array
        y: spike count array
        n_bins: number of bins to use when approximating the distribution
    Returns:
        p_z: P(z)
        p_z_spike: P(z|spike)
        z_edges: bin edge values
    Raises:
        Exception if z has more than two receptive fields (columns)
    """

    # The histogram range goes linearly between -n_std to + n_std
    n_std = 3

    # scores resulting in one or more spikes
    spike_in_bin = (y > 0).ravel()  # spike indicator vector
    z_spike = z.compress(spike_in_bin, axis=0)

    # We use weights to account for situations were an input caused more
    # than one spike.
    z_edges = []

    # One receptive field
    if z.shape[1] == 1:
        edges = np.linspace(z.mean() - n_std * z.std(),
                            z.mean() + n_std * z.std(), n_bins - 1)
        edges = np.insert(edges, 0, -np.inf)
        edges = np.append(edges, np.inf)

        # P(z)
        z_count, edges = np.histogram(z.ravel(), edges)
        # P(z|spike)
        weights = y[y > 0]
        z_count_spike, edges = np.histogram(z_spike.ravel(),
                                            edges,
                                            weights=weights.ravel())

        z_count = z_count[:, None]
        z_count_spike = z_count_spike[:, None]
        z_edges.append(edges)

    # Two receptive fields
    elif z.shape[1] >= 2:
        edges_row = np.linspace(z[:, 0].mean() - n_std * z[:, 0].std(),
                                z[:, 0].mean() + n_std * z[:, 0].std(),
                                n_bins - 1)
        edges_row = np.insert(edges_row, 0, -np.inf)
        edges_row = np.append(edges_row, np.inf)

        edges_col = np.linspace(z[:, 1].mean() - n_std * z[:, 1].std(),
                                z[:, 1].mean() + n_std * z[:, 1].std(),
                                n_bins - 1)
        edges_col = np.insert(edges_col, 0, -np.inf)
        edges_col = np.append(edges_col, np.inf)

        # P(z)
        z_count, edges_row, edges_col = \
            np.histogram2d(z[:, 0].ravel(),
                           z[:, 1].ravel(),
                           [edges_row, edges_col])
        # P(z|spike)
        weights = y[y > 0]
        z_count_spike, edges_row, edges_col = \
            np.histogram2d(z_spike[:, 0].ravel(),
                           z_spike[:, 1].ravel(),
                           [edges_row, edges_col],
                           weights=weights)

        z_edges.append(edges_row)
        z_edges.append(edges_col)

    if z.shape[1] > 2:
        print "Warning! Probability distributions are only evaluated using " \
              "the first two filters in LN-models with more than two filters."

    p_z = np.float64(z_count) / np.sum(z_count)
    p_z_spike = np.float64(z_count_spike) / np.sum(z_count_spike)

    # Manipulates the last score bin edge to make sure that also the
    # largest score falls into the last bin
    for dim in range(len(z_edges)):
        z_edges[dim][-1] += 1e-10

    return p_z, p_z_spike, z_edges


def calculate_r(vec_1, vec_2):
    """ Calculates the pearson r correlation coefficient

    Args:
        vec_1: first vector
        vec_2: second vector
    Returns:

    Raises:

    """

    # Make sure the both vectors are one-dimensional
    vec_1 = vec_1.ravel()
    vec_2 = vec_2.ravel()

    # The following should be equal to scipy.stats.pearsonr
    r = np.mean((vec_1 - np.mean(vec_1)) * (vec_2 - np.mean(vec_2))) / np.std(vec_1) / np.std(vec_2)

    return r


def load_mat_dat_file(file_name):
    """ Load simulated or recorded data

    :param file_name: file name including path
    :return data:
    """
    # Separate behaviour for pickled Python *.dat files
    if file_name[-3:] == 'dat':
        data = pickle.load(open(file_name, 'rb'))
    # and Matlab *.mat files
    elif file_name[-3:] == 'mat':
        data_mat = loadmat(file_name)
        data = {'x': np.float64(data_mat['x']),
                'x_labels': [label[0] for label in data_mat['x_labels'][0]],
                'x_ticks': [ticks.tolist() for ticks in data_mat['x_ticks'][0]],
                'y': np.float64(data_mat['y']),
                'name': data_mat['name'][0],
                'origin': data_mat['origin'][0],
                'params': {'dt': data_mat['dt_ms'][0, 0]}
                }
    else:
        raise Exception("Unknown file format: {}".format(file_name[-3:]))

    return data


def load_saved_models(load_path, tag=None):
    """ Load saved rf models in specified directory

    :param load_path:
    :return:
    """

    models = []

    if load_path is not None:
        if os.path.isdir(load_path):
            contents = os.listdir(load_path)
            # Filter by tag
            if tag is not None:
                contents = [s for s in contents if tag in s]

            for file_name in sorted(contents):
                # Assume that all *.dat files are saved models
                if file_name[-3:] == 'dat':
                    model = pickle.load(open(load_path + file_name, 'rb'))
                    models.append(model)
        else:
            print "Provided model path does not exist!"
    else:
        print "No model path provided!"

    return models


def load_saved_models_old(results_path, result_files=[]):
    """ Read pickled models

    Args:
        results_path: path to results folder
        result_files: stored files to read
    Returns:
        all_fields: rfs and cfs in all files
        all_simulation_data: simulation data form all files
    Raises:

    """

    all_fields = []  # STRF, CF, and r-values
    all_simulation_data = []  # Configuration used

    # Load all files with a *.dat extension if no file names are provided
    if len(result_files) == 0:
        for file in os.listdir(results_path):
            if file.endswith(".dat"):
                result_files.append(file)

    for result_file in result_files:

        with open(results_path+result_file, 'rb') as handle:
            results = pickle.load(handle)

        n_models = len(results['models'])
        rfs = []
        rf_names =[]
        cfs = []
        cf_names = []
        r_train = []
        r_test = []
        obj_fun_val = []
        for i in range(n_models):
            name = results['models'][i].name
            if name.rfind('_') >= 0:
                name = name[0:name.rfind('_')]
            else:
                name += str(len(results['models'][i].rfs))
            for rf in results['models'][i].rfs:
                if len(rf) > 0:
                    # rf_tmp = rf['field']/np.linalg.norm(rf['field'])
                    rf_tmp = rf['field']
                    rfs.append(rf_tmp)
                    rf_names.append(name)
            for cf in results['models'][i].cfs:
                if len(cf) > 0:
                    cfs.append(cf['field'][::-1, ::-1, ::-1])
                    cf_names.append(name)

            r_train.append(results['models'][i].r_train)
            r_test.append(results['models'][i].r_test)
            obj_fun_val.append(results['models'][i].obj_fun_val)

        tmp_dic = {'rfs': rfs,
                   'rf_names': rf_names,
                   'cfs': cfs,
                   'cf_names': cf_names,
                   'r_train': r_train,
                   'r_test': r_test,
                   'obj_fun_val': obj_fun_val}
        all_fields.append(tmp_dic)
        all_simulation_data.append(results['simulation_data'])

    return all_fields, all_simulation_data
