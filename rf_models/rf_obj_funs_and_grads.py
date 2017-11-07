#!/usr/bin/python
"""
" @section DESCRIPTION
" Objective functions and gradients used with different RF models
"""

import numpy as np

from operator import mul
from time import time
from sklearn.model_selection import KFold
from scipy.ndimage import convolve as im_conv
from scipy.ndimage import correlate as im_corr
from scipy import sparse

from rf_field_format import Field
from rf_piecewise_lin_act_fun import ActFun
from rf_helper import add_fake_dimension, cross_corr, \
    inner_product, outer_product, z_dist, cf_mat_der, field_part_der


def mean_squared_error(x, y, params, weights=np.array([])):
    """ Determine the mean squared error (MSE)

    Args:
        x: input array
        y: output array
        params: field parameters
        weights: sample weights
    Returns:
        mse
    Raise

    """

    mse = _error_fun(x, y, params,
                     error_measure='mse',
                     weights=weights)

    return mse


def neg_log_lik_bernoulli(x, y, params, weights=np.array([])):
    """ Determine the mean negative log likelihood for bernoulli GLM

    Args:
        x: input array
        y: output array
        params: field parameters
        weights: sample weights
    Returns:
        neg_log_lik
    Raise

    """

    neg_log_lik = _error_fun(x, y, params,
                             error_measure='neg_log_lik_bernoulli',
                             weights=weights)

    return neg_log_lik


def neg_log_lik_poisson(x, y, params, weights=np.array([])):
    """ Determine the mean negative log likelihood for poisson GLM

    Args:
        x: input array
        y: output array
        params: field parameters
        weights: sample weights
    Returns:
        mean_neg_log_lik
    Raise

    """

    mean_neg_log_lik = _error_fun(x, y, params,
                                  error_measure='neg_log_lik_poisson',
                                  weights=weights)

    return mean_neg_log_lik


def _error_fun(x, y, params,
               error_measure='mse',
               weights = np.array([])):
    """ Evaluation of different error functions

    Args:
        x: input array
        y: output array
        params: field parameters
        act_fun: linear / logistic / tanh
        error_measure: mse / neg_log_lik
        reg_c: regularization parameter (scales the error)
    Returns:
        error_reg_val
    Raise
        Exception if error_fun is unknown
    """

    # Add a unity weights if none are provided
    if weights.size != y.size:
        weights = np.ones([y.size, 1])

    # Similarity score
    x_nd_full = z_rf_der(x, params)
    z = inner_product(x_nd_full, params.rfs)
    if hasattr(params, 'rf_type') and params.rf_type == 'max_rfs':
        z = z.max(axis=1).reshape(z.shape[0], 1)
    elif hasattr(params, 'rf_type') and params.rf_type == 'qn_rfs':
        z = z.sum(axis=1).reshape(z.shape[0], 1)

    # Measured error
    error_val = _get_error_value(z, y, weights, error_measure)

    # Regularization
    reg_val = _get_reg_value(params)

    return error_val + reg_val


def _get_error_value(z, y, weights, error_measure='mse'):
    """ Calculation of error values for linear, logistic, and poisson regression

        Args:
            z: similarity array
            y: output array
            error_measure: mse / neg_log_lik
        Returns:
            error_val
        Raise
            Exception if the selected error measure is unknown
    """

    # regression error
    if error_measure == 'mse':
        y_hat = z
        se = (y - y_hat) ** 2
        error_val = np.sum(se * weights) / 2
    # logistic regression error
    elif error_measure == 'neg_log_lik_bernoulli':
        spike = y > 0
        y_hat = inv_bernoulli_link(z)
        log_lik = spike * np.log(y_hat) + (1 - spike) * np.log(1 - y_hat)
        error_val = -np.sum(log_lik * weights)
    # poisson regression error (slightly modified link function)
    elif error_measure == 'neg_log_lik_poisson':
        y_hat = inv_poisson_link(z)
        log_lik = (y * np.log(y_hat)) - y_hat
        error_val = -np.sum(log_lik * weights)
    else:
        raise Exception("Unknown error_fun: {}".format(error_measure))

    return error_val


def _get_reg_value(params):
    """ Calculation of regularization values for STRF models

    reg_val = 1/2 * sum(w_i**2)
    for all w_i in both the receptive fields and the context fields.

        Args:
            params: field parameters
        Returns:
            reg_val
        Raise

    """

    reg_val = 0

    # RF parameters
    for rf in params.rfs:
        rf_tmp_reg = 0
        if rf.multilin:
            for part, reg_l in zip(rf.parts, rf.reg_l):
                reg_l_tmp = reg_l.toarray()
                ltl = np.dot(reg_l_tmp.T, reg_l_tmp)
                tmp = np.dot(part.ravel(), ltl)
                rf_tmp_reg += np.dot(tmp, part.ravel())
        else:
            reg_l_tmp = rf.reg_l.toarray()
            ltl = np.dot(reg_l_tmp.T, reg_l_tmp)
            tmp = np.dot(rf.field.ravel(), ltl)
            rf_tmp_reg += np.dot(tmp, rf.field.ravel())
        reg_val += rf_tmp_reg / rf.reg_c

    # CF parameters
    for cf in params.cfs:
        cf_tmp_reg = 0
        if cf.multilin:
            for part, reg_l in zip(cf.parts, cf.reg_l):
                reg_l_tmp = reg_l.toarray()
                ltl = np.dot(reg_l_tmp.T, reg_l_tmp)
                tmp = np.dot(part.ravel(), ltl)
                cf_tmp_reg += np.dot(tmp, part.ravel())
        else:
            reg_l_tmp = cf.reg_l.toarray()
            ltl = np.dot(reg_l_tmp.T, reg_l_tmp)
            tmp = np.dot(cf.field.ravel(), ltl)
            cf_tmp_reg += np.dot(tmp, cf.field.ravel())
        reg_val += cf_tmp_reg / cf.reg_c

    # CF activation function parameters
    if hasattr(params, 'cf_act_funs'):
            for act_fun in params.cf_act_funs:
                reg_l_tmp = act_fun.reg_l.toarray()
                ltl = np.dot(reg_l_tmp.T, reg_l_tmp)
                tmp = np.dot(act_fun.alpha.ravel(), ltl)
                act_fun_tmp_reg = np.dot(tmp, act_fun.alpha.ravel())
                reg_val += act_fun_tmp_reg / act_fun.reg_c

    reg_val *= 0.5

    return reg_val


def z_alpha_der(x, params):
    """ Partial derivatives for the CF activation function's parameters

    :param x: input array
    :param params: RF model parameters
    :return z_alpha_der_nds:
    """

    rf = params.rfs[0]
    rf_win_size = rf.shape[0]
    x_nd = add_fake_dimension(x.copy(), rf_win_size)

    z_alpha_der_nds = []
    rf_tmp = Field(rf.shape, reg_c=None, multilin=False, init_values='zeros')
    for act_fun_idx in range(len(params.cf_act_funs)):

        # Select current activation function and its CF
        cf_act_fun = params.cf_act_funs[act_fun_idx]
        cf = params.cfs[act_fun_idx]

        # Get the context similarity score
        origin = cf.origin
        z_ctx = im_corr(x, cf.field, mode='constant', origin=origin)
        z_ctx += cf.bias
        cf_act_fun.set_limits(z_ctx)

        # Temporary copy of the rf with non-zero values for mapped elements only
        rf_tmp.field[:] = rf.field[:]
        rf_tmp.field[params.context_map != act_fun_idx] = 0

        # Preallocate memory
        n_tents = cf_act_fun.n_tents
        z_alpha_der_nd = np.zeros([x_nd.shape[0], n_tents])

        # Calculate partial derivatives
        for tent_id in range(n_tents):
            if params.cf_type == 'ctx_nl':
                t_ctx_tmp = cf_act_fun.single_tent_fun(z_ctx, tent_id) * x
            elif params.cf_type == 'subunit':
                t_ctx_tmp = cf_act_fun.single_tent_fun(z_ctx, tent_id)
            z_alpha_der_nd[:, tent_id] = cross_corr(t_ctx_tmp, rf_tmp).ravel()

        z_alpha_der_nds.append(z_alpha_der_nd)

    return z_alpha_der_nds


def z_rf_der(x, params):
    """ Calculates the partial derivative for the receptive field

    without context field:
    z = w_rf*x(t)
    dz/drf = x(t)
    with context field:
    ctx = w_cf1 + w_cf2*x(t-1) + w_cf2*x(t-2) + ...
    z = w_rf*ctx*x(t)
    dz/drf = ctx*x(t)

    The context sum is determined by a convolution between the input array (X)
    ant the context field (cf). Make sure to use ndimage.convolve here as this
    is clearly faster than signal.convolve

    Args:
        x: input array
        params: RF model parameters
    Returns:
        z_rf_der_nd:
        context:
    """

    rf_win_size = params.rfs[0].shape[0]

    # Allocate memory in advance
    x_nd = add_fake_dimension(x.copy(), rf_win_size)
    z_rf_der_nd = np.empty_like(x_nd)

    # Default values without a cf
    z_rf_der_nd[:] = x_nd

    # Adjustment all elements associated with a cf
    for cf_idx in range(len(params.cfs)):

        cf = params.cfs[cf_idx]  # selected cf
        rf_pos = params.context_map == cf_idx  # elements mapped to cf

        # Cross-correlation between the input and the CF
        origin = cf.origin
        z_ctx_part = im_corr(x, cf.field, mode='constant', origin=origin)
        z_ctx_part += cf.bias

        # Multiply with the context
        if hasattr(params, 'cf_type') and params.cf_act_funs:
            if params.cf_type == 'ctx':
                x_context_part = z_ctx_part * x
            elif params.cf_type == 'ctx_nl':
                x_context_part = params.cf_act_funs[cf_idx].act_fun(z_ctx_part)
                x_context_part *= x
            elif params.cf_type == 'subunit':
                x_context_part = params.cf_act_funs[cf_idx].act_fun(z_ctx_part)
        else:
            x_context_part = z_ctx_part * x

        z_rf_der_nd[:, rf_pos] = \
            add_fake_dimension(x_context_part, rf_win_size)[:, rf_pos]

    return z_rf_der_nd


def z_cf_der(x, params):
    """ Calculates the partial derivative for the context field

    z = w_rf1*act_fun(ctx)x(t) + w_rf2*act_fun(ctx)x(t-1) + ...
    ctx = w_cf1 + w_cf2*x(t-1) + w_cf2*x(t-2) + ...
    dz/dcf = w_rf1*x(t) * dact_fun(ctx)/dctx * dctx/dcf
    dact_fun(ctx)/dctx = depends on the selected function
    dctx/dcf = 1 or x(t-1) or x(t-2) ... depending on i for dcf_i

    Args:
        x: input array
        params: field parameters
    Returns:
        z_cf_der_nd
        z_cf_der_bias
    Raise
        Exception if act_fun is unknown
    """

    assert len(params.rfs) == 1

    rf = params.rfs[0]
    rf_win_size = rf.shape[0]

    z_cf_der_nds = []
    z_cf_der_biases = []
    rf_tmp = Field(rf.shape, reg_c=None, multilin=False, init_values='zeros')
    for cf_idx in range(len(params.cfs)):

        # Assume all CFs have the same shape
        cf = params.cfs[cf_idx]
        cf_shape = cf.shape
        cf_origin = cf.origin
        x_shape = x.shape

        # Initialize temporary input matrices
        x_pad = _get_padded_x(x, params.cfs[cf_idx])
        x_tmp = np.zeros_like(x)

        # Allocate memory in advance
        z_cf_der_nd = np.empty([x.shape[0]-rf_win_size+1] + cf.shape)

        # Temporary copy of the rf with non-zero values for mapped elements only
        rf_tmp.field[:] = rf.field[:]
        rf_tmp.field[params.context_map != cf_idx] = 0

        # Bias term
        if hasattr(params, 'cf_type'):
            if params.cf_type == 'ctx':
                x_tmp[:] = x
            else:
                z_ctx = im_corr(x, cf.field, mode='constant', origin=cf_origin)
                z_ctx += cf.bias
                f_ctx_der = params.cf_act_funs[cf_idx].act_fun_der(z_ctx)
                if params.cf_type == 'ctx_nl':
                    x_tmp[:] = f_ctx_der * x
                elif params.cf_type == 'subunit':
                    x_tmp[:] = f_ctx_der
        else:
            x_tmp[:] = x
        z_cf_der_bias = cross_corr(x_tmp, rf_tmp)
        z_cf_der_bias = z_cf_der_bias.reshape(z_cf_der_bias.size, 1)

        for c0 in range(cf_shape[0]-1, -1, -1):
            for c1 in range(cf_shape[1]):
                for c2 in range(cf_shape[2]):
                    x_tmp[:] = x_pad[c0:c0 + x_shape[0],
                                     c1:c1 + x_shape[1],
                                     c2:c2 + x_shape[2]]
                    if hasattr(params, 'cf_type') and params.cf_act_funs:
                        if params.cf_type == 'ctx':
                            x_tmp *= x
                        elif params.cf_type == 'ctx_nl':
                            x_tmp *= (f_ctx_der * x)
                        elif params.cf_type == 'subunit':
                            x_tmp *= f_ctx_der
                    else:
                        x_tmp *= x
                    z_cf_der_nd[:, c0, c1, c2] = \
                        cross_corr(x_tmp, rf_tmp).ravel()

        # Set origin derivatives to zero
        if params.cfs[cf_idx].zero_origin:
            cf_origin_pos = params.cfs[cf_idx].get_raveled_origin_pos()
            z_cf_der_nd[:,
                        cf_origin_pos[0],
                        cf_origin_pos[1],
                        cf_origin_pos[2]] = 0

        z_cf_der_nds.append(z_cf_der_nd)
        z_cf_der_biases.append(z_cf_der_bias)

    return z_cf_der_nds, z_cf_der_biases


def _get_padded_x(x, cf):
    """ Returns a padded input (x) matrix with wrapped padding

    :param x: original input matrix
    :param cf: context field
    :return x_pad: padded input matrix
    """

    cf_shape = cf.shape
    origin = cf.origin

    # Determine the number of padded elements along each edge
    min_ax_0 = -np.int64(np.floor((cf_shape[0] - 1) / 2.)) + origin[0]
    max_ax_0 = np.int64(np.floor((cf_shape[0]) / 2.)) + origin[0]
    min_ax_1 = -np.int64(np.floor((cf_shape[1]) / 2.))
    max_ax_1 = np.int64(np.floor((cf_shape[1] - 1) / 2.))
    min_ax_2 = -np.int64(np.floor((cf_shape[2]) / 2.))
    max_ax_2 = np.int64(np.floor((cf_shape[2] - 1) / 2.))
    x_padd_dim = [max_ax_0 - min_ax_0 + x.shape[0],
                  max_ax_1 - min_ax_1 + x.shape[1],
                  max_ax_2 - min_ax_2 + x.shape[2]]
    x_padd_size = reduce(mul, x_padd_dim)

    # Initialize the padded input matrix
    x_pad = np.zeros(x_padd_size).reshape(x_padd_dim)

    # Fill center of x_pad with x
    x_pad[max_ax_0:x_pad.shape[0] + min_ax_0,
          -min_ax_1:x_pad.shape[1] - max_ax_1,
          -min_ax_2:x_pad.shape[2] - max_ax_2] = x

    # Wrapped padding is a BAD idea as we do not know how ndimage treats corners
    # # Pad axis 0
    # if max_ax_0 != 0:
    #     x_pad[0:max_ax_0,
    #           -min_ax_1:x_pad.shape[1] - max_ax_1,
    #           -min_ax_2:x_pad.shape[2] - max_ax_2] = x[-max_ax_0:, :, :]
    # if min_ax_0 != 0:
    #     x_pad[min_ax_0:,
    #           -min_ax_1:x_pad.shape[1] - max_ax_1,
    #           -min_ax_2:x_pad.shape[2] - max_ax_2] = x[0:-min_ax_0, :, :]
    #
    # # Pad axis 1
    # if x_pad.shape[1] > 1:
    #     if min_ax_1 != 0:
    #         x_pad[max_ax_0:x_pad.shape[0] + min_ax_0,
    #               0:-min_ax_1,
    #               -min_ax_2:x_pad.shape[2] - max_ax_2] = x[:, min_ax_1:, :]
    #     if max_ax_1 != 0:
    #         x_pad[max_ax_0:x_pad.shape[0] + min_ax_0,
    #               -max_ax_1:,
    #               -min_ax_2:x_pad.shape[2] - max_ax_2] = x[:, 0:max_ax_1, :]
    #
    # # Pad axis 2
    # if x_pad.shape[2] > 1:
    #     if min_ax_2 != 0:
    #         x_pad[max_ax_0:x_pad.shape[0] + min_ax_0,
    #               -min_ax_1:x_pad.shape[1] - max_ax_1,
    #               0:-min_ax_2] = x[:, :, min_ax_2:]
    #     if max_ax_2 != 0:
    #         x_pad[max_ax_0:x_pad.shape[0] + min_ax_0,
    #               -min_ax_1:x_pad.shape[1] - max_ax_1,
    #               -max_ax_2:] = x[:, :, 0:max_ax_2]

    return x_pad


def mean_squared_error_der(x, y, params, weights=np.array([])):
    """ Partial rf and cf derivatives when minimizing the MSE.

    Args:
        x: input array
        y: output array
        params: field parameters
        weights: sample weights
    Returns:
        rf_ders:
        cf_ders:
        act_fun_ders:
    Raise

    """

    rf_ders, cf_ders, act_fun_ders = \
        _error_fun_der(x, y, params,
                       error_measure='mse',
                       weights=weights)

    return rf_ders, cf_ders, act_fun_ders


def neg_log_lik_bernoulli_der(x, y, params, weights=np.array([])):
    """ Partial rf and cf derivatives when minimizing the mean negative
        log likelihood for bernoulli GLM.

    Args:
        x: input array
        y: output array
        params: field parameters
        weights: sample weights
    Returns:
        rf_ders:
        cf_ders:
        act_fun_ders:
    Raise

    """

    rf_ders, cf_ders, act_fun_ders = \
        _error_fun_der(x, y, params,
                       error_measure='neg_log_lik_bernoulli',
                       weights=weights)

    return rf_ders, cf_ders, act_fun_ders


def neg_log_lik_poisson_der(x, y, params, weights=np.array([])):
    """ Partial rf and cf derivatives when minimizing the mean negative
        log likelihood for poisson GLM.

    Args:
        x: input array
        y: output array
        params: field parameters
        weights: sample weights
    Returns:
        rf_ders:
        cf_ders:
        act_fun_ders:
    Raise

    """

    rf_ders, cf_ders, act_fun_ders = \
        _error_fun_der(x, y, params,
                       error_measure='neg_log_lik_poisson',
                       weights=weights)

    return rf_ders, cf_ders, act_fun_ders


def cf_bernoulli_der(x, y, params):
    """ Bernoulli GLM cf derivatives

    This function calculates the derivatives in a non-optimal way so as to
    obtain the specific contributions from each RF element to the full
    derivative.

    Args:
        x: input array
        y: output array
        params: field parameters
    Returns:
        cf_der_mat: matrix with cf elements as rows and rf element
                    contributions as cols.
    Raise

    """

    assert len(params.rfs) == 1

    # Partial derivatives for the RF parameters
    z_rf_der_nd = z_rf_der(x, params)
    # Get the similarity score
    z = inner_product(z_rf_der_nd, params.rfs)
    # Loss function errors
    e = _get_error_der(z, y, 'neg_log_lik_bernoulli')

    rf = params.rfs[0]
    cf_shape = params.cfs[0].shape

    max_ax_0 = cf_shape[0]
    min_ax_1 = -np.int64(np.floor((cf_shape[1] - 1) / 2.))
    max_ax_1 = np.int64(np.floor((cf_shape[1]) / 2.))
    min_ax_2 = -np.int64(np.floor((cf_shape[2] - 1) / 2.))
    max_ax_2 = np.int64(np.floor((cf_shape[2]) / 2.))
    x_padd_dim = [max_ax_0 + x.shape[0],
                  max_ax_1 - min_ax_1 + x.shape[1],
                  max_ax_2 - min_ax_2 + x.shape[2]]
    x_padd_size = reduce(mul, x_padd_dim)

    # Allocate memory in advance
    x_padd = np.zeros(x_padd_size).reshape(x_padd_dim)
    x_padd[max_ax_0:,
           -min_ax_1:x_padd.shape[1] - max_ax_1,
           -min_ax_2:x_padd.shape[2] - max_ax_2] = x
    x_tmp = np.empty_like(x)

    # Mask indicating which RF elements that contribute to CF derivatives
    cf_der_mats = []
    rf_tmp = Field(params.rfs[0].shape, None, False, 'zeros')
    for cf_idx in range(len(params.cfs)):
        cf = params.cfs[cf_idx]

        # Temporary copy of the rf with non-zero values for mapped elements only
        rf_tmp.field[:] = params.rfs[0].field[:]
        if cf_idx == 0:
            selection = (params.context_map != cf_idx) & (params.context_map != -1)
            rf_tmp.field[selection] = 0
        else:
            rf_tmp.field[params.context_map != cf_idx] = 0

        # Allocate memory in advance
        cf_der_mat = np.zeros([cf.field.size + 1, rf.field.size])

        # Bias term
        cf_der_mat[0, :] = -cf_mat_der(x, e, rf_tmp)

        # Context field
        cf_der_mat_tmp = np.zeros(cf.shape + [rf.field.size])
        for c0 in range(0, max_ax_0):
            for c1 in range(max_ax_1 - min_ax_1 + 1):
                for c2 in range(max_ax_2 - min_ax_2 + 1):
                    if c0 == 0:
                        x_tmp[:] = \
                            x_padd[max_ax_0:,
                            c1:x_padd.shape[1] + min_ax_1 - max_ax_1 + c1,
                            c2:x_padd.shape[2] + min_ax_2 - max_ax_2 + c2] * x
                    else:
                        x_tmp[:] = \
                            x_padd[max_ax_0 - c0:-c0,
                            c1:x_padd.shape[1] + min_ax_1 - max_ax_1 + c1,
                            c2:x_padd.shape[2] + min_ax_2 - max_ax_2 + c2] * x

                    cf_der_mat_tmp[max_ax_0-c0-1, c1, c2, :] = \
                        -cf_mat_der(x_tmp, e, rf_tmp)

        cf_der_mat[1:, :] = cf_der_mat_tmp.reshape(cf.field.size, rf.field.size)
        cf_der_mats.append(cf_der_mat)

    return cf_der_mats


def _error_fun_der(x, y, params,
                   error_measure='mse',
                   weights=np.array([])):
    """ Partial rf and cf derivatives for different error functions

    The function returns lists with partial derivatives for both rf and cf
    parameters. The list includes separate entries for all field parts as well
    as the bias term, which is always in the last element of the list.

    Args:
        x: input array
        y: output array
        params: field parameters
        weights: sample weights
        error_measure: mse | neg_log_lik_bernoulli | neg_log_lik_poisson
        reg_c: regularization parameter
    Returns:
        rf_ders:
        cf_ders:
        act_fun_ders:
    Raise

    """

    rf_ders = []
    cf_ders = []
    act_fun_ders = []

    # Add a unity weights if none are provided
    if weights.size != y.size:
        weights = np.ones([y.size, 1])

    # Get the similarity score
    z_rf_der_nd = z_rf_der(x, params)
    z_rf_der_bias = np.ones([z_rf_der_nd.shape[0], 1])
    z = inner_product(z_rf_der_nd, params.rfs)
    if hasattr(params, 'rf_type') and params.rf_type == 'max_rfs':
        z = z.max(axis=1).reshape(z.shape[0], 1)
    elif hasattr(params, 'rf_type') and params.rf_type == 'qn_rfs':
        z = z.sum(axis=1).reshape(z.shape[0], 1)

    # Loss function errors derivatives
    e = _get_error_der(z, y, weights, error_measure)

    # Partial derivatives for rf parameters
    for rf in params.rfs:
        rf_der = _sum_z_field_der(rf, z_rf_der_nd, z_rf_der_bias, e)
        rf_ders.append(rf_der)

    # Partial derivatives for cf parameters
    if params.cfs:
        # Only calculate these when actually used
        z_cf_der_nds, z_cf_der_biases = z_cf_der(x, params)
        for cf_idx in range(len(params.cfs)):
            cf_der = _sum_z_field_der(params.cfs[cf_idx],
                                      z_cf_der_nds[cf_idx],
                                      z_cf_der_biases[cf_idx],
                                      e)
            cf_ders.append(cf_der)

    # Partial derivatives for activation function parameters
    if hasattr(params, 'cf_act_funs') and params.cf_act_funs:
        # Only calculate these when actually used
        z_alpha_der_nds = z_alpha_der(x, params)
        for act_fun_idx in range(len(params.cf_act_funs)):
            act_fun = params.cf_act_funs[act_fun_idx]
            act_fun_der = _sum_z_act_fun_der(
                act_fun, z_alpha_der_nds[act_fun_idx], e)
            act_fun_ders.append(act_fun_der)

    return rf_ders, cf_ders, act_fun_ders


def _get_error_der(z, y, weights, error_measure='mse'):
    """ Derivative of error function with respect to z

        Args:
            z: similarity score
            y: output array
            weights: sample weights
            error_measure: mse | neg_log_lik_bernoulli | neg_log_lik_poisson
        Returns:
            e_der: error derivative for each input vector
        Raise
            Exception if error_fun is unknown
    """

    # regression error
    if error_measure == 'mse':
        e_der = y - z
    # Bernoulli GLM error
    elif error_measure == 'neg_log_lik_bernoulli':
        e_der = _bernoulli_glm_der(z, y)
    # Poisson GLM error
    elif error_measure == 'neg_log_lik_poisson':
        e_der = _poisson_glm_der(z, y)
    else:
        raise Exception("Unknown error_fun: {}".format(error_measure))

    e_der *= weights

    return e_der


def _sum_z_field_der(field, z_field_der_nd, z_bias_der, e):
    """ Sum partial derivative contributions from all inputs

        Args:
            field: field for which the partial derivatives are calculated
            z_field_der_nd: field partial derivatives for each input
            z_bias_der: bias term partial derivatives for each input
            e: loss functions errors for each einput vector
        Returns:
            field_der: field class with partial derivatives for each element
        Raise

        """

    field_der = Field(field.shape, None, field.multilin, init_values='zeros')

    # Parts in a multilinear model
    if field.multilin:

        n_parts = len(field.parts)
        for part_idx in range(n_parts):
            part_der = field_part_der(z_field_der_nd, field, part_idx)

            # Expand e if needed to enable broadcasting
            while e.ndim < part_der.ndim:
                e = np.expand_dims(e, axis=e.ndim)

            # Calculating the partial derivative
            part_der = np.sum(-e * part_der, axis=0)
            part_der += _reg_der(field.parts[part_idx],
                                   field.reg_l[part_idx],
                                   field.reg_c)

            field_der.parts[part_idx] = part_der

    # Every element in a full model
    else:
        # Calculating the partial derivative
        if hasattr(field, 'qn_square') and field.qn_square:
            # This part should belong in z_rf_der() but as this would require
            # z_rf_der() to return multiple large matrices it is
            # implemented here instead.
            x_axes = range(1, len(z_field_der_nd.shape))
            rf_axes = range(len(field.shape))
            z_lin = np.tensordot(z_field_der_nd, field.field,
                                 axes=(x_axes, rf_axes))
            z_lin = z_lin[:, np.newaxis] * e
            field_der_tmp = 2 * field.qn_lambda * \
                            np.sum(-z_lin[:, np.newaxis, np.newaxis] *
                                   z_field_der_nd, axis=0)

        else:
            field_der_tmp = np.sum(-e[:, np.newaxis, np.newaxis] *
                                   z_field_der_nd, axis=0)
        field_der_tmp += _reg_der(field.field, field.reg_l, field.reg_c)
        field_der.field = field_der_tmp

    # Bias term
    e = np.reshape(e, z_bias_der.shape)
    bias_der_tmp = np.sum(-e * z_bias_der, axis=0)
    field_der.bias = bias_der_tmp[0]

    return field_der


def _sum_z_act_fun_der(act_fun, z_alpha_der_nd, e):
    """

    :param act_fun: activation fucntion
    :param z_alpha_der_nd: partial derivatives for each input
    :param e: loss functions errors for each einput vector
    :return act_fun_der: partial derivatie for each alpha value
    """

    act_fun_der = ActFun(reg_c=None)

    alpha_der = np.sum(-e * z_alpha_der_nd, axis=0)
    alpha_der += _reg_der(act_fun.alpha, act_fun.reg_l, act_fun.reg_c)
    act_fun_der.alpha = alpha_der

    return act_fun_der


def _reg_der(param, reg_l, reg_c):
    """ Partial derivatives from the regularization part of objective functions

        Args:
            param: scalar or array of parameters (field elements)
            reg_l: regularization matrix
            reg_c: regularization parameter
        Returns:
            param_reg_der: partial derivatives
        Raise

    """

    if reg_l.size == 0:
        reg_l = sparse.csr_matrix(np.eye(param.size))

    reg_l_tmp = reg_l.toarray()
    reg_ltl = np.dot(reg_l_tmp.T, reg_l_tmp)
    param_reg_der = np.dot(reg_ltl, param.ravel())
    param_reg_der = param_reg_der.reshape(param.shape)
    param_reg_der /= reg_c

    return param_reg_der


def cf_der_and_hessian(x, y, params,
                       error_measure='mse',
                       weights=np.array([])):
    """ Calculate the derivative and Hessian for all CF parameters

    :param x:
    :param y:
    :param params:
    :param error_measure:
    :param reg_c:
    :param weights:
    :return:
    """

    assert len(params.rfs) == 1
    assert params.cfs[0].multilin == False

    cf_ders = []
    cf_hessians = []

    # Add a unity weights if none are provided
    if weights.size != y.size:
        weights = np.ones([y.size, 1])

    # 1. Get the similarity score
    z_rf_der_nd = z_rf_der(x, params)
    z = inner_product(z_rf_der_nd, params.rfs)
    if hasattr(params, 'rf_type') and params.rf_type == 'max_rfs':
        z = z.max(axis=1).reshape(z.shape[0], 1)
    elif hasattr(params, 'rf_type') and params.rf_type == 'qn_rfs':
        z = z.sum(axis=1).reshape(z.shape[0], 1)

    # 2. Derivatives
    e = _get_error_der(z, y, weights, error_measure)
    z_cf_der_nds, z_cf_der_biases = z_cf_der(x, params)
    for cf_idx in range(len(params.cfs)):
        cf_der = _sum_z_field_der(params.cfs[cf_idx],
                                  z_cf_der_nds[cf_idx],
                                  z_cf_der_biases[cf_idx],
                                  e)
        cf_ders.append(cf_der)

    # 3. Hessian
    # 3.1 Sample multiplication factor
    if error_measure == 'mse':
        hess_tmp = np.ones_like(weights)
    elif error_measure == 'neg_log_lik_bernoulli':
        y_hat = inv_bernoulli_link(z)
        hess_tmp = (y_hat * (1 - y_hat))
    elif error_measure == 'neg_log_lik_poisson':
        hess_tmp = inv_poisson_link(z)
    else:
        raise Exception("Unknown error_fun: {}".format(error_measure))

    # 3.2 Reshape the field derivatives into a 2-dimensional matrix
    for cf_idx in range(len(params.cfs)):
        cf = params.cfs[cf_idx]
        cf_size = cf.field.size
        z_cf_der_2d = z_cf_der_nds[cf_idx].reshape(y.size, cf_size)
        z_cf_der_2d = np.hstack([z_cf_der_biases[cf_idx], z_cf_der_2d])

        # 3.3 Calculate the dot product
        hess_tmp *= weights
        hess_tmp = hess_tmp * z_cf_der_2d
        cf_hessian = np.dot(hess_tmp.T, z_cf_der_2d)

        # 3.4 Add the regularization term
        reg_l = cf.reg_l.toarray()
        reg_l = np.insert(reg_l, 0, 0, axis=0)  # bias term
        reg_l = np.insert(reg_l, 0, 0, axis=1)  # bias term
        reg_ltl = np.dot(reg_l, reg_l)
        cf_hessian += (reg_ltl / params.cfs[cf_idx].reg_c)
        cf_hessians.append(cf_hessian)

    return cf_ders, cf_hessians


def inv_bernoulli_link(z):
    y_hat = _sigmoid(z)
    y_hat[y_hat < 1e-15] = 1e-15  # Guard against log(0)
    y_hat[1 - y_hat < 1e-15] = 1 - 1e-15  # Guard against log(0)
    return y_hat


def _bernoulli_glm_der(z, y):
        return (y > 0) - inv_bernoulli_link(z)


def inv_poisson_link(z):
    return np.exp(z)


def _poisson_glm_der(z, y):
    return y - np.exp(z)


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _sigmoid_der(x):
    return _sigmoid(x) * (1 - _sigmoid(x))


def z_rf_pos_inv(x, params):
    """ Calculates the similarity score for position invariant RFs

    Args:
        x: input array
        params: field parameters
    Returns:
        z_pos_inv:
    """

    rf_win_size = params.rfs[0].shape[0]
    z_pos_inv = np.empty([x.shape[0]-rf_win_size+1, len(params.rfs)])

    # Cross-correlate the input with all position invaraint RFs
    for rf_idx in range(len(params.rfs)):

        # Extract maximum similarity scores
        rf = params.rfs[rf_idx]  # selected cf
        origin = rf.origin
        context_part = im_corr(x, rf.field, mode='wrap', origin=origin)
        context_part_nd = add_fake_dimension(context_part, rf_win_size)
        n_samples = context_part_nd.shape[0]
        n_params = reduce(lambda x, y: x * y, context_part_nd.shape[1:])
        context_part_2d = context_part_nd.reshape([n_samples, n_params])
        z_pos_inv[:, rf_idx] = context_part_2d.max(axis=1)

    return z_pos_inv


def z_rf_pos_inv_der(x, params):
    """ Calculates the partial derivatives for position invariant RFs

    NOTE!
    This is trial code for testing purposes that only works for two-dimensional
    stimuli, and it can so far only find one position invariant RF.

    Args:
        x: input array
        params: field parameters
    Returns:
        z_pos_inv:
    """

    assert x.shape[2] == 1
    assert len(params.rfs) == 1

    rf_win_size = params.rfs[0].shape[0]
    z_pos_inv = np.empty([x.shape[0]-rf_win_size+1, len(params.rfs)])

    # Cross-correlate the input with all position invaraint RFs
    for rf_idx in range(len(params.rfs)):

        # Extract maximum similarity scores
        rf = params.rfs[rf_idx]  # selected cf
        origin = rf.origin
        context_part = im_corr(x, rf.field, mode='wrap', origin=origin)
        context_part_nd = add_fake_dimension(context_part, rf_win_size)
        n_samples = context_part_nd.shape[0]
        n_params = reduce(lambda x, y: x * y, context_part_nd.shape[1:])
        context_part_2d = context_part_nd.reshape([n_samples, n_params])
        z_pos_inv[:, rf_idx] = context_part_2d.max(axis=1)

        # Find the index to the input window that elicit the maximum z-values
        z_max_idx = context_part_2d.argmax(axis=1)
        ax0, ax1, ax2 = np.unravel_index(z_max_idx, context_part_nd.shape[1:])
        ax0 = ax0 + np.arange(ax0.size)

        # Find the unique stimuli that caused maximum similarity scores
        mask_all = np.diff(ax0) > 0
        mask_all = np.append(mask_all, True)
        count_all = np.bincount(ax0)
        count_all = count_all[count_all > 0]
        ax0_all = ax0[mask_all]
        ax1_all = ax1[mask_all]
        ax2_all = ax2[mask_all]

        # Create a new full stimulus matrix and fill it with copies of the
        # unique stimuli found above.
        i = 0
        n_pad = max(rf.shape)
        ax0_all += n_pad
        ax1_all += n_pad
        x_pad = np.pad(x[:, :, 0], n_pad, 'wrap')
        z_rf_der_nd = np.empty_like(context_part_nd)
        for row, col, count in zip(ax0_all, ax1_all, count_all):
            sample_tmp = x_pad[row - rf.shape[0] / 2:row + (rf.shape[0]+1) / 2,
                               col - rf.shape[1] / 2:col + (rf.shape[1]+1) / 2]
            z_rf_der_nd[i:i+count, :, :, 0] = sample_tmp
            i += count

    return z_rf_der_nd


def mid_mi(x, y, params, n_bins, rnd_state=0):
    """Calculate the mutual information (MI) between spike times and
       receptive field projections

    Args:
        x: input matrix
        y: spike count matrix
        params: field parameters
        n_bins: number of bins to use when approximating the distribution
    Returns:
        mutual_information:
    Raises:

    """

    # Projection scores
    if hasattr(params, 'rf_type') and params.rf_type == 'pos_inv_rfs':
        z = z_rf_pos_inv(x, params)
    else:
        x_nd_full = z_rf_der(x, params)
        z = inner_product(x_nd_full, params.rfs)

    kf = KFold(n_splits=5, shuffle=True, random_state=rnd_state)
    train_idx, val_idx = kf.split(range(y.size)).next()
    z_train = z[train_idx, :]
    z_val = z[val_idx, :]
    y_train = y[train_idx, :]
    y_val = y[val_idx, :]

    mi_train = mutual_information(z_train, y_train, n_bins)[0]
    mi_val = mutual_information(z_val, y_val, n_bins)[0]

    return mi_train, mi_val


def mutual_information(z, y, n_bins):
    """ Calculates the mutual information between z and y

    MI = int(P(score) log2(P(score|spike)/P(score)), dscore)

    :param z: similarity score, 1- or 2-dimensional
    :param y:
    :param n_bins: histogram resolution
    :return mi: mutual infomration
    """

    # Get P(score) and P(score|spike)
    p_z, p_z_spike, _ = z_dist(z, y, n_bins)

    # Determine the mutual information
    prob_frac = np.zeros(p_z.shape)
    no_zero = p_z > 0
    prob_frac[no_zero] = p_z_spike[no_zero] / p_z[no_zero]
    mi = np.sum(p_z_spike[p_z_spike > 0] * np.log2(prob_frac[p_z_spike > 0]))

    n_used_bins = (p_z_spike > 0).sum() + (p_z > 0).sum()

    return mi, n_used_bins


def mid_mi_der(x, y, params, n_bins, rnd_state=0):
    """Calculate the gradient for the mutual information (MI)
    between spike times and receptive field projections.

    delta = d / dz [P(z|spike) / P(z)]
    gradient = int(P(z)[<x|z,spike>-<x|z>]delta, dx)

    Args:
        x: input matrix
        y: spike count matrix
        params: field parameters
        n_bins: number of bins to use when approximating the distribution
    Returns:
        rf_ders:
    Raises:

    """

    n_rfs = len(params.rfs)
    n_cfs = len(params.cfs)
    rf_shape = params.rfs[0].shape
    rf_multilin = params.rfs[0].multilin

    # Projection scores
    if hasattr(params, 'rf_type') and params.rf_type == 'pos_inv_rfs':
        x_nd_full = z_rf_pos_inv_der(x, params)
        z = z_rf_pos_inv(x, params)
    else:
        x_nd_full = z_rf_der(x, params)
        z = inner_product(x_nd_full, params.rfs)

    # Calculate the partial derivatives from training set indices only
    kf = KFold(n_splits=5, shuffle=True, random_state=rnd_state)
    train_idx, _ = kf.split(range(y.size)).next()
    z_train = z[train_idx, :]
    y_train = y[train_idx, :]
    x_nd_full_train = x_nd_full[train_idx, :, :, :]

    # Get P(score), P(score|spike) and the bin edges
    p_z, p_z_spike, z_edges = z_dist(z_train, y_train, n_bins)

    # Estimate the derivative of the probability fraction
    prob_frac_ders = _estimate_prob_frac_der(p_z, p_z_spike, z_edges)

    # Determine the average rf partial derivative in each bin
    if rf_multilin:
        parts_z = []
        parts_z_spike = []
        field = params.rfs[0]
        n_parts = len(field.parts)
        for part_idx in range(n_parts):
            part_der_train = field_part_der(x_nd_full_train, field, part_idx)
            part_z, part_z_spike = _estimate_avg_bin_stimuli(
                part_der_train, z_train, y_train, p_z, z_edges)
            parts_z.append(part_z)
            parts_z_spike.append(part_z_spike)
    else:
        x_nd_z, x_nd_z_spike = _estimate_avg_bin_stimuli(
            x_nd_full_train, z_train, y_train, p_z, z_edges)

    # RF: Integrating over all score bins
    rows = n_bins
    cols = 1 if n_rfs == 1 else n_bins
    rf_ders = []
    for rf_idx in range(n_rfs):
        rf_der = Field(rf_shape, None, rf_multilin, init_values='zeros')
        if rf_multilin:
            for part_idx in range(n_parts):
                rf_der.parts[part_idx][:] = 0  # Make sure all parts are zero
        for row in range(rows):
            for col in range(cols):
                if rf_multilin:
                    n_parts = len(params.rfs[rf_idx].parts)
                    for part_idx in range(n_parts):
                        rf_der.parts[part_idx] += \
                            (p_z[row, col] *
                             (parts_z_spike[part_idx][row][col] -
                              parts_z[part_idx][row][col]) *
                             prob_frac_ders[rf_idx][row, col])
                else:
                    rf_der.field += \
                        (p_z[row, col] *
                         (x_nd_z_spike[row][col] - x_nd_z[row][col]) *
                         prob_frac_ders[rf_idx][row, col])
        if rf_multilin:
            n_parts = len(params.rfs[rf_idx].parts)
            for part_idx in range(n_parts):
                rf_der.parts[part_idx] *= 1/np.log(2)
        else:
            rf_der.field *= 1/np.log(2)
        rf_ders.append(rf_der)

    # Get partial derivatives for the CF when used
    cf_ders = []
    if n_cfs:
        assert n_rfs == 1
        cf_shape = params.cfs[0].shape
        z_cf_der_nds, z_cf_der_biases = z_cf_der(x, params)

        for cf_idx in range(n_cfs):
            # Determine the average cf partial derivative in each bin
            cf_z, cf_z_spike = _estimate_avg_bin_stimuli(
                z_cf_der_nds[cf_idx], z, y, p_z, z_edges)
            bias_z, bias_z_spike = _estimate_avg_bin_stimuli(
                z_cf_der_biases[cf_idx], z, y, p_z, z_edges)
            cf_der = Field(cf_shape, None, multilin=False, init_values='zeros')

            # CF: Integrating over all score bins
            for bin in range(n_bins):
                cf_der.field += (p_z[bin, 0] *
                                 (cf_z_spike[bin][0] - cf_z[bin][0]) *
                                 prob_frac_ders[0][bin, 0])
                cf_der.bias += (p_z[bin, 0] *
                                (bias_z_spike[bin][0] - bias_z[bin][0]) *
                                prob_frac_ders[0][bin, 0])
            cf_der.field *= 1 / np.log(2)
            cf_der.bias *= 1 / np.log(2)
            cf_ders.append(cf_der)

    return rf_ders, cf_ders


def _estimate_avg_bin_stimuli(x_nd_full, z, y, p_z, z_edges):
    """ Determine the average stimuli distribution bins

    <x|z> and <x|z,spike> that falls into each bin for both P(z) and P(z|spike)

    Args:
        x_nd_full: stimuli array
        z: similarity scores
        y: spike counts
        p_z: input matrix
        z_edges: receptive fields
    Returns:
        prob_frac_ders:
    Raises:

    """

    n_rfs = len(z_edges)
    n_bins = p_z.shape[0]

    # Locate which bin every sample falls into
    bin_idx = []
    for rf_idx in range(n_rfs):
        bin_idx.append(
            np.digitize(z[:, rf_idx].ravel(), z_edges[rf_idx]) - 1)

    # Initialize 2 dimensional list for storing the average stimuli in each bin
    rows = n_bins
    cols = 1 if n_rfs == 1 else n_bins
    x_z = [[[] for x in range(cols)] for x in range(rows)]
    x_z_spike = [[[] for x in range(cols)] for x in range(rows)]

    for row in range(rows):
        for col in range(cols):
            # Find all samples in the current bin
            if z.shape[1] == 1:
                row_sel = bin_idx[0] == row
            else:
                row_sel = (bin_idx[0] == row) & (bin_idx[1] == col)

            # Calculate <x|z>
            scaling = y.compress(row_sel)
            scaling[scaling == 0] = 1
            if len(scaling) == 0:
                s_tmp = np.zeros(x_nd_full.shape[1:])
            else:
                add_dims = [1 for dim in x_nd_full.shape[1:]]
                scaling = scaling.reshape([scaling.size] + add_dims)
                s_tmp = np.sum(x_nd_full[row_sel, :] * scaling, axis=0) \
                        / np.sum(scaling)
            x_z[row][col] = s_tmp

            # Calculate <x|z,spike>
            row_sel_spike = row_sel & (y > 0).ravel()
            if not np.any(row_sel_spike):
                s_tmp = np.zeros(x_nd_full.shape[1:])
            else:
                scaling = y[row_sel_spike, :].ravel()
                add_dims = [1 for dim in x_nd_full.shape[1:]]
                scaling = scaling.reshape([scaling.size] + add_dims)
                s_tmp = np.sum(x_nd_full[row_sel_spike, :] * scaling, axis=0) \
                        / np.sum(scaling)
            x_z_spike[row][col] = s_tmp

    return x_z, x_z_spike


def _estimate_prob_frac_der(p_z, p_z_spike, z_edges):
    """ Approximation for the derivative d(P(z|spike) / P(z)) / dz

    The approximation is obtained by convolving the fraction
    P(z|spike) / P(z) with a filter

    Args:
        p_z: P(z)
        p_z_spike: P(z|spike)
        z_edges: list with edge values along each dimension
    Returns:
        prob_frac_ders:
    Raises:

    """

    n_rfs = len(z_edges)

    prob_frac_ders = []
    for rf_idx in range(n_rfs):
        # Create a properly scaled filter
        d_z = z_edges[rf_idx][2] - z_edges[rf_idx][1]
        filter_weights = np.array([1., 0., -1.]).reshape(3, 1) / 2. / d_z

        # Rotate the filter for the second dimensions
        if rf_idx == 1:
            filter_weights = filter_weights.T

        # Caclulte the fraction P(z|spike) / P(z)
        prob_frac = np.zeros(p_z.shape)
        no_zero = p_z > 0
        prob_frac[no_zero] = p_z_spike[no_zero] / p_z[no_zero]

        # Convolve the fraction with the filter to get the derivative
        prob_frac_der = im_conv(prob_frac, filter_weights, mode='nearest')
        prob_frac_ders.append(prob_frac_der)

    return prob_frac_ders


def istac_mi(b, sta, stc):
    """ Calculate the mutual information captured by subspace (b)

    See Pillow and Simoncelli (2006) for details.
    Observer that the expression provided for the gradient is incorrect

    :param b: basis vectors
    :param sta: spike triggered average
    :param stc: spike triggered covariance
    :return mi: mutual information
    """

    stc_sta = stc + np.outer(sta.ravel(), sta.ravel())
    bT_stc_sta_b = np.dot(b.T, np.dot(stc_sta, b))
    bT_stc_b = np.dot(b.T, np.dot(stc, b))

    mi = 1 / (2*np.log(2)) * \
         (np.trace(bT_stc_sta_b) -
          np.log(np.linalg.det(bT_stc_b)) -
          b.shape[1])

    return mi


def istac_mi_der(b, sta, stc):
    """ Calculate the mutual information gradient for a subspace (b)

    See Pillow and Simoncelli (2006) for details.
    Observer that the expression provided for the gradient is incorrect

    :param b: basis vectors
    :param sta: spike triggered average
    :param stc: spike triggered covariance
    :return b_der: gradient for b
    """

    stc_sta = stc + np.outer(sta.ravel(), sta.ravel())
    bT_stc_b = np.dot(b.T, np.dot(stc, b))
    bT_stc_b_inv = np.linalg.inv(bT_stc_b)

    b_der = 1 / np.log(2) * \
            (np.dot(stc_sta, b) -
             np.dot(stc, np.dot(b, bT_stc_b_inv)))

    # stc_inv = np.linalg.inv(stc)
    # stc_bb_stc_inv = np.dot(stc, np.dot(b, np.dot(b.T, stc_inv)))
    # b_der_ori = 1 / np.log(2) * np.dot((stc_sta - stc_bb_stc_inv), b)

    return b_der