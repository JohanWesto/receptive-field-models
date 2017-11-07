#!/usr/bin/python
"""
" @section DESCRIPTION
" Optimization methods for various RF models
"""

import numpy as np
import multiprocessing as mp

from scipy import sparse
from operator import mul
from copy import deepcopy
from time import time
from sklearn.model_selection import KFold
from scipy.ndimage import correlate as im_corr
from scipy.linalg import block_diag
from liblinear.my_liblinear import my_train_wrap, set_verbosity_wrap
from rf_field_format import Field
from rf_piecewise_lin_act_fun import ActFun
from rf_ln_significance import Significance
from rf_evaluation import Evaluation
from rf_helper import add_fake_dimension, field_part_der, \
    outer_product, inner_product, sta_and_stc, calculate_r

import rf_obj_funs_and_grads as rf_obj_fun
import plotting.plotting_functions as plot_fun


###################################
# ---- Alternating solvers ---- #
def alternating_solver_rfs(x, y, params, opt_dict):
    """ Estimates multiple RF models

        This solver assumes an RF model on the form:
            y = max[ f(x^T w_j) ],
        where the maximum operator is taken over the different fields
        indexed by j.

        Options:
            error: error function: 'mse' or 'neg_log_lik_bernoulli/poisson'
            print: print progress: True / False

        Args:
            x: input array
            y: output array
            params: field parameters
            cf_act_fun: context activation function: linear / sigmoid
            opt_dict: optimization parameters
        Returns:
            params: found field parameters
        Raise

        """

    # Read options
    solver = opt_dict['solver']
    verbose = opt_dict['verbose']
    reg_c = params.reg_c

    # Get field dimensions
    n_rfs = len(params.rfs)
    rf_shape = params.rfs[0].shape
    rf_size = reduce(mul, rf_shape)

    # Weights for each sample
    weights = np.ones(y.shape)
    if solver == 'logreg':
        weights[y > 1] = y[y > 1]  # add weight if multiple spikes occurred

    # Assign the correct objective function
    if solver == 'linreg':
        obj_fun = rf_obj_fun.mean_squared_error
    elif solver == 'logreg':
        obj_fun = rf_obj_fun.neg_log_lik_bernoulli
    elif solver == 'poireg':
        obj_fun = rf_obj_fun.neg_log_lik_poisson
    else:
        raise Exception("Unknown solver: {}".format(solver))

    if verbose:
        print "Finding multiple RFs using a {} solver".format(solver)
        print "{:15s}{:15s}{:15s}".format(
            'Iteration', 'Obj. fun. val.', 'Decrease (%)')
        plot_fun.interactive(True)
        fig = plot_fun.plot_parameters(params)

    # Alternating loop over all fields
    n_iterations = 1
    converged = False
    reg_c_cv_done = False
    obj_fun_val = obj_fun(x, y, params, weights=weights, reg_c=reg_c)

    y_neg_idx_all = np.argwhere(y.ravel() == 0).ravel()
    y_pos_idx_all = np.argwhere(y.ravel() > 0).ravel()
    y_pos_idxs = np.array_split(y_pos_idx_all, n_rfs)

    while not converged:

        # 1. Partial derivatives for the receptive field parameters
        z_rf_der_nd = rf_obj_fun.z_rf_der(x, params)

        # 2. Reshape z_rf_der_nd so that it is only 2-dimensional
        z_rf_der_2d = z_rf_der_nd.reshape(y.size, rf_size)
        z_rf_der_2d = np.hstack((np.ones([z_rf_der_2d.shape[0], 1]),
                                 z_rf_der_2d))

        # 3. Solving the rf sub-problems
        for rf_id in range(n_rfs):
            # 3.1 Initialize
            z_rf_der_2d_tmp = np.vstack([z_rf_der_2d[y_neg_idx_all, :],
                                         z_rf_der_2d[y_pos_idxs[rf_id], :]])
            y_tmp = np.vstack([y[y_neg_idx_all],
                               y[y_pos_idxs[rf_id]]])
            weights_tmp = np.vstack([weights[y_neg_idx_all],
                                     weights[y_pos_idxs[rf_id]]])
            # 3.2 Solve the sub-problem
            param_init = np.hstack((params.rfs[rf_id].bias,
                                    params.rfs[rf_id].field.ravel()))
            # reg_c = _reg_c_cv_search(z_rf_der_2d_tmp, y_tmp, solver,
            #                          reg_l=params.rfs[rf_id].reg_l,
            #                          weights=weights_tmp,
            #                          w0=param_init,
            #                          n_folds=4,
            #                          verbose=verbose)
            param = _convex_subproblem_solver(z_rf_der_2d_tmp, y_tmp, solver,
                                              reg_l=params.rfs[rf_id].reg_l,
                                              weights=weights_tmp,
                                              reg_c=reg_c,
                                              w0=param_init)

            # 3.3 Extract field and bias
            params.rfs[rf_id].field = param[1:].reshape(rf_shape)
            params.rfs[rf_id].bias = param[0]

        # 4. Update sample assignments
        z = inner_product(z_rf_der_nd, params.rfs)
        z_argmax = z.argmax(axis=1)
        for rf_id in range(n_rfs):
            y_pos_idxs[rf_id] = np.argwhere((z_argmax == rf_id).ravel() &
                                            (y > 0).ravel()).ravel()

        # Comparing new and old obj. fun. values
        obj_fun_val_old = obj_fun_val
        obj_fun_val = obj_fun(x, y, params,
                              weights=weights,
                              reg_c=reg_c)
        diff = obj_fun_val - obj_fun_val_old
        diff_percent = diff / obj_fun_val_old

        # Show progress
        if verbose:
            print "{:<15d}{:<15.1f}{:<15.3f}".format(
                n_iterations, obj_fun_val, diff_percent * 100)
            plot_fun.plot_parameters(params, fig)

        # Convergence check
        if -1e-4 < diff_percent <= 0 or n_iterations > 200:
            # Terminate if optimal reg_c value is already used
            if reg_c_cv_done:
                converged = True
                if verbose:
                    plot_fun.interactive(False)
                    plot_fun.close(fig)
            # otherwise, use optimal values for one last iteration.
            else:
                reg_c_cv_done = True
        else:
            n_iterations += 1

    return params


def alternating_solver_ctx(x, y, params, opt_dict):
    """ Estimates context models

    This block based solver alternates between solving a convex sub-problems
    for finding RF and CF parameters, respectively.

    Options:
        error: error function: 'mse' or 'neg_log_lik_bernoulli/poisson'
        print: print progress: True / False

    Args:
        x: input array
        y: output array
        params: field parameters
        opt_dict: optimization parameters
    Returns:
        params: found field parameters
    Raise

    """

    # Solver requirements
    assert len(params.rfs) == 1

    # Read options
    solver = opt_dict['solver']
    verbose = opt_dict['verbose']

    # Weights for each sample
    weights = np.ones(y.shape)
    if solver == 'logreg':
        weights[y > 1] = y[y > 1]  # add weight if multiple spikes occurred

    # Assign the correct objective function
    if solver == 'linreg':
        obj_fun = rf_obj_fun.mean_squared_error
    elif solver == 'logreg':
        obj_fun = rf_obj_fun.neg_log_lik_bernoulli
    elif solver == 'poireg':
        obj_fun = rf_obj_fun.neg_log_lik_poisson
    else:
        raise Exception("Unknown solver: {}".format(solver))

    if verbose:
        print "Running an alternating context model" \
              " solver of type: {}".format(solver)
        print "{:15s}{:15s}{:15s}".format(
            'Iteration', 'Obj. fun. val.', 'Decrease (%)')
        plot_fun.interactive(True)
        fig = plot_fun.plot_parameters(params)

    # Set a unity bias for each CF
    for cf in params.cfs:
        cf.bias = 1

    # Alternating loop over all fields
    n_iterations = 1
    converged = False
    cross_val = False
    reg_c_from_cv = params.rfs[0].reg_c_from_cv
    obj_fun_val = obj_fun(x, y, params, weights=weights)
    while not converged:

        # 1. Update the RF
        if n_iterations > 0:
            if params.rfs[0].multilin:
                params = _solve_rf_sub_multilin(x, y, params, weights,
                                                opt_dict)
            else:
                params = _solve_rf_sub_full(x, y, params, weights,
                                            opt_dict, cross_val)
        # obj_fun_val_tmp = obj_fun(x, y, params, weights=weights)
        # print "RF: {}".format(obj_fun_val_tmp)

        # 2. Update CFs
        if params.cfs:
            if params.cfs[0].multilin:
                params = _solve_cfs_sub_mutlilin(x, y, params, weights,
                                                 opt_dict)
            else:
                params = _solve_cfs_sub_full(x, y, params, weights,
                                             opt_dict, cross_val)
        # obj_fun_val_tmp = obj_fun(x, y, params, weights=weights)
        # print "CF: {}".format(obj_fun_val_tmp)

        if cross_val:
            n_iterations = 0
            reg_c_from_cv = True
            cross_val = False

        # Comparing new and old obj. fun. values
        obj_fun_val_old = obj_fun_val
        obj_fun_val = obj_fun(x, y, params, weights=weights)
        diff = obj_fun_val - obj_fun_val_old
        diff_percent = diff / abs(obj_fun_val_old)

        # Show progress
        if verbose:
            print "{:<15d}{:<15.1f}{:<15.3f}".format(
                n_iterations, obj_fun_val, diff_percent * 100)
            plot_fun.plot_parameters(params, fig)

        # Convergence check
        if -1e-4 < diff_percent <= 0 or 1 < n_iterations > 50:
            # Try to add additional context fields
            added_cf = params.update_context_map()
            # Reset the counter if CFs are added
            if added_cf:
                n_iterations = 0
            # alternatively, order a final CV or terminate
            else:
                # converged = True
                # if verbose:
                #     plot_fun.interactive(False)
                #     plot_fun.close(fig)
                # Terminate if optimal reg_c value is already used
                if reg_c_from_cv:
                    converged = True
                    if verbose:
                        plot_fun.interactive(False)
                        plot_fun.close(fig)
                # otherwise, use optimal values for one last iteration.
                else:
                    cross_val = True
                    n_iterations = 1
        else:
            n_iterations += 1

    return params


def alternating_solver_ctx_fixed(x, y, params, opt_dict):
    """ Estimates context models with an fixed CF activation function

    This block based solver alternates between a convex sub-problems
    for finding the RF, and a non-convex sub-problem for finding the CFs

    Options:
        error: error function: 'mse' or 'neg_log_lik_bernoulli/poisson'
        print: print progress: True / False

    Args:
        x: input array
        y: output array
        params: field parameters
        opt_dict: optimization parameters
    Returns:
        params: found field parameters
    Raise

    """

    # Solver requirements
    assert len(params.rfs) == 1

    # Read options
    solver = opt_dict['solver']
    verbose = opt_dict['verbose']
    reg_c = params.reg_c

    # Weights for each sample
    weights = np.ones(y.shape)
    if solver == 'logreg':
        weights[y > 1] = y[y > 1]  # add weight if multiple spikes occurred

    # Assign the correct objective function
    if solver == 'linreg':
        obj_fun = rf_obj_fun.mean_squared_error
    elif solver == 'logreg':
        obj_fun = rf_obj_fun.neg_log_lik_bernoulli
    elif solver == 'poireg':
        obj_fun = rf_obj_fun.neg_log_lik_poisson
    else:
        raise Exception("Unknown solver: {}".format(solver))

    if verbose:
        plot_fun.interactive(True)
        fig = plot_fun.plot_parameters(params)
        print "Running an alternating subunit model" \
              " solver of type: {}".format(solver)
        print "{:15s}{:15s}{:15s}".format(
            'Iteration', 'Obj. fun. val.', 'Decrease (%)')

    # Alternating loop over all fields
    n_iterations = 1
    converged = False
    obj_fun_val = obj_fun(x, y, params, weights=weights, reg_c=reg_c)
    while not converged:

        # 1. CFs
        params = _solve_nonlin_cfs(x, y, params, weights, opt_dict)

        # 2. RF
        params = _solve_rf_sub_full(x, y, params, weights, opt_dict, False)

        # 3. Evaluate outer loop progress
        obj_fun_val_old = obj_fun_val
        obj_fun_val = obj_fun(x, y, params,
                              weights=weights,
                              reg_c=reg_c)
        diff = obj_fun_val - obj_fun_val_old
        diff_percent = diff / obj_fun_val_old

        # Show outer loop progress
        if verbose:
            print "{:<15d}{:<15.1f}{:<15.3f}".format(
                n_iterations, obj_fun_val, diff_percent * 100)
            plot_fun.plot_parameters(params, fig)

        # Convergence check
        if -5e-4 < diff_percent <= 0 or n_iterations > 50:
            converged = True
            if verbose:
                plot_fun.interactive(False)
                plot_fun.close(fig)
        else:
            n_iterations += 1

    return params


def alternating_solver_ctx_adaptive(x, y, params, opt_dict):
    """ Estimates context models with an adaptive CF activation function

    This block based solver alternates between solving two convex sub-problems
    for finding the RF and the activation function, and a non-convex
    sub-problem for finding the CF

    Options:
        error: error function: 'mse' or 'neg_log_lik_bernoulli/poisson'
        print: print progress: True / False

    Args:
        x: input array
        y: output array
        params: field parameters
        opt_dict: optimization parameters
    Returns:
        params: found field parameters
    Raise

    """

    # Solver requirements
    assert len(params.rfs) == 1

    # Read options
    solver = opt_dict['solver']
    rf_truncated = opt_dict['rf_truncated']
    first_fold = opt_dict['first_fold']
    verbose = opt_dict['verbose']

    # Weights for each sample
    weights = np.ones(y.shape)
    if solver == 'logreg':
        weights[y > 1] = y[y > 1]  # add weight if multiple spikes occurred

    # Assign the correct objective function
    if solver == 'linreg':
        obj_fun = rf_obj_fun.mean_squared_error
    elif solver == 'logreg':
        obj_fun = rf_obj_fun.neg_log_lik_bernoulli
    elif solver == 'poireg':
        obj_fun = rf_obj_fun.neg_log_lik_poisson
    else:
        raise Exception("Unknown solver: {}".format(solver))

    # Mask CFs and truncate the RF (subunit only)
    if first_fold:
        params.mask_cf()
        if rf_truncated:
            params.truncate_rf()

    # Get first estimate of the nonlinearity
    params = _solve_act_fun_sub(x, y, params, weights, opt_dict, False)

    if verbose:
        plot_fun.interactive(True)
        fig = plot_fun.plot_parameters(params)
        print "Running an alternating subunit model" \
              " solver of type: {}".format(solver)
        print "{:15s}{:15s}{:15s}".format(
            'Iteration', 'Obj. fun. val.', 'Decrease (%)')

    # Alternating loop over all fields
    n_iterations = 1
    converged = False
    obj_fun_val = obj_fun(x, y, params, weights=weights)

    while not converged:

        # 1. First leg
        params = _solve_nonlin_cfs(x, y, params, weights, opt_dict)

        # 2. Second leg
        params = _solve_rf_and_act_fun(x, y, params, weights, opt_dict)

        # 3. Cross-validation for alphas and the RF (if truncated)
        # if n_iterations == 1:
        #     params = _solve_act_fun_sub(x, y, params, weights, opt_dict, True)
        if n_iterations == 3 and not params.rfs[0].reg_c_from_cv:
            params = _solve_rf_sub_full(x, y, params, weights, opt_dict, True)

        # 4. Evaluate outer loop progress
        obj_fun_val_old = obj_fun_val
        obj_fun_val = obj_fun(x, y, params, weights=weights)
        diff = obj_fun_val - obj_fun_val_old
        diff_percent = diff / obj_fun_val_old

        # Show outer loop progress
        if verbose:
            print "{:<15d}{:<15.1f}{:<15.3f}".format(
                n_iterations, obj_fun_val, diff_percent * 100)
            plot_fun.plot_parameters(params, fig)

        # Convergence check
        if -4e-4 < diff_percent <= 0 or n_iterations > 20:

            # Check for standard solution
            flipped_rf = params.check_rf()
            if flipped_rf:
                params = _solve_act_fun_sub(x, y, params, weights, opt_dict,
                                            False)
                if verbose:
                    plot_fun.plot_parameters(params, fig)
                n_iterations = 1
            else:
                converged = True
                if verbose:
                    plot_fun.interactive(False)
                    plot_fun.close(fig)
        else:
            n_iterations += 1

    return params


def _solve_rf_and_act_fun(x, y, params, weights, opt_dict):
    """ Solves for the RF and act. funs. in a coordinate descent subunit solver

    This solver find the RF and the activation functions by solving complex
    sub-problems in alternating fashion.

    :param x:
    :param y:
    :param params:
    :param weights:
    :param opt_dict:
    :return:
    """

    # Solver requirements
    assert len(params.rfs) == 1

    # Read options
    solver = opt_dict['solver']
    verbose = opt_dict['verbose']

    # Assign the correct objective function
    if solver == 'linreg':
        obj_fun = rf_obj_fun.mean_squared_error
    elif solver == 'logreg':
        obj_fun = rf_obj_fun.neg_log_lik_bernoulli
    elif solver == 'poireg':
        obj_fun = rf_obj_fun.neg_log_lik_poisson
    else:
        raise Exception("Unknown solver: {}".format(solver))

    if verbose:
        plot_fun.interactive(True)
        fig = plot_fun.plot_parameters(params)
        print "\tRF and act. fun. solver of type: {}".format(solver)
        print "\t{:15s}{:15s}{:15s}".format(
            'Iteration', 'Obj. fun. val.', 'Decrease (%)')

    converged = False
    n_iterations = 1
    obj_fun_val = obj_fun(x, y, params, weights=weights)
    while not converged:

        # 1. Update the activation function
        params = _solve_act_fun_sub(x, y, params, weights, opt_dict, False)

        # 2. Update the RF
        if n_iterations > 0:
            if params.rfs[0].multilin:
                params = _solve_rf_sub_multilin(x, y, params, weights,
                                                opt_dict)
            else:
                params = _solve_rf_sub_full(x, y, params, weights,
                                            opt_dict, False)

        # Comparing new and old obj. fun. values
        obj_fun_val_old = obj_fun_val
        obj_fun_val = obj_fun(x, y, params, weights=weights)
        diff = obj_fun_val - obj_fun_val_old
        diff_percent = diff / obj_fun_val_old

        # Show progress
        if verbose:
            print "\t{:<15d}{:<15.1f}{:<15.3f}".format(
                n_iterations, obj_fun_val, diff_percent * 100)
            plot_fun.plot_parameters(params, fig)

        # Convergence check
        if -2e-4 < diff_percent <= 0 or n_iterations > 50:
            converged = True
            if verbose:
                plot_fun.interactive(False)
                plot_fun.close(fig)
        else:
            n_iterations += 1

    return params


def _solve_nonlin_cfs(x, y, params, weights, opt_dict):
    """ Solves for nonlin CFs in a coordinate descent subunit solver

    This solver finds CFs by successive line searches along a Newton direction

    :param x: input array
    :param y: output array
    :param params: RF model parameters
    :param weights: sample weights
    :param opt_dict: optimization parameters
    :return params: found field parameters
    """

    # Solver requirements
    assert len(params.rfs) == 1

    # Read options
    solver = opt_dict['solver']
    verbose = opt_dict['verbose']

    # Assign the correct objective function
    if solver == 'linreg':
        obj_fun = rf_obj_fun.mean_squared_error
    elif solver == 'logreg':
        obj_fun = rf_obj_fun.neg_log_lik_bernoulli
    elif solver == 'poireg':
        obj_fun = rf_obj_fun.neg_log_lik_poisson
    else:
        raise Exception("Unknown solver: {}".format(solver))

    # Initiate arrays for storing the momentum in
    cf_deltas = [Field(cf.shape, None, cf.multilin) for cf in params.cfs]

    obj_fun_val = obj_fun(x, y, params, weights=weights)
    obj_fun_val_best = obj_fun_val

    if verbose:
        plot_fun.interactive(True)
        fig = plot_fun.plot_parameters(params)
        print "\tNonlin CFs solver of type: {}".format(solver)
        print "\t{:15s}{:15s}{:15s}".format(
            'Iteration', 'Obj. fun. val.', 'Decrease (%)')

    n_iterations = 1
    converged = False
    while not converged:

        # Get newton directions fro all CFs
        cf_dirs = _newton_direction(x, y, params, weights, opt_dict)

        # Line search to find optimal eta
        eta_tmp = 0.2
        decreasing = True
        while decreasing:
            params_tmp = deepcopy(params)
            cf_deltas_tmp = [deepcopy(cf_delta) for cf_delta in cf_deltas]
            for cf_idx in range(len(params.cfs)):
                params_tmp.cfs[cf_idx] = _field_step_update(
                    params_tmp.cfs[cf_idx], cf_dirs[cf_idx],
                    cf_deltas_tmp[cf_idx],
                    eta=eta_tmp, alpha=0, sign=-1)
            obj_fun_val_tmp = obj_fun(x, y, params_tmp,
                                      weights=weights)
            # print "eta: {:1.2e},\tobj fun: {}".format(eta_tmp,
            #                                           obj_fun_val_tmp)
            if obj_fun_val_tmp < obj_fun_val_best:
                eta_tmp += 0.2
                obj_fun_val_best = obj_fun_val_tmp
            else:
                eta = eta_tmp - 0.2
                decreasing = False

        # Use the optimal eta
        for cf_idx in range(len(params.cfs)):
            params.cfs[cf_idx] = _field_step_update(
                params.cfs[cf_idx], cf_dirs[cf_idx], cf_deltas[cf_idx],
                eta=eta, alpha=0, sign=-1)

        # Calculate the new error value
        obj_fun_val_old = obj_fun_val
        obj_fun_val = obj_fun(x, y, params, weights=weights)
        diff = obj_fun_val - obj_fun_val_old
        diff_percent = diff / obj_fun_val_old

        # Show progress
        if verbose:
            print "\t{:<15d}{:<15.1f}{:<15.3f}".format(
                n_iterations, obj_fun_val, diff_percent * 100)
            plot_fun.plot_parameters(params, fig)

        # Convergence check
        if -2e-4 < diff_percent or n_iterations >= 15:
            converged = True
            if verbose:
                plot_fun.interactive(False)
                plot_fun.close(fig)
        else:
            n_iterations += 1

    return params


def _solve_act_fun_sub(x, y, params, weights, opt_dict, cross_val):
    """ Solves the RF subproblen when estimating context models

    :param x:
    :param y:
    :param params:
    :param weights:
    :param opt_dict:
    :param cross_val:
    :return params:
    """

    solver = opt_dict['solver']
    verbose = opt_dict['verbose']

    # 1. Partial derivatives for the receptive field parameters
    z_alpha_der_nds = rf_obj_fun.z_alpha_der(x, params)
    z_alpha_der_2d = np.hstack(z_alpha_der_nds)

    # 2 Determine the sample specific constant
    bias_i = np.ones(y.size)*params.rfs[0].bias

    # 3. Initial guess and regularization
    param_init = []
    reg_l = []
    for act_fun_idx in range(len(params.cf_act_funs)):
        param_init.append(params.cf_act_funs[act_fun_idx].alpha)
        reg_l.append(params.cf_act_funs[act_fun_idx].reg_l)
    param_init = np.hstack(param_init)
    if sparse.issparse(reg_l[0]):
        reg_l = sparse.block_diag(reg_l)
    else:
        reg_l = block_diag(*reg_l)

    # 4. Solving the act fun sub-problem
    # 4.1 Optional cv search for an optimal reg_c values
    if cross_val:
        reg_c_tmp = _reg_c_cv_search(z_alpha_der_2d, y, solver,
                                     reg_l=reg_l,
                                     weights=weights,
                                     verbose=verbose)
        for act_fun in params.cf_act_funs:
            act_fun.reg_c = reg_c_tmp
            act_fun.reg_c_from_cv = True
    else:
        reg_c_tmp = params.cf_act_funs[0].reg_c

    # 4.2 Find optimal alpha parameters
    param = _convex_subproblem_solver(z_alpha_der_2d, y, solver,
                                      reg_l=reg_l,
                                      weights=weights,
                                      bias_i=bias_i,
                                      reg_c=reg_c_tmp,
                                      w0=param_init)

    # 5. Extract field and bias
    for act_fun_idx in range(len(params.cf_act_funs)):
        alpha_size = params.cf_act_funs[act_fun_idx].alpha.size
        params.cf_act_funs[act_fun_idx].alpha = \
            param[act_fun_idx * alpha_size:(act_fun_idx + 1) * alpha_size]

    return params


def _solve_rf_sub_full(x, y, params, weights, opt_dict, cross_val):
    """ Solves the RF subproblen when estimating context models

    :param x:
    :param y:
    :param params:
    :param weights:
    :param opt_dict:
    :param cross_val:
    :return params:
    """

    solver = opt_dict['solver']
    verbose = opt_dict['verbose']

    # Get field dimensions
    rf = params.rfs[0]
    rf_shape = rf.shape
    rf_size = reduce(mul, rf_shape)

    # 1. Partial derivatives for the receptive field parameters
    z_rf_der_nd = rf_obj_fun.z_rf_der(x, params)

    # 2. Reshape z_rf_der_nd so that it is only 2-dimensional
    z_rf_der_2d_tmp = z_rf_der_nd.reshape(y.size, rf_size)
    z_rf_der_2d = np.hstack((np.ones([z_rf_der_2d_tmp.shape[0], 1]),
                             z_rf_der_2d_tmp))

    # 3. Solving the rf sub-problem
    # 3.1 Optional cv search for an optimal reg_c values
    if cross_val:
        rf.reg_c = _reg_c_cv_search(z_rf_der_2d, y, solver,
                                    reg_l=rf.reg_l,
                                    weights=weights,
                                    verbose=verbose)
        rf.reg_c_from_cv = True

    # 3.2 Find field parameters
    param_init = np.hstack((rf.bias, rf.field.ravel()))
    param = _convex_subproblem_solver(z_rf_der_2d, y, solver,
                                      reg_l=rf.reg_l,
                                      weights=weights,
                                      reg_c=rf.reg_c,
                                      w0=param_init)

    # 4. Extract field and bias
    rf.field = param[1:].reshape(rf_shape)
    rf.bias = param[0]

    return params


def _solve_rf_sub_multilin(x, y, params, weights, opt_dict):
    """ Solves the RF subproblen when estimating context models

    :param x:
    :param y:
    :param params:
    :param weights:
    :param opt_dict:
    :param cross_val:
    :return params:
    """

    solver = opt_dict['solver']

    # 1. Partial derivatives for the receptive field parameters
    z_rf_der_nd = rf_obj_fun.z_rf_der(x, params)

    # 2. Solving the subproblems for each field part iteratively
    n_parts = len(params.rfs[0].parts)
    for part_idx in range(n_parts):
        # 2.1. Partial derivatives for the selected field part
        part_der = field_part_der(z_rf_der_nd, params.rfs[0], part_idx)

        # 2.2. Reshape x_part so that it is only 2-dimensional
        part_shape = params.rfs[0].parts[part_idx].shape
        part_size = params.rfs[0].parts[part_idx].size
        part_der = part_der.reshape(y.size, part_size)

        # 2.3. Solve the part subproblem
        param_init = params.rfs[0].parts[part_idx].ravel()
        bias_i = params.rfs[0].bias * np.ones([y.size, 1])
        reg_l = params.rfs[0].reg_l[part_idx]
        reg_c = params.rfs[0].reg_c
        rf_part = _convex_subproblem_solver(part_der, y, solver,
                                            reg_l=reg_l,
                                            reg_c=reg_c,
                                            weights=weights,
                                            bias_i=bias_i,
                                            w0=param_init)
        params.rfs[0].parts[part_idx] = rf_part.reshape(part_shape)

    # 2.4. Calculate the receptive field
    params.rfs[0].field = outer_product(params.rfs[0].parts)

    # 3. Update the bias
    # 3.1. Calculate the inner product between the partial
    # derivatives and the receptive field
    x_axes = range(1, len(z_rf_der_nd.shape))
    rf_axes = range(len(params.rfs[0].shape))
    bias_i = np.tensordot(z_rf_der_nd, params.rfs[0].field,
                          axes=(x_axes, rf_axes))

    # 3.2. Solve the bias subproblem
    x_part = np.ones([y.size, 1])
    reg_l = 0 * np.eye(x_part.shape[1])  # No regularization on the bias term
    reg_c = params.rfs[0].reg_c
    params.rfs[0].bias = _convex_subproblem_solver(x_part, y, solver,
                                                   reg_l=reg_l,
                                                   reg_c=reg_c,
                                                   weights=weights,
                                                   bias_i=bias_i)[0]

    return params


def _solve_cfs_sub_full(x, y, params, weights, opt_dict, cross_val):
    """ Solves the CFs subproblen when estimating context models

    :param x:
    :param y:
    :param params:
    :param weights:
    :param opt_dict:
    :param cross_val:
    :return params:
    """

    solver = opt_dict['solver']
    verbose = opt_dict['verbose']

    # 1. Partial derivatives for the receptive field parameters
    z_cf_der_nds, z_cf_der_biases = rf_obj_fun.z_cf_der(x, params)

    # 2 Reshape all field derivatives into a 2-dimensional matrix
    z_cf_der_2d = []
    for cf_idx in range(len(params.cfs)):
        cf_shape = params.cfs[cf_idx].shape
        cf_size = reduce(mul, cf_shape)
        z_cf_der_2d_tmp = z_cf_der_nds[cf_idx].reshape(y.size, cf_size).copy()
        z_cf_der_2d.append(z_cf_der_2d_tmp)
    z_cf_der_2d = np.hstack(z_cf_der_2d)

    # 3 Determine the sample specific constant
    bias_i = np.zeros(y.size)
    for cf_idx in range(len(params.cfs)):
        bias_i += z_cf_der_biases[cf_idx].ravel()
    bias_i += params.rfs[0].bias

    # 4. Solving the cf sub-problem
    # 4.1 Initialization
    param_init = []
    reg_l = []
    for cf_idx in range(len(params.cfs)):
        param_init.append(params.cfs[cf_idx].field.ravel())
        reg_l.append(params.cfs[cf_idx].reg_l)
    param_init = np.hstack(param_init)
    if sparse.issparse(reg_l[0]):
        reg_l = sparse.block_diag(reg_l)
    else:
        reg_l = block_diag(*reg_l)
    # 4.2 Optional cv search for an optimal reg_c values
    if cross_val:
        reg_c_tmp = _reg_c_cv_search(z_cf_der_2d, y, solver,
                                     reg_l=reg_l,
                                     weights=weights,
                                     bias_i=bias_i,
                                     w0=param_init,
                                     verbose=verbose)
        for cf in params.cfs:
            cf.reg_c = reg_c_tmp
            cf.reg_c_from_cv = True
    else:
        reg_c_tmp = params.cfs[0].reg_c

    # 4.3 Find field parameters
    param = _convex_subproblem_solver(z_cf_der_2d, y, solver,
                                      reg_l=reg_l,
                                      weights=weights,
                                      bias_i=bias_i,
                                      reg_c=reg_c_tmp,
                                      w0=param_init)

    # 5. Extract fields
    for cf_idx in range(len(params.cfs)):
        cf_shape = params.cfs[cf_idx].shape
        cf_size = reduce(mul, cf_shape)
        params.cfs[cf_idx].field = \
            param[cf_idx * cf_size:(cf_idx + 1) * cf_size].reshape(cf_shape)

    return params


def _solve_cfs_sub_mutlilin(x, y, params, weights, opt_dict):

    solver = opt_dict['solver']

    # 1. Partial derivatives for the context field parameters
    z_cf_der_nds, z_cf_der_biases = rf_obj_fun.z_cf_der(x, params)

    for cf_idx in range(len(params.cfs)):

        # 2. Solving the subproblems for each field part iteratively
        n_parts = len(params.cfs[cf_idx].parts)
        for part_idx in range(n_parts):
            # 2.1. Partial derivatives for the selected field part
            part_der = field_part_der(z_cf_der_nds[cf_idx],
                                      params.cfs[cf_idx],
                                      part_idx)

            # 2.2. Reshape x_part so that it is only 2-dimensional
            part_shape = params.cfs[cf_idx].parts[part_idx].shape
            part_size = params.cfs[cf_idx].parts[part_idx].size
            part_der = part_der.reshape(y.size, part_size)

            # 2.3. Solve the part subproblem
            rf_tmp = deepcopy(params.rfs[0])
            rf_tmp.field[params.context_map == cf_idx] = 0
            z_rf_der_nd = rf_obj_fun.z_rf_der(x, params)
            bias_i = inner_product(z_rf_der_nd, [rf_tmp]) + \
                     z_cf_der_biases[cf_idx]
            param_init = params.cfs[cf_idx].parts[part_idx].ravel()
            reg_l = params.cfs[cf_idx].reg_l[part_idx]
            reg_c = params.cfs[cf_idx].reg_c
            cf_part = _convex_subproblem_solver(part_der, y, solver,
                                                reg_l=reg_l,
                                                reg_c=reg_c,
                                                weights=weights,
                                                bias_i=bias_i,
                                                w0=param_init)
            params.cfs[cf_idx].parts[part_idx] = cf_part.reshape(part_shape)

        # 2.4. Calculate the context field
        params.cfs[cf_idx].field = \
            outer_product(params.cfs[cf_idx].parts)

    return params


def _newton_direction(x, y, params, weights, opt_dict):
    """ Finds a newton direction when optimizing subunit CFs

    :param x:
    :param y:
    :param params:
    :param weights:
    :param opt_dict:
    :param cross_val:
    :return params:
    """

    # Read options
    solver = opt_dict['solver']

    # Assign the correct objective function
    if solver == 'linreg':
        error_measure = 'mse'
    elif solver == 'logreg':
        error_measure = 'neg_log_lik_bernoulli'
    elif solver == 'poireg':
        error_measure = 'neg_log_lik_poisson'
    else:
        raise Exception("Unknown solver: {}".format(solver))

    # Derivatives and Hessian
    cf_ders, cf_hessians = \
        rf_obj_fun.cf_der_and_hessian(x, y, params,
                                      error_measure=error_measure,
                                      weights=weights)

    newton_dirs = []
    for cf_idx in range(len(params.cfs)):
        # Newton step as inv(Hessian) * derivative
        cf_der = np.insert(cf_ders[cf_idx].field.ravel(), 0,
                           cf_ders[cf_idx].bias)
        inv_hessian = np.linalg.inv(cf_hessians[cf_idx])
        direction = np.dot(inv_hessian, cf_der)

        # Reshape
        cf_shape = params.cfs[cf_idx].shape
        newton_dir = Field(cf_shape, None, params.cfs[cf_idx].multilin)
        newton_dir.field = direction[1:].reshape(cf_shape)
        newton_dir.bias = direction[0]
        newton_dirs.append(newton_dir)

    return newton_dirs


def _reg_c_cv_search(x, y, solver,
                     reg_l=np.array([]),
                     weights=np.array([]),
                     bias_i=np.array([]),
                     w0=np.array([]),
                     n_splits=4,
                     verbose=1):

    """ CV search for optimal reg_c value

    :param x:
    :param y:
    :param solver:
    :param weights:
    :param bias_i:
    :param n_splits:
    :return reg_c_opt:
    """

    if verbose:
        print "Running cross-validation to find an optimal reg_c value."

    reg_c_values = np.logspace(-4, 1, 6)
    eval_train = Evaluation()
    eval_test = Evaluation()
    r_train = []
    r_test = []
    mi_train = []
    mi_test = []

    # Use a default diagonal ltl regularization matrix if none provided
    if reg_l.shape[0] != x.shape[1]:
        reg_l = np.eye(x.shape[1])

    # Add a unity weights if none are provided
    if weights.size != y.size:
        weights = np.ones(y.size)

    # Add zero biases if none are provided
    if bias_i.size != y.size:
        bias_i = np.zeros(y.size)

    kf = KFold(n_splits=n_splits, shuffle=True)
    for reg_c in reg_c_values:
        mi_train_tmp = 0
        mi_test_tmp = 0
        r_train_tmp = 0
        r_test_tmp = 0
        for train_idx, test_idx in kf.split(range(y.size)):
            x_train = x[train_idx, :]
            x_test = x[test_idx, :]
            y_train = y[train_idx, :]
            y_test = y[test_idx, :]

            w = _convex_subproblem_solver(x_train, y_train, solver,
                                          reg_l=reg_l,
                                          weights=weights[train_idx],
                                          bias_i=bias_i[train_idx],
                                          reg_c=reg_c,
                                          w0=w0.ravel())

            z_train = np.dot(x_train, w) + bias_i.ravel()[train_idx]
            z_test = np.dot(x_test, w) + bias_i.ravel()[test_idx]
            z_train = z_train.reshape(z_train.size, 1)
            z_test = z_test.reshape(z_test.size, 1)
            if solver == 'linreg':
                y_hat_train = z_train
                y_hat_test = z_test
            elif solver == 'logreg':
                y_hat_train = rf_obj_fun.inv_bernoulli_link(z_train)
                y_hat_test = rf_obj_fun.inv_bernoulli_link(z_test)
            elif solver == 'poireg':
                y_hat_train = rf_obj_fun.inv_poisson_link(z_train)
                y_hat_test = rf_obj_fun.inv_poisson_link(z_test)
            else:
                raise Exception("Unknown solver: {}".format(solver))

            eval_train.evaluate_mi_raw(z_train, y_train)
            eval_train.evaluate_r(y_hat_train, y_train)
            eval_test.evaluate_mi_raw(z_test, y_test)
            eval_test.evaluate_r(y_hat_test, y_test)

            mi_train_tmp += eval_train.mi['raw']
            mi_test_tmp += eval_test.mi['raw']
            r_train_tmp += eval_train.r
            r_test_tmp += eval_test.r

        mi_train.append(mi_train_tmp / n_splits)
        mi_test.append(mi_test_tmp / n_splits)
        r_train.append(r_train_tmp / n_splits)
        r_test.append(r_test_tmp / n_splits)

    mi_train = np.array(mi_train)
    mi_test = np.array(mi_test)
    r_train = np.array(r_train)
    r_test = np.array(r_test)
    # best_idx = np.argmax(np.around(r_test, 3))  # ignore very small improvements
    best_idx = np.argmax(np.around(mi_test, 3))  # ignore very small improvements
    reg_c_opt = reg_c_values[best_idx]

    if verbose:
        print "Optimal reg_c found to be: {:1.1e}".format(reg_c_opt)
        print_format = '{:10s}' + ''.join(['{:10.1e}'] * reg_c_values.size)
        print print_format.format(*['reg_c'] + reg_c_values.tolist())
        print_format = '{:10s}' + ''.join(['{:10.3f}'] * reg_c_values.size)
        print print_format.format(*['r_train'] + r_train.tolist())
        print print_format.format(*['r_test'] + r_test.tolist())
        print print_format.format(*['mi_train'] + mi_train.tolist())
        print print_format.format(*['mi_test'] + mi_test.tolist())

    return reg_c_opt


def _convex_subproblem_solver(x, y, solver, reg_l,
                              reg_c=1.0,
                              weights=np.array([]),
                              bias_i=np.array([]),
                              w0=np.array([])):

    # Use identity reg_l if empty
    if reg_l.size == 0:
        reg_l = np.eye(x.shape[1])

    # Get the full regularization matrix reg_ltl
    if sparse.issparse(reg_l):
        reg_l_tmp = reg_l.toarray()
        reg_ltl = np.dot(reg_l_tmp.T, reg_l_tmp)
    else:
        reg_ltl = np.dot(reg_l.T, reg_l)

    # Add a reg_ltl zero column and row for the bias term (if missing)
    if x.shape[1] - reg_ltl.shape[0] == 1:
        reg_ltl = np.insert(reg_ltl, 0, 0, axis=0)
        reg_ltl = np.insert(reg_ltl, 0, 0, axis=1)

    # Add a unity weights if none are provided
    if weights.size != y.size:
        weights = np.ones(y.size)

    # Add zero biases if none are provided
    if bias_i.size != y.size:
        bias_i = np.zeros(y.size)

    y_tmp = np.empty_like(y.ravel(), dtype=np.float64)

    if solver == 'linreg':
        y_tmp[:] = y.ravel() - bias_i.ravel()
        w = _least_squares_solver(x, y_tmp, reg_ltl, reg_c=reg_c)

    elif solver == 'logreg':
        y_tmp[y.ravel() > 0] = 1.0
        y_tmp[y.ravel() <= 0] = -1.0
        solver_id = 0
        w = _liblinear_solver(x, y_tmp, reg_ltl,
                              reg_c=reg_c,
                              weights=weights,
                              bias_i=bias_i,
                              w0=w0,
                              solver_id=solver_id)

    elif solver == 'poireg':
        y_tmp[:] = np.float64(np.around(y).ravel())
        solver_id = 1
        w = _liblinear_solver(x, y_tmp, reg_ltl,
                              reg_c=reg_c,
                              weights=weights,
                              bias_i=bias_i,
                              w0=w0,
                              solver_id=solver_id)

    else:
        raise Exception("Unknown solver: {}".format(solver))

    return w


def _least_squares_solver(x, y, reg_ltl, reg_c=1):
    """ Finds the regularized least squares solution

    This function determines the parameters w by either Cholesky
    decomposition or a matrix inversion.

    Hw = X'y, where H = (c_ss + lambda I) and c_ss is the stimulus
    auto-correlation matrix and lambda a regularization paramter

    1. Cholesky
    LL' = H
    Lw_tmp = X'y
    L'w = w_tmp

    2. Matrix inversion
    w = inv(H)X'y

    Tests have indicated that the Cholesky method solves the problem in half
    the time compared to the matrix inversion method

    Args:
        x: input matrix
        y: spike count matrix
        reg_c: regularization parameter
    Returns:
        w:
    Raise
    """

    assert reg_ltl.shape[0] == x.shape[1]

    reg_lambda = 1./reg_c

    # Solving using cholesky decomposition
    # L = np.linalg.cholesky(np.dot(x.T, x) + reg_lambda * reg_ltl)
    # w_tmp = np.linalg.solve(L, np.dot(x.T, y))
    # w = np.linalg.solve(L.T, w_tmp)

    # Solving using a matrix inverse
    c_inv = np.linalg.inv(np.dot(x.T, x) + reg_lambda * reg_ltl)
    w = np.dot(np.dot(c_inv, x.T), y)

    return w.ravel()


def _liblinear_solver(x, y, reg_ltl,
                      reg_c=1.0,
                      weights=np.array([]),
                      bias_i=np.array([]),
                      w0=np.array([]),
                      solver_id=0):
    """ Call upon liblinear's truncated trust region conjugate newton solver

    :param x: input array
    :param y: class labels
    :param weights: sample specific weight
    :param bias_i: sample specific biases
    :param w0: initial guess
    :param tol: tolerance
    :param reg_c: regularization parameter
    :return param: w
    """

    assert reg_ltl.shape[0] == x.shape[1]

    tol = 1e-5
    nr_thread = mp.cpu_count()

    # Find the solution
    set_verbosity_wrap(0)  # suppress output from liblinear
    w = my_train_wrap(x,
                      y.ravel(),
                      reg_ltl,
                      weights.ravel(),
                      bias_i.ravel(),
                      w0.ravel(),
                      solver_id,
                      tol,
                      reg_c,
                      nr_thread)

    return w.ravel()


#############################################
# ---- Gradient ascent/descent solvers ---- #
def gradient_ascent(x, y, params, opt_dict):
    """ Fitting MID models using gradient ascent

    This version of gradient ascent is specialized for fitting MID models.
    It utilizes a line search along the gradient and selects the step length
    probabilistically based on a decaying temperature.

    Options:
        n_bins: number of bins to use when appriximating distributions
        print: print progress: True / False

    Args:
        x: input array
        y: output array
        params: field parameters
        opt_dict: optimization parameters
    Returns:
        params_val: field parameters
    Raise

    """

    # Initialize gradient ascent
    temp = 1e-2  # temperature
    alpha = 0.  # momentum
    rnd_state = 0  # random seed when dividing into training and validation sets
    n_bins = opt_dict['n_bins']
    verbose = opt_dict['verbose']
    n_iterations = 1
    converged = False

    # Initiate arrays for storing the momentum in
    rf_deltas = [Field(rf.shape, None, rf.multilin) for rf in params.rfs]
    cf_deltas = [Field(cf.shape, None, cf.multilin) for cf in params.cfs]

    # Keep in mind that validation values are likely higher due to a larger bias
    mi_train, mi_val = rf_obj_fun.mid_mi(x, y, params, n_bins, rnd_state)
    if verbose:
        print "Running gradient ascent"
        print "{:15s}{:15s}{:15s}{:15s}{:15s}".format(
            'Iteration', 'eta avg.', 'Temp', 'I_train', 'I_val')
        print "{:<15d}{:<15.1e}{:<15.1e}{:<15.3f}{:<15.3f}".format(
            0, 0, temp, mi_train, mi_val)
        plot_fun.interactive(True)
        fig = plot_fun.plot_parameters(params)

    # Run gradient descent
    n_search_steps = 10
    etas = np.ones(5)
    eta_search = np.logspace(-5, 1, n_search_steps)
    mi_search = np.empty_like(eta_search)
    mi_val_best = mi_val
    params_val = deepcopy(params)
    while not converged:
        # Get gradients
        rf_ders, cf_ders = \
            rf_obj_fun.mid_mi_der(x, y, params, n_bins, rnd_state)

        for search_idx, eta_tmp in zip(range(n_search_steps), eta_search):
            params_tmp = deepcopy(params)
            # Make a gradient ascent step
            for mid_id in range(len(params_tmp.rfs)):
                params_tmp.rfs[mid_id] = _field_step_update(
                    params_tmp.rfs[mid_id], rf_ders[mid_id], rf_deltas[mid_id],
                    eta=eta_tmp, alpha=alpha, sign=1)
            for cf_idx in range(len(params_tmp.cfs)):
                cf_ders[cf_idx].bias = 0  # keep the CF bias at unity
                params_tmp.cfs[cf_idx] = _field_step_update(
                    params_tmp.cfs[cf_idx], cf_ders[cf_idx], cf_deltas[cf_idx],
                    eta=eta_tmp, alpha=alpha, sign=1)

            mi_search[search_idx], _ = \
                rf_obj_fun.mid_mi(x, y, params_tmp, n_bins, rnd_state)

        # Select eta probabilistically
        exp_tmp = np.exp(-(mi_train-mi_search) / temp)
        prob_tmp = np.cumsum(exp_tmp) / exp_tmp.sum()
        r = np.random.rand()
        eta = eta_search[np.argwhere(r <= prob_tmp)[0][0]]

        # Make a gradient ascent step with the best eta value
        for mid_id in range(len(params.rfs)):
            params.rfs[mid_id] = _field_step_update(
                params.rfs[mid_id], rf_ders[mid_id], rf_deltas[mid_id],
                eta=eta, alpha=alpha, sign=1)
        for cf_idx in range(len(params.cfs)):
            cf_ders[cf_idx].bias = 0  # keep the CF bias at unity
            params.cfs[cf_idx] = _field_step_update(
                params.cfs[cf_idx], cf_ders[cf_idx], cf_deltas[cf_idx],
                eta=eta, alpha=alpha, sign=1)

        # Normalize filters when optimizing an multi-filter LN model
        if len(params.cfs) == 0:
            if params.rfs[mid_id].multilin:
                for part in params.rfs[mid_id].parts:
                    part /= np.linalg.norm(part)
            else:
                params.rfs[mid_id].field /= \
                    np.linalg.norm(params.rfs[mid_id].field)

        # Always keep a copy of the parameters with the highest mi_val value
        mi_train, mi_val = rf_obj_fun.mid_mi(x, y, params, n_bins, rnd_state)
        if mi_val > mi_val_best:
            params_val = deepcopy(params)
            mi_val_best = mi_val
        etas[0:-1] = etas[1:]
        etas[-1] = eta
        temp *= 0.95

        # Print progress and check for convergence
        if verbose:
            if np.mod(n_iterations, 1) == 0:
                print "{:<15d}{:<15.1e}{:<15.1e}{:<15.3f}{:<15.3f}".format(
                    n_iterations, etas.mean(), temp, mi_train, mi_val)
                plot_fun.plot_parameters(params, fig)

        if n_iterations == 200 or etas.mean() < 5e-4:
            if verbose:
                plot_fun.interactive(False)
                plot_fun.close(fig)
            converged = True
        else:
            n_iterations += 1

    return params_val


def gd_with_search(x, y, params, opt_dict):
    """ Fitting RF/CF models using gradient descent

    This solver utilizes a validation set for setting a regularization
    parameter during training. The regularization parameter is set based upon
    validation set performance.

    Options:
        error: error function: 'mse' or 'neg_log_lik_bernoulli/poisson'
        eta: learning rate
        alpha: momentum term
        batch_frac: fraction of training samples to use in each batch
        print: print progress: True / False

    Args:
        x: input array
        y: output array
        params: rf and possible cf field parameters
        opt_dict: optimization parameters
    Returns:
        params: found field parameters
    Raise

    """

    rf_win_size = params.rfs[0].shape[0]
    reg_c_values = np.logspace(-3, -0, 4)
    score_train = np.zeros_like(reg_c_values)
    score_val = np.zeros_like(reg_c_values)
    params_cv = []

    split_idx = int(round(3/4.*y.size))
    x_train = x[:split_idx+rf_win_size-1, :, :]
    y_train = y[:split_idx, :]
    x_val = x[split_idx:, :, :]
    y_val = y[split_idx:, :]

    for reg_c, reg_c_idx in zip(reg_c_values, range(len(reg_c_values))):

        params_tmp = deepcopy(params)
        for rf in params_tmp.rfs:
            rf.reg_c = reg_c
        params_tmp = gradient_descent(x_train, y_train, params_tmp, opt_dict)
        params_cv.append(params_tmp)

        # Projection scores
        x_nd_full_train = rf_obj_fun.z_rf_der(x_train, params_tmp)
        x_nd_full_val = rf_obj_fun.z_rf_der(x_val, params_tmp)
        z_train = inner_product(x_nd_full_train, params_tmp.rfs)
        z_val = inner_product(x_nd_full_val, params_tmp.rfs)
        if params_tmp.rf_type == 'qn_rfs':
            z_train = z_train.sum(axis=1).reshape(z_train.shape[0], 1)
            z_val = z_val.sum(axis=1).reshape(z_val.shape[0], 1)
        if opt_dict['solver'] == 'mse':
            y_hat_train = z_train
            y_hat_val = z_val
        elif opt_dict['solver'] == 'logreg':
            y_hat_train = rf_obj_fun.inv_bernoulli_link(z_train)
            y_hat_val = rf_obj_fun.inv_bernoulli_link(z_val)
        elif opt_dict['solver'] == 'poireg':
            y_hat_train = rf_obj_fun.inv_poisson_link(z_train)
            y_hat_val = rf_obj_fun.inv_poisson_link(z_val)

        # score_train[reg_c_idx] = \
        #     calculate_r(y_hat_train, y_train)
        # score_val[reg_c_idx] = \
        #     calculate_r(y_hat_val, y_val)
        score_train[reg_c_idx] = \
            rf_obj_fun.mutual_information(z_train, y_train, 20)[0]
        score_val[reg_c_idx] = \
            rf_obj_fun.mutual_information(z_val, y_val, 20)[0]

    # Print results
    best_idx = score_val.argmax()
    reg_c_opt = reg_c_values[best_idx]
    print "Optimal reg_c found to be: {:1.1e}".format(reg_c_opt)
    print_format = '{:10s}' + ''.join(['{:10.1e}'] * reg_c_values.size)
    print print_format.format(*['reg_c'] + reg_c_values.tolist())
    print_format = '{:10s}' + ''.join(['{:10.3f}'] * reg_c_values.size)
    print print_format.format(*['score_train'] + score_train.tolist())
    print print_format.format(*['score_val'] + score_val.tolist())

    # Do a final gradient descent run with the optimal regularization
    params_tmp = params_cv[best_idx]
    for rf in params_tmp.rfs:
        rf.reg_c = reg_c
        rf.reg_c_from_cv = True
    params_tmp = gradient_descent(x, y, params_tmp, opt_dict)

    return params_tmp


def gradient_descent(x, y, params, opt_dict):
    """ Fitting RF/CF models using gradient descent

    This version of gradient descent implements a bold driver strategy where eta
    is increased as long the error measure decreases and decreased otherwise.
    The momentum term is element specific and reset whenever the partial
    derivative for an element changes sign.

    Options:
        error: error function: 'mse' or 'neg_log_lik_bernoulli/poisson'
        eta: learning rate
        alpha: momentum term
        batch_frac: fraction of training samples to use in each batch
        reg_c: regularization parameter
        print: print progress: True / False

    Args:
        x: input array
        y: output array
        params: rf and possible cf field parameters
        opt_dict: optimization parameters
    Returns:
        params: found field parameters
    Raise

    """

    # Select error function
    if opt_dict['solver'] == 'linreg':
        obj_fun = rf_obj_fun.mean_squared_error
        error_fun_der = rf_obj_fun.mean_squared_error_der
        error_name = 'MSE'
    elif opt_dict['solver'] == 'logreg':
        obj_fun = rf_obj_fun.neg_log_lik_bernoulli
        error_fun_der = rf_obj_fun.neg_log_lik_bernoulli_der
        error_name = 'Neg log lik'
    elif opt_dict['solver'] == 'poireg':
        obj_fun = rf_obj_fun.neg_log_lik_poisson
        error_fun_der = rf_obj_fun.neg_log_lik_poisson_der
        error_name = 'Neg log lik'
    else:
        raise Exception("Unknown error fun: {}".format(opt_dict['error']))

    # Weights for each sample
    weights = np.ones(y.shape)
    if opt_dict['solver'] == 'logreg':
        weights[y > 1] = y[y > 1]  # add weight if multiple spikes occurred

    # Initialize gradient descent
    eta = opt_dict['eta']
    alpha = opt_dict['alpha']
    verbose = opt_dict['verbose']
    if 'batch_frac' in opt_dict:
        batch_frac = opt_dict['batch_frac']
    else:
        batch_frac = 1.0
    n_iterations = 1
    der_norm = 1e10
    converged = False

    # Initiate arrays for storing the momentum in
    rf_deltas = [Field(rf.shape, None, rf.multilin) for rf in params.rfs]
    cf_deltas = [Field(cf.shape, None, cf.multilin) for cf in params.cfs]

    obj_fun_val = obj_fun(x, y, params, weights=weights)
    if verbose:
        print "Running gradient descent"
        print "eta_init: %1.5f\talpha: %1.5f" % (eta, alpha)
        print "{:15s}{:15s}{:15s}".format('Iteration', 'eta', error_name)
        print "{:<15d}{:<15.1e}{:<15.1f}".format(0, eta, obj_fun_val)
        plot_fun.interactive(True)
        fig = plot_fun.plot_parameters(params)

    # Run gradient descent
    err_frac = 0
    obj_fun_val_frac = obj_fun_val
    batch_size = np.int64(batch_frac*x.shape[0])
    while not converged:

        # Get gradients
        batch_start = np.random.randint(0, x.shape[0]-batch_size+1, 1)[0]
        batch_x_end = batch_start+batch_size
        batch_y_end = batch_start+batch_size - params.rfs[0].shape[0] + 1
        rf_ders, cf_ders, act_fun_ders = \
            error_fun_der(x[batch_start:batch_x_end, :, :],
                          y[batch_start:batch_y_end],
                          params,
                          weights=weights)

        # Protection from sudden increases in gradient magnitude
        der_norm_old = der_norm
        der_norm = 0
        for rf_der in rf_ders:
            der_norm += np.linalg.norm(rf_der.field)
        for cf_der in cf_ders:
            der_norm += np.linalg.norm(cf_der.field)
        for act_fun_der in act_fun_ders:
            der_norm += np.linalg.norm(act_fun_der.alpha)
        norm_frac = der_norm/der_norm_old
        if n_iterations > 1 and norm_frac > 3:
            eta /= norm_frac
            # print "Forced learning rate decrease: %2.1f" % norm_frac

        # Update rf parameters
        for rf_idx in range(len(params.rfs)):
            params.rfs[rf_idx] = _field_step_update(
                params.rfs[rf_idx], rf_ders[rf_idx], rf_deltas[rf_idx],
                eta=eta, alpha=alpha, sign=-1)

        # Update cf parameters
        if params.cfs:
            for cf_idx in range(len(params.cfs)):
                cf_ders[cf_idx].bias = 0  # keep the CF bias at unity
                params.cfs[cf_idx] = _field_step_update(
                    params.cfs[cf_idx], cf_ders[cf_idx], cf_deltas[cf_idx],
                    eta=eta, alpha=alpha, sign=-1)

        # Calculate the new error value
        obj_fun_val_old = obj_fun_val
        obj_fun_val = obj_fun(x, y, params, weights=weights)

        # Implement bold driver strategy for adjusting eta and reset the
        # momentum term if we overshoot
        if obj_fun_val > obj_fun_val_old:
            eta *= 0.5
        else:
            eta *= 1.1

        # Print progress
        if np.mod(n_iterations, 10) == 0:
            err_frac = (obj_fun_val - obj_fun_val_frac) / \
                       np.abs(obj_fun_val_frac)
            obj_fun_val_frac = obj_fun_val
            if verbose:
                print "{:<15d}{:<15.1e}{:<15.1f}{:<15.1e}".format(
                    n_iterations, eta, obj_fun_val, err_frac)
                plot_fun.plot_parameters(params, fig)

        # Convergence check
        if (n_iterations > 40) and (-1e-4 < err_frac < 0):
            if verbose:
                print "{:<15d}{:<15.1e}{:<15.1f}".format(
                    n_iterations, eta, obj_fun_val)
                plot_fun.interactive(False)
                plot_fun.close(fig)
            converged = True
        else:
            n_iterations += 1

    return params


def _field_step_update(field, field_der, field_delta,
                       eta, alpha, sign=-1):
    """ Gradient step update for RFs and CFs

        Args:
            field: field to update
            field_der: partial derivatives for each field element
            field_delta: receptive fields
            eta: learning rate
            alpha: momentum term
            sign: descent (-1) or ascent (1)
        Returns:
            field: updated field
        Raise

        """

    # Multi linear model
    if field.multilin:
        n_parts = len(field.parts)
        for part_idx in range(n_parts):
            signs = np.sign(field_delta.parts[part_idx] *
                            field_der.parts[part_idx])
            field_delta.parts[part_idx][signs < 0] = 0
            field_delta.parts[part_idx] = \
                alpha * field_delta.parts[part_idx] + \
                eta * field_der.parts[part_idx]
            field.parts[part_idx] += sign * field_delta.parts[part_idx]
        field.field = outer_product(field.parts)
    # Full model
    else:
        signs = np.sign(field_delta.field * field_der.field)
        field_delta.field[signs < 0] = 0
        field_delta.field = alpha * field_delta.field + eta * field_der.field
        field.field += sign * field_delta.field

    # Bias
    signs = np.sign(field_delta.bias * field_der.bias)
    if signs > 0:
        field_delta.bias = alpha * field_delta.bias + eta * field_der.bias
    else:
        field_delta.bias = eta * field_der.bias
    field.bias += sign * field_delta.bias

    return field


def _alpha_step_update(act_fun, act_fun_der, act_fun_delta,
                        eta, alpha, sign=-1):
    """ Gradient step update for CF activation function paramters

        Args:
            act_fun: act_fun to update
            act_fun_der: partial derivatives for the act_fun
            act_fun_delta: act_fun deltas
            eta: learning rate
            alpha: momentum term
            sign: descent (-1) or ascent (1)
        Returns:
            act_fun: updated act_fun
        Raise

    """

    signs = np.sign(act_fun_delta.alpha * act_fun_der.alpha)
    act_fun_delta.alpha[signs < 0] = 0
    act_fun_delta.alpha = alpha * act_fun_delta.alpha + eta * act_fun_der.alpha
    act_fun.alpha += sign * act_fun_delta.alpha

    return act_fun


##############################
# ---- LN model solvers ---- #
def ln_solver(x, y, params, opt_dict):

    # Read options
    n_dims = opt_dict['n_dims']
    solver = opt_dict['solver']
    verbose = opt_dict['verbose']

    # Initializing
    significance = Significance(solver)
    if solver == 'stc':
        subspace_solver = _find_stc_subspace
    elif solver == 'stc_w':
        subspace_solver = _find_stc_subspace_whiten
    elif solver == 'istac':
        subspace_solver = _find_istac_subspace
    else:
        raise Exception("Unknown LN solver: {}".format(opt_dict['solver']))
    rf_shape = params.rfs[0].shape

    # Make a 2D matrix
    x_nd = add_fake_dimension(x, rf_shape[0])
    x_nd_full = x_nd.copy()
    n_samples = x_nd_full.shape[0]
    rf_size = reduce(mul, rf_shape)
    x_2d = x_nd_full.reshape(n_samples, rf_size)

    # Autoscale
    x_2d -= x_2d.mean(axis=0)
    x_2d /= x_2d.std(axis=0)

    # Find the subspace and related values
    basis, values = subspace_solver(x_2d, y, verbose=verbose)
    significance.add_values(values)

    # Create an field out of every significant basis vector
    rfs = []
    most_sig_idxs, _ = significance.get_most_sig_idxs(n_dims)
    for idx in most_sig_idxs:
        field = Field(rf_shape, None)
        field.field = basis[:, idx].reshape(rf_shape)
        rfs.append(field)
    params.rfs = rfs

    return params, significance


def _find_stc_subspace(x_2d, y, verbose=1):
    """ Finds LN-model filters using STC analysis

    We estimate vectors that span a stimulus subspace using STC.
    Vectors are found as the eigenvectors of the STC matrix

    See Touryan et al. (2002) for details.

    Args:
        x_2d: mean centered input array
        y: output array
    Returns:
        basis:
        eigen_values
    Raise

    """

    sta, stc = sta_and_stc(x_2d, y)

    eig_val, eig_vec = np.linalg.eig(stc)
    sort_idxs = np.argsort(eig_val)

    basis = eig_vec[:, sort_idxs]
    eigen_values = eig_val[sort_idxs]

    return basis, eigen_values


def _find_stc_subspace_whiten(x_2d, y, verbose=1):
    """ Finds LN-model filters using STC analysis

    We estimate vectors that span a stimulus subspace using a whitened STC.
    Vectors are found as the eigenvectors of the STC matrix

    See Touryan et al. (2005) for details.

    Args:
        x_2d: mean centered input array
        y: output array
    Returns:
        basis:
        eigen_values
    Raise

    """

    cut_off_frac = 0.75
    cut_off = np.int64(cut_off_frac*x_2d.shape[1])

    cov = np.cov(x_2d.T)
    eig_val_cov, eig_vec_cov = np.linalg.eig(cov)
    sort_cov_idxs = np.argsort(eig_val_cov)
    sigma_inv = np.diag(1. / np.sqrt(eig_val_cov[sort_cov_idxs]))
    x_2d_w = np.dot(x_2d,
                    np.dot(eig_vec_cov[:, sort_cov_idxs],
                           sigma_inv[:, cut_off:]))

    sta, stc = sta_and_stc(x_2d_w, y)

    eig_val, eig_vec = np.linalg.eig(stc)
    sort_idxs = np.argsort(eig_val)

    basis = eig_vec[:, sort_idxs]
    basis = np.dot(basis.T,
                   np.dot(sigma_inv[cut_off:, cut_off:],
                          eig_vec_cov[:, sort_cov_idxs[cut_off:]].T))
    basis = basis.T

    eigen_values = eig_val[sort_idxs]

    return basis, eigen_values


def _find_istac_subspace(x_2d, y, verbose=1):
    """ Finds LN-model filters using the iSTAC model

    We estimate vectors that span a stimulus subspace from the STA and STC.
    New vectors are found using a gradient ascend procedure where
    the new vector is orthonormalized after each update.

    See Pillow and Simoncelli (2006) for details.
    Observer that the expression provided for the gradient is incorrect!

    Options:


    Args:
        x: input array
        y: output array
        params: field parameters
        opt_dict: optimization parameters
    Returns:
        params
    Raise

    """

    # Parameters
    n_best = 4
    n_dims = 8

    # Get field dimensions

    # STA and STC
    sta, stc = sta_and_stc(x_2d, y)
    stastaT = np.outer(sta.ravel(), sta.ravel())

    eig_val_a, eig_vec_a = np.linalg.eig(stc)
    eig_val_b, eig_vec_b = np.linalg.eig(stc + stastaT)
    idxs_a = np.argsort(np.abs(eig_val_a))
    idxs_b = np.argsort(np.abs(eig_val_b))

    # Select top n most important eigen vectors as possible initial guesses
    basis = np.zeros([x_2d.shape[1], n_dims])
    b_0 = np.hstack((eig_vec_a[:, idxs_a[-n_best:]],
                     eig_vec_b[:, idxs_b[-n_best:]]))
    mi_values = np.zeros(n_dims)
    mi_tot = 0

    # Loop to find selected number of dimensions
    for dim in range(n_dims):

        mi_values_tmp = np.zeros([2*n_best])
        b_found = np.zeros([x_2d.shape[1], 2*n_best])
        for probe_id in range(2*n_best):

            # Create a initial basis guess and orthonormalize
            b_tmp = np.hstack([basis[:, :dim],
                               b_0[:, probe_id].reshape(x_2d.shape[1], 1)])
            b_tmp = np.linalg.qr(b_tmp)[0]

            # Initialize gradient ascent
            mi = rf_obj_fun.istac_mi(b_tmp, sta, stc)
            converged = False
            n_iteration = 0
            eta = 1e3

            while not converged:

                # Update step and orthonormalization
                b_der = rf_obj_fun.istac_mi_der(b_tmp, sta, stc)
                b_tmp[:, dim] += eta*b_der[:, dim]
                b_tmp = np.linalg.qr(b_tmp)[0]

                # Evaluation of the objective function
                mi_old = mi
                mi = rf_obj_fun.istac_mi(b_tmp, sta, stc)
                diff = mi - mi_old

                # Adaptive learning rate
                if diff < 0:
                    eta *= 0.5

                # Convergance check
                if 0 < diff < 1e-5 and n_iteration > 10:
                    converged = True

                # Max iteration check
                if n_iteration == 500:
                    if verbose:
                        print "Max iterations reached"
                    converged = True
                else:
                    n_iteration += 1

            # Store the new basis and the objective function value
            b_found[:, probe_id] = b_tmp[:, dim]
            mi_values_tmp[probe_id] = mi

        # Select the new basis column based upon highest objective fun value
        best_idx = np.argmax(mi_values_tmp)
        basis[:, dim] = b_found[:, best_idx]
        mi_values[dim] = np.max(mi_values_tmp) - mi_tot
        mi_tot = np.max(mi_values_tmp)

        if verbose:
            print "---{}. MI: {}---".format(dim+1, mi_values[dim])

    # Orthogonality check
    bTb = np.dot(basis.T, basis)
    bTb[np.diag_indices_from(bTb)] = 0
    if np.abs(bTb).max() > 1e-10:
        print "Warning! B not orthogonal"

    return basis, mi_values


#######################################
# ---- Low-rank QN model solvers ---- #
def qn_solver(x, y, params, opt_dict):

    # Read options
    solver = opt_dict['solver']
    w0 = opt_dict['w0']
    verbose = opt_dict['verbose']

    reg_c = params.rfs[0].reg_c
    rf_shape = params.rfs[0].shape

    # Initializing
    if solver == 'mne':
        subspace_solver = _find_mne_params_mem
    else:
        raise Exception("Unknown LN solver: {}".format(opt_dict['solver']))

    # Make a 2D matrix
    x_nd = add_fake_dimension(x, rf_shape[0])
    x_nd_full = x_nd.copy()
    n_samples = x_nd_full.shape[0]
    rf_size = reduce(mul, rf_shape)
    x_2d = x_nd_full.reshape(n_samples, rf_size)

    # Autoscale
    # x_2d -= x_2d.mean(axis=0)
    # x_2d /= x_2d.std(axis=0)

    # Find QN parameters
    # [1, x1, x2,...,xn, x1x1, x1x2, x2x2, x1x3, x2x3,...,xnxn]
    poly_params = subspace_solver(x_2d, y, reg_c=reg_c, w0=w0, verbose=verbose)
    params.qn_poly = poly_params

    return params


def qn_extract_filters(x, y, params, opt_dict):

    # Read options
    solver = opt_dict['solver']
    n_filters = opt_dict['n_filters']

    rf_shape = params.rfs[0].shape

    # Reconstruct h and J from found parameters
    rf_size = reduce(mul, rf_shape)
    bias = params.qn_poly[0].copy()
    h = params.qn_poly[1:rf_size+1].copy()
    J = np.zeros([rf_size, rf_size])
    w_idx = rf_size+1
    for i in range(rf_size):
        for j in range(i+1):
            J[i, j] += params.qn_poly[w_idx] / 2
            J[j, i] += params.qn_poly[w_idx] / 2
            w_idx += 1

    # Diagonalize J
    eig_val, eig_vec = np.linalg.eig(J)
    sort_idxs = np.argsort(eig_val)
    basis = eig_vec[:, sort_idxs]
    eigen_values = eig_val[sort_idxs]

    significance = Significance(solver)
    significance.add_values(eigen_values)

    # Create an field out of every significant basis vector or h
    most_sig_idxs, most_sig_vals = significance.get_most_sig_idxs(n_filters)
    x_nd_full = rf_obj_fun.z_rf_der(x, params)
    z = np.zeros([x_nd_full.shape[0], 1])
    for idx, rf in zip(most_sig_idxs, params.rfs):
        filter_id = np.argwhere(most_sig_idxs == idx)[0][0]
        # Select quadratic filters first
        if filter_id < n_filters - 1:
            rf.field = basis[:, idx].reshape(rf_shape)
            rf.qn_lambda = eigen_values[idx]
            rf.qn_square = True
            rf.bias = 0
            h -= np.dot(h, basis[:, idx]) * basis[:, idx]
            z_tmp = inner_product(x_nd_full, [rf])
            z += (z_tmp**2) * eigen_values[idx]
        # Select the last filter based upon which that leads to higher MI
        else:
            x_nd_full = rf_obj_fun.z_rf_der(x, params)
            rf.field = h.reshape(rf_shape)
            z_linear = inner_product(x_nd_full, [rf]) + z
            rf.field = basis[:, idx].reshape(rf_shape)
            z_quadratic = inner_product(x_nd_full, [rf])
            z_quadratic = (z_quadratic**2) * eigen_values[idx] + z
            mi_linear = rf_obj_fun.mutual_information(z_linear, y, 20)
            mi_quadratic = rf_obj_fun.mutual_information(z_quadratic, y, 20)
            if mi_linear > mi_quadratic:
                rf.field = h.reshape(rf_shape)
                rf.qn_lambda = 1
                rf.qn_square = False
                rf.bias = bias
            else:
                rf.field = basis[:, idx].reshape(rf_shape)
                rf.qn_lambda = eigen_values[idx]
                rf.qn_square = True
                rf.bias = bias

    return params, significance


def _find_mne_params_mem(x_2d, y, reg_c, w0=np.array([]), verbose=1):
    """ Relevant dimensions from 2nd order polynomial logistic regression model

    We estimate relevant vectors that span a stimulus subspace by diagonalizing
    the function f(x) = bias + h^T*x + x^T*J*s in the logistic regression model
    P(spike|x) = 1 / (1 + exp(f(x)))

    see Fitzgerald et al. (2011a, 2011b) for details.

    Args:
        x_2d: input array
        y: output array
        verbose:
    Returns:
        basis: basis vectors
        eigen_values:
    Raise

    """

    # Add bias term
    x_2d_bias = np.hstack([np.ones([x_2d.shape[0], 1]), x_2d.copy()])

    # Make sure y is binary with labels -1 and 1
    y_sign = np.zeros(y.shape)
    y_sign[y == 0] = -1
    y_sign[y > 0] = 1

    # Fit a model
    w_poly = _fit_mne_liblinear(x_2d_bias, y_sign.ravel(), reg_c=reg_c, w0=w0)

    return w_poly


def _fit_mne_liblinear(x, y,
                       reg_c=1e-1,
                       w0=np.array([])):
    """ Call upon liblinear's truncated trust region conjugate newton solver

    :param x: input array
    :param y: class labels
    :param weights: sample specific weight
    :param bias_i: sample specific biases
    :param w0: initial guess
    :param tol: tolerance
    :param reg_c: regularization parameter
    :return param: w
    """

    tol = 1e-3
    nr_thread = mp.cpu_count()
    reg_ltl = np.array([0.]).reshape(1, 1)
    solver_id = 2

    # Add a unity weights if none are provided
    weights = np.ones(y.size)

    # Add zero biases if none are provided
    bias_i = np.zeros(y.size)

    # Find the solution
    set_verbosity_wrap(0)  # suppress output from liblinear
    w = my_train_wrap(x,
                      y.ravel(),
                      reg_ltl,
                      weights.ravel(),
                      bias_i.ravel(),
                      w0.ravel(),
                      solver_id,
                      tol,
                      reg_c,
                      nr_thread)

    return w.ravel()