#!/usr/bin/python
"""
" @section DESCRIPTION
" RF model classes
"""

import os
import copy
import numpy as np
from operator import mul
from time import time, sleep
from scipy.ndimage import correlate as im_corr

import plotting.plotting_functions as plt_fun
import rf_obj_funs_and_grads as rf_obj_fun
import rf_optimization as rf_opt
from rf_field_format import Field
from rf_piecewise_lin_act_fun import ActFun
from rf_parameters import Parameters
from rf_evaluation import Evaluation
from rf_helper import add_fake_dimension, get_insignificant_basis,\
    inner_product, outer_product, sta_and_stc, load_saved_models
from rf_nonlinearity import Nonlinearity


class RFModel(object):
    """Base class for different receptive field models"""

    def __init__(self):
        self.params = []
        self.multilin = False
        self.solver = None
        self.init_method = None

        # Receptive fields
        self.n_rfs = None
        self.rf_type = 'lin_rfs'
        self.rf_truncated = False
        self.rf_significance = []
        self.qn_poly_params = np.array([])

        # Context fields
        self.cf_zero_origin = False
        self.cf_mapping = None
        self.cf_type = None
        self.cf_alignment = 'edge'
        self.cf_act_fun = None
        self.pos_sol = None

        # Regularization
        self.reg_c_init = None
        self.reg_type = None

        # Nonlinearity
        self.nonlinearity = []
        self.dist_res = 20  # histogram resolution in bins per dimension

        self.name = 'Base'
        self.trained = []

        # Evaluation
        self.obj_fun_val = []
        self.eval_train = []
        self.eval_test = []

        # Training meta data
        self.data = None
        self.basis_fun = None
        self.labels = None
        self.ticks = None
        self.dt = None

    def __str__(self):
        return "{} model".format(self.name)

    def initialize_params(self, rf_shape, cf_shape, load_path=None):

        # Default initialization
        params = Parameters()

        if self.name[:3] == 'MID':
            rf_init_values = 'randn'
            cf_init_values = 'randn'
        else:
            rf_init_values = 'zeros'
            cf_init_values = 'zeros'
        rf_multilin = self.multilin
        params.init_rfs(rf_shape, self.reg_c_init,
                        type=self.rf_type,
                        n_rfs=self.n_rfs,
                        multilin=rf_multilin,
                        reg_type=self.reg_type,
                        init_values=rf_init_values)
        cf_multilin = self.multilin
        params.init_cfs(cf_shape, self.cf_mapping, self.cf_type,
                        self.reg_c_init,
                        multilin=cf_multilin,
                        reg_type=self.reg_type,
                        init_values=cf_init_values,
                        zero_origin=self.cf_zero_origin,
                        alignment=self.cf_alignment)
        params.init_act_funs(self.cf_act_fun, self.reg_c_init)

        # RF bias not used with MID models
        if self.name[:3] == 'MID':
            for rf in params.rfs:
                rf.bias = 0

        # Try to load parameters if requested
        params_loaded = False
        split_idx = len(self.params)
        if self.init_method is not None:

            # Load appropriate models
            loaded_models = load_saved_models(load_path, tag=self.init_method)

            # Check if these models are trained on the same data,
            # using the same basis function, and for the same split
            for model in loaded_models:
                # Do not load multilinear models
                if not model.multilin:
                    same_data = model.data == self.data
                    same_basis_fun = model.basis_fun == self.basis_fun
                    same_split = split_idx < len(model.params)
                    if same_data and same_basis_fun and same_split:
                        params_tmp = copy.deepcopy(model.params[split_idx])
                        params_loaded = True
                        print "Model parameters loaded!"

        # Extract loaded parameters
        if params_loaded:
            for rf, rf_tmp in zip(params.rfs, params_tmp.rfs):
                rf.field = rf_tmp.field
                rf.parts = rf_tmp.parts
                if rf.multilin and len(rf.parts) == 0:
                    rf.make_separable(self.reg_c_init, self.reg_type)
            for cf, cf_tmp in zip(params.cfs, params_tmp.cfs):
                cf.field = cf_tmp.field
                cf.parts = cf_tmp.parts
                if cf.multilin and len(cf.parts) == 0:
                    cf.make_separable(self.reg_c_init, self.reg_type)
            if hasattr(params_tmp, 'cf_act_funs'):
                for act_fun, act_fun_tmp in zip(params.cf_act_funs,
                                                params_tmp.cf_act_funs):
                    act_fun.base_peaks = act_fun_tmp.base_peaks
                    act_fun.alpha = act_fun_tmp.alpha
            if hasattr(params_tmp, 'qn_poly'):
                params.qn_poly = params_tmp.qn_poly
            elif hasattr(model, 'qn_poly_params'): # backward compatibility
                params.qn_poly = model.qn_poly_params[split_idx]
            elif hasattr(model, 'poly_params'):  # backward compatibility
                params.qn_poly = model.poly_params[split_idx]
        else:
            if self.cf_type == 'ctx_nl' or self.cf_type == 'subunit':
                raise Exception("{} type context models require a "
                                "loaded ctx type model as an initial "
                                "guess".format(self.cf_type))

        self.params.append(params)
        self.nonlinearity.append(Nonlinearity(self.dist_res))
        self.eval_train.append(Evaluation())
        self.eval_test.append(Evaluation())
        self.trained.append(False)

    def train(self, x, y, rf_win_size, split_idx=0, verbose=1):
        """Trains the RF model

        Args:
            x: input array
            y: output array
            rf_win_size: rf window size in time bins
            split_idx: fold index when estimating models for each KFold
            verbose: show output if greater than 0
        Returns:

        Raises:

        """

        self._check_initialization(split_idx)

        self._estimate_nonlinearity(x, y, split_idx)

        self.trained[split_idx] = True

    def predict(self, x, split_idx=0):
        """ Predict response to input

        IMPORTANT!
        Predictions only use the first two receptive fields on the LN-model.

        Args:
            x: input array
            split_idx: use parameters from specific KFold
        Returns:
            y_hat: model output
        Raises:
            No parameters found
            Model not trained
        """

        self._check_initialization(split_idx)
        self._check_training(split_idx)

        # Calculate the similarity score and make a prediction
        nonlinearity = self.nonlinearity[split_idx]
        z = self._calculate_z(x, split_idx=split_idx)
        y_hat = nonlinearity.predict(z)

        return y_hat

    def set_meta_data(self, name, basis_fun, labels=[], ticks=[], dt=1.):
        self.data = name
        self.basis_fun = basis_fun
        self.labels = labels
        self.ticks = ticks
        self.dt = dt
        self.name += '_' + basis_fun[:3]

    def evaluate(self, x, y, set, split_idx):
        """ Evaluate a trained model on data

        IMPORTANT!
        Evaluation only uses the first two receptive fields on the LN-model.

        :param x:
        :param y:
        :param set:
        :return:
        """

        self._check_initialization(split_idx)
        self._check_training(split_idx)

        y_mean = np.vstack(y.mean(axis=1))
        y_sum = np.vstack(y.sum(axis=1))

        z = self._calculate_z(x, split_idx=split_idx)
        z_null = self._calculate_z_null(x, y_sum, split_idx=split_idx)
        y_hat = self.predict(x, split_idx)

        if set == 'train':
            if len(self.eval_train) < split_idx + 1:
                for fold_idx_tmp in range(len(self.eval_train), split_idx + 1):
                    self.eval_train.append(Evaluation())
            self.eval_train[split_idx].evaluate_mi_stc(z, z_null, y_sum)
            # self.eval_train[split_idx].evaluate_mi_qe(z, y_sum)
            # self.eval_train[split_idx].evaluate_mi_raw(z, y_sum)
            self.eval_train[split_idx].evaluate_r(y_hat, y)

        elif set == 'test':
            if len(self.eval_test) < split_idx + 1:
                for fold_idx_tmp in range(len(self.eval_test), split_idx + 1):
                    self.eval_test.append(Evaluation())
            self.eval_test[split_idx].evaluate_mi_stc(z, z_null, y_sum)
            # self.eval_test[split_idx].evaluate_mi_qe(z, y_sum)
            # self.eval_test[split_idx].evaluate_mi_raw(z, y_sum)
            self.eval_test[split_idx].evaluate_r(y_hat, y)

    def _add_obj_fun_val(self, obj_fun_val, split_idx):
        if len(self.obj_fun_val) < split_idx + 1:
            for fold_idx_tmp in range(len(self.obj_fun_val), split_idx + 1):
                self.obj_fun_val.append(None)
        self.obj_fun_val[split_idx] = obj_fun_val

    def _add_rf_significancen(self, split_idx):
        if len(self.rf_significance) < split_idx + 1:
            for fold_idx_tmp in range(len(self.rf_significance), split_idx + 1):
                self.rf_significance.append(None)

    def _calculate_z(self, x, split_idx=0):
        """ Calculate the similarity score vector z

        :param x:
        :param y:
        :return z:
        """

        # Get params for the selected fold
        params = self.params[split_idx]

        # Calculate the similarity score
        if hasattr(params, 'rf_type') and params.rf_type == 'pos_inv_rfs':
            z = rf_obj_fun.z_rf_pos_inv(x, params)
        else:
            # Get the full x-matrix (CF processed if CFs are used)
            x_nd_full = rf_obj_fun.z_rf_der(x, params)
            z = inner_product(x_nd_full, params.rfs)
            if hasattr(params, 'rf_type') and params.rf_type == 'max_rfs':
                z = z.max(axis=1).reshape(z.shape[0], 1)
            elif hasattr(params, 'rf_type') and params.rf_type == 'qn_rfs':
                z = z.sum(axis=1).reshape(z.shape[0], 1)

        return z

    def _calculate_z_null(self, x, y, split_idx=0):
        """ Calculates the null similarity score vector z_null

        :param x:
        :param y:
        :param split_idx:
        :return z_null:
        """

        # Get params for the selected fold
        params = self.params[split_idx]

        # Get the full x-matrix (CF processed if CFs are used)
        x_nd_full = rf_obj_fun.z_rf_der(x, params)

        # Get the null vector from STC analysis
        rfs_null = []
        null_fields = get_insignificant_basis(x, y, params.rfs[0].shape)
        for null_field in null_fields:
            rf_null = Field(params.rfs[0].shape, reg_c=None)
            rf_null.field = null_field
            rfs_null.append(rf_null)

        # Calculate the null similarity score
        z_null = inner_product(x_nd_full, rfs_null)

        return z_null

    def _estimate_nonlinearity(self, x, y, split_idx):
        """Estimate a general nonlinearity mapping z to spike probabilities

        Args:
            x: input array
            y: output array
            split_idx:
        Returns:

        Raises:

        """

        # Calculate the similarity score
        z = self._calculate_z(x, split_idx=split_idx)

        # Non-linearity, P(spike|score)
        self.nonlinearity[split_idx] = Nonlinearity(self.dist_res)
        self.nonlinearity[split_idx].estimate(z, y, solver=self.solver)

    def _check_initialization(self, split_idx):
        if len(self.params) < split_idx + 1:
            raise Exception("No parameters for split: {}".format(split_idx))

    def _check_training(self, split_idx):
        if not self.trained[split_idx]:
            raise Exception("Model NOT trained on split: {}".format(split_idx))

    def _print_model_info(self):
        info = "{:25s}{:15s}{:15s}".format(
            self.name, self.cf_mapping, str(self.multilin))
        print info


# Only included for backwards compatibility
class LinModel(RFModel):
    """RF model class: Generalized Linear Model (GLM)"""

    def __init__(self, params_dict):
        """configures the model according to provided parameters

        Args:
            params_dict: model parameters
        Returns:

        Raises:
            Unknown optimization method

        """

        super(GenLinModel, self).__init__()
        self.n_rfs = params_dict['n_rfs']
        self.solver = params_dict['solver']
        self.optimization = params_dict['optimization']
        self.multilin = params_dict['multilin']
        self.cf_mapping = params_dict['cf_mapping']
        self.cf_alignment = params_dict['cf_alignment']
        self.pos_sol = params_dict['pos_sol']
        self.init_method = params_dict['init']

        if self.solver == 'linreg':
            self.name = 'LinReg'
        elif self.solver == 'logreg':
            self.name = 'LogReg'
        elif self.solver == 'poireg':
            self.name = 'PoiReg'
        else:
            raise Exception("Unknown solver: {}".format(self.solver))

        if self.cf_mapping == 'same':
            self.name += 'Ctx'
            if not self.pos_sol:
                self.name += 'Neg'
        elif self.cf_mapping == 'dim0_split':
            self.name += 'CtxDim0'
        elif self.cf_mapping == 'dim1_split':
            self.name += 'CtxDim1'
        elif self.cf_mapping == 'dim2_split':
            self.name += 'CtxDim2'
        elif self.cf_mapping == 'level':
            self.name += 'CtxInt'

        if self.cf_alignment == 'center':
            self.name += '_c'

        self.reg_c_init = params_dict['reg_c_init']
        self.reg_type = params_dict['reg_type']
        self.name += '_' + self.reg_type

        if self.multilin:
            self.name += '_ml'

        if self.optimization == 'gradient_descent':
            self.name += '_gd'

        self._print_model_info()

    def train(self, x, y, rf_win_size, split_idx=0, verbose=1):
        """Trains the RF model

        Args:
            x: input array
            y: output array
            rf_win_size: rf window size in time bins
            split_idx: training fold index
            verbose: print progress
        Returns:

        Raises:

        """

        # 1. Check initialization
        self._check_initialization(split_idx)

        # 2. Handle data with y-values from multiple trials
        if self.solver == 'linreg':
            y_tmp = np.vstack(y.mean(axis=1))
        elif self.solver == 'logreg':
            y_tmp = np.vstack(y.mean(axis=1))
        elif self.solver == 'poireg':
            y_tmp = np.vstack(y.sum(axis=1))
        else:
            raise Exception("Unknown solver: {}".format(self.solver))

        # 3. Find receptive and context filters using the model dependent method
        # OPTION 1. Alternating multiple RFs solver
        if self.optimization == 'alternating_rfs':
            opt_dict = {'solver': self.solver,
                        'verbose': verbose > 0}
            self.params[split_idx] = rf_opt.alternating_solver_rfs(
                x, y_tmp, self.params[split_idx], opt_dict)

        # OPTION 2. Alternating context model solver
        if self.optimization == 'alternating':
            opt_dict = {'solver': self.solver,
                        'pos_sol': self.pos_sol,
                        'verbose': verbose > 0}
            if self.multilin:
                self.params[split_idx] = rf_opt.alternating_solver_ctx_multilin(
                    x, y_tmp, self.params[split_idx], opt_dict)
            else:
                self.params[split_idx] = rf_opt.alternating_solver_ctx(
                    x, y_tmp, self.params[split_idx], opt_dict)

        # OPTION 3. Gradient descent
        elif self.optimization == 'gradient_descent':
            # Run gradient descent to find CF parameters
            opt_dict = {'eta': 1e-6,
                        'alpha': 0.9,
                        'batch_frac': 1.0,
                        'solver': self.solver,
                        'verbose': verbose > 0}
            self.params[split_idx] = rf_opt.gradient_descent(
                x, y_tmp, self.params[split_idx], opt_dict)

        # Get final objection function value
        weights = np.ones(y_tmp.shape)
        if self.solver == 'logreg':
            weights[y_tmp > 1] = y_tmp[y_tmp > 1]

        if self.solver == 'linreg':
            obj_fun = rf_obj_fun.mean_squared_error
        elif self.solver == 'logreg':
            obj_fun = rf_obj_fun.neg_log_lik_bernoulli
        elif self.solver == 'poireg':
            obj_fun = rf_obj_fun.neg_log_lik_poisson

        obj_fun_val = obj_fun(x, y_tmp, self.params[split_idx],
                              weights=weights,
                              reg_c=self.params[split_idx].reg_c)
        self._add_obj_fun_val(obj_fun_val, split_idx)

        # 4. Estimate the nonlinearity
        self._estimate_nonlinearity(x, y_tmp, split_idx)

        self.trained[split_idx] = True

    def predict(self, x, split_idx=0):
        """ Predict response to input

        Args:
            x: input array
        Returns:
            y_hat: model output
        Raises:
            Model not trained
        """

        self._check_initialization(split_idx)
        self._check_training(split_idx)

        # Get the similarity score
        z = self._calculate_z(x, split_idx=split_idx)

        if self.solver == 'linreg':
            y_hat = z
        if self.solver == 'logreg':
            y_hat = rf_obj_fun.inv_bernoulli_link(z)
        elif self.solver == 'poireg':
            y_hat = rf_obj_fun.inv_poisson_link(z)

        return y_hat


class GenLinModel(RFModel):
    """RF model class: Generalized Linear Model (GLM)"""

    def __init__(self, params_dict):
        """configures the model according to provided parameters

        Args:
            params_dict: model parameters
        Returns:

        Raises:
            Unknown optimization method

        """

        super(GenLinModel, self).__init__()
        self.rf_type = 'max_rfs'
        self.n_rfs = params_dict['n_rfs']
        self.solver = params_dict['solver']
        self.multilin = params_dict['multilin']
        self.init_method = params_dict['init']

        if self.solver == 'linreg':
            self.name = 'LinReg'
        elif self.solver == 'logreg':
            self.name = 'LogReg'
        elif self.solver == 'poireg':
            self.name = 'PoiReg'
        else:
            raise Exception("Unknown solver: {}".format(self.solver))

        self.reg_c_init = params_dict['reg_c_init']
        self.reg_type = params_dict['reg_type']
        self.name += '_' + self.reg_type

        if self.multilin:
            self.name += '_ml'

        self._print_model_info()

    def train(self, x, y, rf_win_size, split_idx=0, verbose=1):
        """Trains the RF model

        Args:
            x: input array
            y: output array
            rf_win_size: rf window size in time bins
            split_idx: training fold index
            verbose: print progress
        Returns:

        Raises:

        """

        # 1. Check initialization
        self._check_initialization(split_idx)

        # 2. Handle data with y-values from multiple trials
        if self.solver == 'linreg':
            y_tmp = np.vstack(y.mean(axis=1))
        elif self.solver == 'logreg':
            y_tmp = np.vstack(y.mean(axis=1))
        elif self.solver == 'poireg':
            y_tmp = np.vstack(y.sum(axis=1))
        else:
            raise Exception("Unknown solver: {}".format(self.solver))

        # 3. Find receptive fields using an appropriate solver
        opt_dict = {'solver': self.solver,
                    'verbose': verbose > 0}
        self.params[split_idx] = rf_opt.alternating_solver_ctx(
            x, y_tmp, self.params[split_idx], opt_dict)

        # Get final objection function value
        weights = np.ones(y_tmp.shape)
        if self.solver == 'logreg':
            weights[y_tmp > 1] = y_tmp[y_tmp > 1]

        if self.solver == 'linreg':
            obj_fun = rf_obj_fun.mean_squared_error
        elif self.solver == 'logreg':
            obj_fun = rf_obj_fun.neg_log_lik_bernoulli
        elif self.solver == 'poireg':
            obj_fun = rf_obj_fun.neg_log_lik_poisson

        obj_fun_val = obj_fun(x, y_tmp, self.params[split_idx], weights=weights)
        self._add_obj_fun_val(obj_fun_val, split_idx)

        # 4. Estimate the nonlinearity
        self._estimate_nonlinearity(x, y_tmp, split_idx)

        self.trained[split_idx] = True

    def predict(self, x, split_idx=0):
        """ Predict response to input

        Args:
            x: input array
            split_idx:
        Returns:
            y_hat: model output
        Raises:
            Model not trained
        """

        self._check_initialization(split_idx)
        self._check_training(split_idx)

        # Get the similarity score
        z = self._calculate_z(x, split_idx=split_idx)

        if self.solver == 'linreg':
            y_hat = z
        if self.solver == 'logreg':
            y_hat = rf_obj_fun.inv_bernoulli_link(z)
        elif self.solver == 'poireg':
            y_hat = rf_obj_fun.inv_poisson_link(z)

        return y_hat


class CtxModel(RFModel):
    """RF model class: Context model | subunit model)"""

    def __init__(self, params_dict):
        """configures the model according to provided parameters

        Args:
            params_dict: model parameters
        Returns:

        Raises:
            Unknown optimization method

        """

        super(CtxModel, self).__init__()
        self.rf_type = 'lin_rfs'
        self.n_rfs = 1
        self.rf_truncated = params_dict['rf_truncated']
        self.solver = params_dict['solver']
        self.multilin = params_dict['multilin']
        self.cf_mapping = params_dict['cf_mapping']
        self.cf_type = params_dict['cf_type']
        self.cf_alignment = params_dict['cf_alignment']
        self.cf_act_fun = params_dict['cf_act_fun']
        self.pos_sol = params_dict['pos_sol']
        self.init_method = params_dict['init']

        if self.solver == 'linreg':
            self.name = 'LinReg'
        elif self.solver == 'logreg':
            self.name = 'LogReg'
        elif self.solver == 'poireg':
            self.name = 'PoiReg'
        else:
            raise Exception("Unknown solver: {}".format(self.solver))

        if self.cf_type == 'ctx':
            self.name += 'Ctx'
        elif self.cf_type == 'ctx_nl':
            self.name += 'CtxNl'
        elif self.cf_type == 'subunit':
            self.name += 'Sub'

        if self.rf_truncated:
            self.name += 'T'

        if self.cf_mapping == 'temporal':
            self.name += 'Temp'

        if self.cf_act_fun is not None:
            self.name += self.cf_act_fun[:4].capitalize()
        else:
            self.cf_zero_origin = True

        if self.cf_alignment == 'center':
            self.name += '_c'

        self.reg_c_init = params_dict['reg_c_init']
        self.reg_type = params_dict['reg_type']
        self.name += '_' + self.reg_type

        if self.multilin:
            self.name += '_ml'

        self._print_model_info()

    def train(self, x, y, rf_win_size, split_idx=0, verbose=1):
        """Trains the RF model

        Args:
            x: input array
            y: output array
            rf_win_size: rf window size in time bins
            split_idx: training fold index
            verbose: print progress
        Returns:

        Raises:

        """

        # 1. Check initialization
        self._check_initialization(split_idx)

        # 2. Handle data with y-values from multiple trials
        if self.solver == 'linreg':
            y_tmp = np.vstack(y.mean(axis=1))
        elif self.solver == 'logreg':
            y_tmp = np.vstack(y.mean(axis=1))
        elif self.solver == 'poireg':
            y_tmp = np.vstack(y.sum(axis=1))
        else:
            raise Exception("Unknown solver: {}".format(self.solver))

        # 3. Find receptive and context fields using the model dependent method
        # OPTION 1. No CF activation function
        if self.cf_act_fun is None:
            opt_dict = {'solver': self.solver,
                        'pos_sol': self.pos_sol,
                        'verbose': verbose > 0}
            self.params[split_idx] = rf_opt.alternating_solver_ctx(
                x, y_tmp, self.params[split_idx], opt_dict)

        # OPTION 2. Fixed CF activation function
        if self.cf_act_fun and self.cf_act_fun != 'adaptive':
            opt_dict = {'solver': self.solver,
                        'pos_sol': self.pos_sol,
                        'verbose': verbose > 0}
            self.params[split_idx] = rf_opt.alternating_solver_ctx_fixed(
                x, y_tmp, self.params[split_idx], opt_dict)

        # OPTION 3. Adaptive CF activation function
        if self.cf_act_fun == 'adaptive':
            opt_dict = {'solver': self.solver,
                        'rf_truncated': self.rf_truncated,
                        'first_fold': split_idx == 0,
                        'pos_sol': self.pos_sol,
                        'verbose': verbose > 0}
            self.params[split_idx] = rf_opt.alternating_solver_ctx_adaptive(
                x, y_tmp, self.params[split_idx], opt_dict)

        # Get final objection function value
        weights = np.ones(y_tmp.shape)
        if self.solver == 'logreg':
            weights[y_tmp > 1] = y_tmp[y_tmp > 1]

        if self.solver == 'linreg':
            obj_fun = rf_obj_fun.mean_squared_error
        elif self.solver == 'logreg':
            obj_fun = rf_obj_fun.neg_log_lik_bernoulli
        elif self.solver == 'poireg':
            obj_fun = rf_obj_fun.neg_log_lik_poisson

        obj_fun_val = obj_fun(x, y_tmp, self.params[split_idx], weights=weights)
        self._add_obj_fun_val(obj_fun_val, split_idx)

        # 4. Estimate the nonlinearity
        self._estimate_nonlinearity(x, y_tmp, split_idx)

        self.trained[split_idx] = True

    def predict(self, x, split_idx=0):
        """ Predict response to input

        Args:
            x: input array
            split_idx: fold index
        Returns:
            y_hat: model output
        Raises:
            Model not trained
        """

        self._check_initialization(split_idx)
        self._check_training(split_idx)

        # Get the similarity score
        z = self._calculate_z(x, split_idx=split_idx)

        if self.solver == 'linreg':
            y_hat = z
        if self.solver == 'logreg':
            y_hat = rf_obj_fun.inv_bernoulli_link(z)
        elif self.solver == 'poireg':
            y_hat = rf_obj_fun.inv_poisson_link(z)

        return y_hat


class MIDModel(RFModel):
    """RF model class: Maximally Informative Dimensions (MID)

    This model class can train:
    - Multi-filter LN models (Sharpee et.a. 2004 and Sharpee et.a. 2006)
    - Position invariant models (Eickenberg et al. 2012)
    - Context models (Westo and May, 2017)
    by maximizing the single spike information directly.

    """

    def __init__(self, params_dict):
        """configures the model according to provided parameters

        Args:
            params_dict: model parameters
        Returns:

        Raises:
            Unknown optimization method

        """

        super(MIDModel, self).__init__()
        self.name = 'MID'
        self.init_method = params_dict['init']
        self.rf_type = params_dict['rf_type']
        self.n_rfs = params_dict['n_rfs']
        self.multilin = params_dict['multilin']
        self.cf_mapping = params_dict['cf_mapping']
        self.cf_alignment = params_dict['cf_alignment']

        if self.cf_mapping == 'same':
            self.name += 'Ctx'
        elif self.cf_mapping == 'level':
            self.name += 'CtxInt'
        else:
            self.name += str(self.n_rfs)

        if self.init_method is not None:
            self.name += '_' + self.init_method

        self._print_model_info()

    def train(self, x, y, rf_win_size, split_idx=0, verbose=1):
        """Trains the RF model

        Args:
            x: input array
            y: output array
            rf_win_size: rf window size in time bins
        Returns:

        Raises:

        """

        # 1. Check initialization
        y_sum = np.vstack(y.sum(axis=1))
        self._check_initialization(split_idx)

        # 2. Find field parameters through gradient ascent
        opt_dict = {'n_bins': self.dist_res,
                    'verbose': verbose > 0}
        self.params[split_idx] = rf_opt.gradient_ascent(
            x, y_sum, self.params[split_idx], opt_dict)

        # Get final objection function value
        mi_val = rf_obj_fun.mid_mi(
            x, y_sum, self.params[split_idx], self.dist_res)
        self._add_obj_fun_val(mi_val, split_idx)

        # 3. Estimate the nonlinearity
        self._estimate_nonlinearity(x, y_sum, split_idx)

        self.trained[split_idx] = True


class LNModel(RFModel):
    """RF model class: multi-filter linear-nonlinear (LN) model

    Multi-filter LN models can be estimated using STC or iSTAC analysis.
    These methods both assume a Gaussian stimulus distribution

    See Touryan et al. (2002) and Pillow and Simoncelli (2006) for details.

    """

    def __init__(self, params_dict):
        """configures the model according to provided parameters

        Args:
            params_dict: model parameters
        Returns:

        Raises:
            Unknown optimization method

        """

        super(LNModel, self).__init__()
        self.solver = params_dict['solver']
        self.n_rfs = params_dict['n_rfs']

        if self.solver == 'stc':
            self.name = 'STC'
        elif self.solver == 'stc_w':
            self.name = 'STCw'
        elif self.solver == 'istac':
            self.name = 'iSTAC'
        else:
            raise Exception("Unknown solver: {}".format(self.solver))

        self.name += str(self.n_rfs)
        self._print_model_info()

    def train(self, x, y, rf_win_size, split_idx=0, verbose=1):
        """Trains the RF model

        Args:
            x: input array
            y: output array
            rf_win_size: rf window size in time bins
            split_idx: fold index when estimating models for each KFold
            verbose: show output if greater than 0
        Returns:

        Raises:

        """

        # 1. Check initialization
        self._check_initialization(split_idx)

        # 2. Find LN-filters using the model dependent solver
        self._add_rf_significancen(split_idx)
        opt_dict = {'n_dims': self.n_rfs,
                    'solver': self.solver,
                    'verbose': verbose > 0}
        self.params[split_idx], self.rf_significance[split_idx] = \
            rf_opt.ln_solver(x, y, self.params[split_idx], opt_dict)

        # 3. Estimate the nonlinearity (only first two filters)
        self._estimate_nonlinearity(x, y, split_idx)

        self.trained[split_idx] = True


class QNModel(RFModel):
    """RF model class: quadratic-nonlinear (QN) model

    QN-models for a subest of LN models with a one-dimensional nonlinearity.
    These can hence be used together with more than two fitlers as they
    do not suffer from the curse of dimensionality.

    See Fitzgerald et al. (2011) and Sharpee (2013) for details

    """

    def __init__(self, params_dict):
        """configures the model according to provided parameters

        Args:
            params_dict: model parameters
        Returns:

        Raises:
            Unknown optimization method

        """

        super(QNModel, self).__init__()
        self.solver = params_dict['solver']
        self.rf_type = 'qn_rfs'
        self.n_rfs = params_dict['n_rfs']
        self.init_method = params_dict['init']
        self.reg_c_init = params_dict['reg_c_init']
        self.reg_type = 'norm'

        if self.solver == 'mne':
            self.name = 'MNE'
        else:
            raise Exception("Unknown solver: {}".format(self.solver))

        self.name += str(self.n_rfs)
        self.name += '_C' + str(self.reg_c_init)
        self._print_model_info()

    def train(self, x, y, rf_win_size, split_idx=0, verbose=1):
        """Trains the RF model

        Args:
            x: input array
            y: output array
            rf_win_size: rf window size in time bins
            split_idx: fold index when estimating models for each KFold
            verbose: show output if greater than 0
        Returns:

        Raises:

        """

        # 1. Check initialization
        y_tmp = np.vstack(y.mean(axis=1))
        self._check_initialization(split_idx)

        # 2. Find QN polynomial parameters using the selected solver
        if self.params[split_idx].qn_poly.size == 0:
            # Use an initial guess if available
            if self.params[0].qn_poly.size > 0:
                w0 = self.params[0].qn_poly
            else:
                w0 = np.array([])
            opt_dict = {'solver': self.solver,
                        'w0': w0,
                        'verbose': verbose > 0}
            self.params[split_idx] = \
                rf_opt.qn_solver(x, y_tmp, self.params[split_idx], opt_dict)

        # 3. Extract filters from the polynomial's parameters
        self._add_rf_significancen(split_idx)
        opt_dict = {'solver': self.solver,
                    'n_filters': self.n_rfs}
        self.params[split_idx], self.rf_significance[split_idx] = \
            rf_opt.qn_extract_filters(x, y_tmp,
                                      self.params[split_idx],
                                      opt_dict)

        # 4. Fine tune filter weights by optimizing the QN obj. fun. directly
        opt_dict = {'eta': 1e-6,
                    'alpha': 0.95,
                    'batch_frac': 1.0,
                    'solver': 'logreg',
                    'verbose': verbose > 0}
        self.params[split_idx] = rf_opt.gd_with_search(
            x, y_tmp, self.params[split_idx], opt_dict)
        # Normalize quadratic filters to unity length
        for rf in self.params[split_idx].rfs:
            if rf.qn_square:
                tmp = np.linalg.norm(rf.field)
                rf.field /= tmp
                rf.qn_lambda *= (tmp**2)

        # 5. Estimate the nonlinearity
        self._estimate_nonlinearity(x, y_tmp, split_idx)

        self.trained[split_idx] = True


def gradient_checking(x, y, rf_win_size):
    """Gradient checking for all error_measure functions using provided data

    Args:
        x: input array
        y: output array
        rf_win_size:
    Returns:

    Raises:
        Unknown error measure
    """

    diff_lim = 1e-5
    n_bins = 11

    # Field shapes
    x_nd = add_fake_dimension(x, rf_win_size)
    rf_shape = x_nd.shape[1:]
    cf_shape = [5, 5, 5]  # default shape
    cf_shape = _modify_cf_shape(rf_shape, cf_shape)  # modification if needed

    # Combinations to test
    # error_measures = ['mse', 'neg_log_lik_bernoulli', 'neg_log_lik_poisson']
    error_measures = ['neg_log_lik_bernoulli']
    multilin_options = [True]
    cf_mappings = ['same']
    cf_alignments = ['edge']
    cf_types= ['ctx', 'ctx_nl', 'subunit']
    reg_c_vals = [1e-3, 1e1]
    istac_dims = [1, 2]
    mids = []

    print "Running gradient checking for MSE and log-likelihood models"
    print "{:20}{:15}{:15}{:15}{:15}{:15}{:15}{:15}{:15}".format(
        'Error', 'MultiLin', 'CF mapping', 'CF alignment', 'CF type',
        'Reg C', 'Field', 'Diff', 'Status')

    # Part 1: MSE or neg-log-likelihood error functions
    for error_measure in error_measures:

        weights = np.array([])

        # Select proper functions
        if error_measure == 'mse':
            err_fun = rf_obj_fun.mean_squared_error
            err_fun_der = rf_obj_fun.mean_squared_error_der
        elif error_measure == 'neg_log_lik_bernoulli':
            err_fun = rf_obj_fun.neg_log_lik_bernoulli
            err_fun_der = rf_obj_fun.neg_log_lik_bernoulli_der
        elif error_measure == 'neg_log_lik_poisson':
            err_fun = rf_obj_fun.neg_log_lik_poisson
            err_fun_der = rf_obj_fun.neg_log_lik_poisson_der
        else:
            raise Exception("Unknown error measure: {}".format(error_measure))

        for multilin in multilin_options:
            for cf_mapping in cf_mappings:
                for cf_alignment in cf_alignments:
                    for cf_type in cf_types:
                        for reg_c in reg_c_vals:
                            # Initialize rf/cf models
                            params = Parameters()
                            params.init_rfs(rf_shape, reg_c,
                                            multilin=multilin,
                                            init_values='randn')
                            params.init_cfs(cf_shape, cf_mapping,
                                            type=cf_type,
                                            reg_c=reg_c,
                                            multilin=multilin,
                                            init_values='randn',
                                            zero_origin=False,
                                            alignment=cf_alignment)
                            params.init_act_funs('random', reg_c)
                            params.update_context_map()

                            rf_ders_ana, cf_ders_ana, act_fun_ders_ana = \
                                err_fun_der(x, y, params,
                                            weights=weights)
                            rf_ders_num, cf_ders_num, act_fun_ders_num = \
                                _num_param_der(x, y, params,
                                               error_fun=err_fun,
                                               weights=weights)

                            # Check RF parameters
                            for rf_idx in range(len(params.rfs)):
                                rf_diff = _field_diff(rf_ders_ana[rf_idx],
                                                      rf_ders_num[rf_idx])
                                outcome = 'passed' \
                                    if rf_diff < diff_lim else 'FAILED'
                                print "{:20.15}{:15}{:15}{:15}{:15}{:<15.1e}" \
                                      "{:15}{:<15.1e}{:15}".format(
                                      error_measure, str(multilin), cf_mapping,
                                      cf_alignment, cf_type, reg_c,
                                      'rf', rf_diff, outcome)
                                if outcome == 'FAILED':
                                    plt_fun.plot_field_gradients(
                                        rf_ders_ana[rf_idx],
                                        rf_ders_num[rf_idx])

                            # Check CF parameters
                            for cf_idx in range(len(params.cfs)):
                                cf_diff = _field_diff(cf_ders_ana[cf_idx],
                                                      cf_ders_num[cf_idx])
                                outcome = 'passed' \
                                    if cf_diff < diff_lim else 'FAILED'
                                print "{:20.15}{:15}{:15}{:15}{:15}{:<15.1e}" \
                                      "{:15}{:<15.1e}{:15}".format(
                                      error_measure, str(multilin), cf_mapping,
                                      cf_alignment, cf_type, reg_c,
                                      'cf', cf_diff, outcome)
                                if outcome == 'FAILED':
                                    plt_fun.plot_field_gradients(
                                        cf_ders_ana[cf_idx],
                                        cf_ders_num[cf_idx])

                            # Check activation function parameters
                            for act_fun_idx in range(len(params.cf_act_funs)):
                                alpha_diff = _alpha_diff(
                                    act_fun_ders_ana[act_fun_idx],
                                    act_fun_ders_num[act_fun_idx])
                                outcome = 'passed' \
                                    if alpha_diff < diff_lim else 'FAILED'
                                print "{:20.15}{:15}{:15}{:15}{:15}{:<15.1e}" \
                                      "{:15}{:<15.1e}{:15}".format(
                                      error_measure, str(multilin), cf_mapping,
                                      cf_alignment, cf_type, reg_c,
                                      'act_fun', alpha_diff, outcome)
                                if outcome == 'FAILED':
                                    plt_fun.plot_act_fun_gradients(
                                        act_fun_ders_ana[cf_idx],
                                        act_fun_ders_num[cf_idx])

    # Part 2: iSTAC
    print "Running gradient checking for iSTAC"
    print "{:15}{:15}{:15}".format('Dims', 'Diff', 'Status')

    rf_size = reduce(mul, rf_shape)
    x_nd = add_fake_dimension(x, rf_shape[0])
    x_2d = x_nd.reshape(x_nd.shape[0], rf_size)
    sta, stc = sta_and_stc(x_2d, y)

    # Loop over different number of bases vectors
    for dim in istac_dims:

        # Initialize b
        b = np.random.randn(sta.size*dim).reshape(sta.size, dim)
        b /= np.linalg.norm(b, axis=0)

        # Get the gradients
        b_der_ana = rf_obj_fun.istac_mi_der(b, sta, stc)
        b_der_num = _num_istack_der(b, sta, stc)

        for field_id in range(dim):
            ana_field_der = Field(rf_shape, reg_c=None, init_values='zeros')
            ana_field_der.field = b_der_ana[:, field_id].reshape(rf_shape)
            num_field_der = Field(rf_shape, reg_c=None, init_values='zeros')
            num_field_der.field = b_der_num[:, field_id].reshape(rf_shape)

            # Check RF parameters
            rmse = _field_diff(ana_field_der, num_field_der)
            outcome = 'passed' if rmse < diff_lim else 'FAILED'
            print "{:<15}{:<15.1e}{:15}".format(dim, rmse, outcome)
            if outcome == 'FAILED':
                plt_fun.plot_field_gradients(ana_field_der, num_field_der)

    # Part 3: MID
    # This part is not always worth running as the partial derivatives are very
    # noisy due to the probability functions being estimated with histograms.
    print "Maximally Informative Dimensions"
    print "{:15}{:15}{:15}{:15}{:15}".format(
        'MIDs', 'CF mapping', 'Field', 'RMSE', 'Status')

    # Loop over different number of mids
    for n_mids in mids:
        for cf_mapping in cf_mappings:
            for multilin in multilin_options:

                # Initialize rf/cf models
                params = Parameters()
                params.init_rfs(rf_shape,
                                reg_c=None,
                                n_rfs=n_mids,
                                multilin=multilin,
                                init_values='randn')
                params.init_cfs(cf_shape, cf_mapping, 'ctx', None,
                                multilin=multilin,
                                init_values='randn')

                # Get the gradients
                rf_ders_ana, cf_ders_ana = \
                    rf_obj_fun.mid_mi_der(x, y, params, n_bins)
                rf_ders_num, cf_ders_num = \
                    _num_mi_der(x, y, params, n_bins)

                # Check mid derivatives
                for mid_idx in range(len(rf_ders_ana)):
                    rf_diff = _field_diff(rf_ders_ana[mid_idx],
                                          rf_ders_num[mid_idx])

                    if rf_diff < diff_lim:
                        status = 'passed'
                    else:
                        status = 'FAILED'
                        plt_fun.plot_field_gradients(rf_ders_ana[mid_idx],
                                                     rf_ders_num[mid_idx])

                    print "{:<15d}{:15}{:15}{:<15.1e}{:15}".format(
                        n_mids, cf_mapping, 'rf', rf_diff, status)

                # Check mid derivatives
                for cf_idx in range(len(cf_ders_ana)):
                    cf_diff = _field_diff(cf_ders_ana[cf_idx],
                                          cf_ders_num[cf_idx])

                    if cf_diff < diff_lim:
                        status = 'passed'
                    else:
                        status = 'FAILED'
                        plt_fun.plot_field_gradients(cf_ders_ana[mid_idx],
                                                     cf_ders_num[mid_idx])

                    print "{:<15d}{:15}{:15}{:<15.1e}{:15}".format(
                        n_mids, cf_mapping, 'cf', cf_diff, status)


def cf_der_checking(x, y, rf_win_size):
    """ Calculate the CF gradient using two methods and compare the results

    Args:
        x: input array
        y: output array
    Returns:

    Raises:
        Unknown error measure
    """
    # Field shapes
    x_nd = add_fake_dimension(x, rf_win_size)
    rf_shape = x_nd.shape[1:]
    cf_shape = [5, 5, 5]  # default shape
    cf_shape = _modify_cf_shape(rf_shape, cf_shape)  # modification if needed

    params = Parameters()
    params.init_rfs(rf_shape,
                    multilin=False,
                    init_values='randn')
    params.init_cfs(cf_shape,
                    method='same',
                    multilin=False,
                    init_values='randn')

    t0 = time()
    rf_ders_ana, cf_ders_ana = \
        rf_obj_fun.neg_log_lik_bernoulli_der(x, y, params,
                                             cf_act_fun='linear',
                                             reg_c=0.0)
    cf_standard_time = time() - t0
    print "CF standard: {:f}".format(cf_standard_time)

    sleep(0.1)

    t0 = time()
    cfs_mat = [Field(cf_shape, None, multilin=False, init_values='zeros')]
    cf_der_mats = rf_obj_fun.cf_bernoulli_der(x, y, params, 'linear')
    cf_der_mat_sum = cf_der_mats[0].sum(axis=1)
    cfs_mat[0].bias = cf_der_mat_sum[0]
    cfs_mat[0].field = cf_der_mat_sum[1:].reshape(cf_shape)
    cf_mat_time = time() - t0
    print "CF matrix: {:f}".format(cf_mat_time)

    plt_fun.plot_field_gradients(cf_ders_ana[0], cfs_mat[0])


# def lines_in_parameter_space(x, y, rf_win_size):
#     """ Calculates the objective function for lines in the parameter space

#     Args:
#         x: input array
#         y: output array
#         rf_win_size: rf window size in time bins
#     Returns:

#     Raises:

#     """

#     n_lines = 50
#     n_points = 25

#     # Add a fake dimension by sliding a window over x
#     x_nd = add_fake_dimension(x, rf_win_size)
#     rf_dims = x_nd.shape[1:]

#     # Modify cf_shape according to the input array
#     cf_dims = [5, 5, 5]  # default shape
#     cf_dims = _modify_cf_shape(rf_dims, cf_dims)

#     # Errors: ['mse', 'neg_log_lik_bernoulli', 'neg_log_lik_poisson']
#     error_measure = 'mse'
#     multilin = False

#     # Preallocation
#     delta = np.linspace(0, 1, n_points)
#     lines = np.empty(n_points)

#     plt.figure()
#     plt.ion()
#     plt.show()
#     for line_idx in range(n_lines):

#         # Initialize rf/cf models
#         rf1 = Field(rf_dims, None, multilin=multilin, init_values='randn')
#         rf2 = Field(rf_dims, None, multilin=multilin, init_values='randn')
#         cf1 = Field(rf_dims, None, multilin=multilin, init_values='randn')
#         cf2 = Field(rf_dims, None, multilin=multilin, init_values='randn')
#         rf = Field(rf_dims, None, multilin=multilin, init_values='zeros')
#         cf = Field(rf_dims, None, multilin=multilin, init_values='zeros')

#         for point_idx in range(n_points):
#             rf.field = rf1.field + delta[point_idx]*rf2.field
#             cf.field = cf1.field + delta[point_idx]*cf2.field
#             lines[point_idx] = rf_obj_fun.mean_squared_error(x, y, [rf], [cf])

#         print line_idx
#         plt.cla()
#         plt.plot(lines.T, 'o-', lw=2)
#         plt.draw()
#         plt.pause(0.25)


def _field_diff(field_a, field_b):
    """ Calculate the RMSE between two fields

        Args:
            field_a: first field
            field_b: second field
        Returns:
            field_diff:
        Raises:

        """

    assert field_a.multilin == field_b.multilin
    assert field_a.shape == field_b.shape

    v_a = []
    v_b = []
    if field_a.multilin:
        for part_idx in range(len(field_a.parts)):
            v_a.append(field_a.parts[part_idx].ravel())
            v_b.append(field_b.parts[part_idx].ravel())
    else:
        v_a.append(field_a.field.ravel())
        v_b.append(field_b.field.ravel())
    v_a.append(field_a.bias)
    v_b.append(field_b.bias)

    v_a = np.hstack(v_a)
    v_b = np.hstack(v_b)

    field_diff = np.linalg.norm(v_a - v_b)**2 / \
                 (np.linalg.norm(v_a)*np.linalg.norm(v_b))

    return field_diff


def _alpha_diff(act_fun_a, act_fun_b):
    """ Calculate the RMSE between two fields

        Args:
            field_a: first field
            field_b: second field
        Returns:
            rmse:
        Raises:

        """

    assert act_fun_a.alpha.shape == act_fun_b.alpha.shape

    v_a = act_fun_a.alpha
    v_b = act_fun_b.alpha

    alpha_diff = np.linalg.norm(v_a - v_b)**2 / \
                 (np.linalg.norm(v_a)*np.linalg.norm(v_b))

    return alpha_diff


def _modify_cf_shape(rf_shape, cf_shape):
    """ Modifies the shape of the context field after the input dimensionality

    Rank 1 field models differ from full field models by having a list of part
    vectors stored under the 'parts' key. For full field models this list is
    instead empty.

    Args:
        rf_shape: [n_time_bins, n_frequencies, n_levels]
        cf_shape: default context shape [time, frequency, level]
    Returns:
        cf_shape:
    Raise

    """

    # Loop over all dimensions and replace to large values in cf_shape with
    # the dimensionality of x_shape.
    for idx in range(len(cf_shape)):
        if cf_shape[idx] > rf_shape[idx]:
            cf_shape[idx] = rf_shape[idx]
            if np.mod(rf_shape[idx], 2) == 0:
                cf_shape[idx] += 1

    return cf_shape


def _num_param_der(x, y, params, error_fun, weights):
    """ Numerical partial derivatives

    This function estimates the numerical partial derivatives for both rf
    and cf parameters on the provided error function

    Args:
        x: input array
        y: output array
        params: field parameters
        error_fun: error function
        weights:
    Returns:
        rf_ders: list with derivatives for rf parts and rf bias
        cf_ders: list with derivatives for cf parts and rf bias
    Raise

    """

    kappa = 1e-5  # step length when determining derivatives numerically
    rf_ders = []
    cf_ders = []
    act_fun_ders = []

    # Partial derivative for RF parameters
    for rf in params.rfs:
        rf_der = _num_field_der(x, y, params, rf, error_fun,
                                weights, kappa)
        rf_ders.append(rf_der)

    # Partial derivative for RF parameters
    for cf in params.cfs:
        cf_der = _num_field_der(x, y, params, cf, error_fun,
                                weights, kappa)
        cf_ders.append(cf_der)

    for act_fun in params.cf_act_funs:
        act_fun_der = _num_act_fun_der(x, y, params, act_fun, error_fun,
                                       weights, kappa)
        act_fun_ders.append(act_fun_der)

    return rf_ders, cf_ders, act_fun_ders


def _num_field_der(x, y, params, field, error_fun, weights, kappa):
    """ Estimates numerical derivatives for RF and CF parameters

    :param x:
    :param y:
    :param params:
    :param field:
    :param error_fun:
    :param weights:
    :param kappa:
    :return:
    """

    field_der = Field(field.shape, reg_c=None, multilin=field.multilin)

    if field.multilin:
        for part_idx in range(len(field.parts)):
            for idx in range(field.parts[part_idx].size):
                field.parts[part_idx].ravel()[idx] += kappa
                field.field = outer_product(field.parts)
                err_plus = error_fun(x, y, params, weights)
                field.parts[part_idx].ravel()[idx] -= 2 * kappa
                field.field = outer_product(field.parts)
                err_minus = error_fun(x, y, params, weights)
                field.parts[part_idx].ravel()[idx] += kappa
                field.field = outer_product(field.parts)
                field_der.parts[part_idx].ravel()[idx] = \
                    (err_plus - err_minus) / 2 / kappa

    else:
        for idx in range(field.field.size):
            field.field.ravel()[idx] += kappa
            err_plus = error_fun(x, y, params, weights)
            field.field.ravel()[idx] -= 2 * kappa
            err_minus = error_fun(x, y, params, weights)
            field.field.ravel()[idx] += kappa
            field_der.field.ravel()[idx] = (err_plus - err_minus) \
                                        / 2 / kappa

    field.bias += kappa
    err_plus = error_fun(x, y, params, weights)
    field.bias -= 2 * kappa
    err_minus = error_fun(x, y, params, weights)
    field.bias += kappa
    field_der.bias = (err_plus - err_minus) / 2 / kappa

    return field_der


def _num_act_fun_der(x, y, params, act_fun, error_fun, weights, kappa):
    """ Estimates numerical derivatives for the cf_act_fun's parameters

    :param x:
    :param y:
    :param params:
    :param field:
    :param error_fun:
    :param weights:
    :param kappa:
    :return act_fun_der:
    """

    alpha = act_fun.alpha
    act_fun_der = ActFun(reg_c=None)

    for idx in range(alpha.size):
        alpha[idx] += kappa
        err_plus = error_fun(x, y, params, weights)
        alpha[idx] -= 2 * kappa
        err_minus = error_fun(x, y, params, weights)
        alpha[idx] += kappa
        act_fun_der.alpha[idx] = (err_plus - err_minus) / 2 / kappa

    return act_fun_der


def _num_istack_der(b, sta, stc):
    """ Numerical partial derivatives (I_iSTAC)

    This function estimates numerical partial derivatives for the vectors
    that span the relevant subspace (Multi filter LN models, iSTAC)

    Args:
        x: input matrix
        y: spike count matrix
        params: field parameters
        n_bins: number of bins to use when approximating the distribution
    Returns:
        rf_ders: partial derivatives for mids
        cf_ders:
    Raise

    """

    kappa = 1e-5
    b_der = np.zeros(b.shape)

    # Partial derivative for RF parameters

    for idx in range(b.size):
        b.ravel()[idx] += kappa
        mi_plus = rf_obj_fun.istac_mi(b, sta, stc)
        b.ravel()[idx] -= 2 * kappa
        mi_minus = rf_obj_fun.istac_mi(b, sta, stc)
        b.ravel()[idx] += kappa
        b_der.ravel()[idx] = (mi_plus - mi_minus) / 2 / kappa

    return b_der


def _num_mi_der(x, y, params, n_bins):
    """ Numerical partial derivatives (I_mid)

    This function estimates the numerical partial derivatives for RFs and CFs
    when these are optimized using the nutual information objective function.

    Args:
        x: input matrix
        y: spike count matrix
        params: field parameters
        n_bins: number of bins to use when approximating the distribution
    Returns:
        rf_ders: partial derivatives for mids
        cf_ders:
    Raise

    """

    kappa = 1e-2
    rf_ders = []
    cf_ders = []

    # Partial derivative for RF parameters
    for rf in params.rfs:
        rf_der = Field(rf.shape, None, multilin=rf.multilin)

        if rf.multilin:
            for part_idx, part in enumerate(rf.parts):
                for idx in range(part.size):
                    part.ravel()[idx] += kappa
                    rf.field = outer_product(rf.parts)
                    mi_plus = rf_obj_fun.mid_mi(x, y, params, n_bins)[0]
                    part.ravel()[idx] -= 2 * kappa
                    rf.field = outer_product(rf.parts)
                    mi_minus = rf_obj_fun.mid_mi(x, y, params, n_bins)[0]
                    part.ravel()[idx] += kappa
                    rf.field = outer_product(rf.parts)
                    rf_der.parts[part_idx].ravel()[idx] = \
                        (mi_plus - mi_minus) / 2 / kappa
        else:
            for idx in range(rf.field.size):
                rf.field.ravel()[idx] += kappa
                mi_plus = rf_obj_fun.mid_mi(x, y, params, n_bins)[0]
                rf.field.ravel()[idx] -= 2 * kappa
                mi_minus = rf_obj_fun.mid_mi(x, y, params, n_bins)[0]
                rf.field.ravel()[idx] += kappa
                rf_der.field.ravel()[idx] = (mi_plus - mi_minus) / 2 / kappa

        rf.bias += kappa
        mi_plus = rf_obj_fun.mid_mi(x, y, params, n_bins)[0]
        rf.bias -= 2 * kappa
        mi_minus = rf_obj_fun.mid_mi(x, y, params, n_bins)[0]
        rf.bias += kappa
        rf_der.bias = (mi_plus - mi_minus) / 2 / kappa

        rf_ders.append(rf_der)

    # Partial derivative for RF parameters
    for cf in params.cfs:
        cf_der = Field(cf.shape, None, multilin=cf.multilin)

        for idx in range(cf.field.size):
            cf.field.ravel()[idx] += kappa
            mi_plus = rf_obj_fun.mid_mi(x, y, params, n_bins)[0]
            cf.field.ravel()[idx] -= 2 * kappa
            mi_minus = rf_obj_fun.mid_mi(x, y, params, n_bins)[0]
            cf.field.ravel()[idx] += kappa
            cf_der.field.ravel()[idx] = (mi_plus - mi_minus) / 2 / kappa

        cf.bias += kappa
        mi_plus = rf_obj_fun.mid_mi(x, y, params, n_bins)[0]
        cf.bias -= 2 * kappa
        mi_minus = rf_obj_fun.mid_mi(x, y, params, n_bins)[0]
        cf.bias += kappa
        cf_der.bias = (mi_plus - mi_minus) / 2 / kappa

        cf_ders.append(cf_der)

    return rf_ders, cf_ders
