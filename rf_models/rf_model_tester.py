#!/usr/bin/python
"""
" @section DESCRIPTION
" Model tester class for training and evaluating RF models
"""


import os
import copy
import cPickle as pickle

import rf_models
import plotting.plotting_functions as plot_fun
from rf_basis_functions import *
from rf_helper import load_saved_models

from sklearn.model_selection import KFold


class ModelTester:
    """Class for training and evaluating RF models"""

    def __init__(self, params):
        """configures the model tester according to the config dictionary

        Args:
            params:
        Returns:

        Raises:
            Exception if the RF model is unknown
        """
        self._rf_win_size = params['rf_win_size']
        self._cf_shape = params['rf_win_size']
        self._basis_fun = params['basis_fun']

        print "\nInitializing models:"
        info = "{:25s}{:15s}{:15s}".format('Name', 'CF-mapping', 'Multi-linear')
        print info
        models = []
        for model in params['rf_models']:
            params = model['params']
            if model['name'] == 'glm':
                models.append(rf_models.GenLinModel(params))
            elif model['name'] == 'ctx':
                models.append(rf_models.CtxModel(params))
            elif model['name'] == 'ln':
                models.append(rf_models.LNModel(params))
            elif model['name'] == 'mid':
                models.append(rf_models.MIDModel(params))
            elif model['name'] == 'qn':
                models.append(rf_models.QNModel(params))
            else:
                raise Exception("Unknown model: {}".format(model['model']))
        self._models = models

        self._x_org = np.array([])
        self._x_basis = np.array([])
        self._x_labes = []
        self._x_ticks = []
        self._y = np.array([])
        self._time_bin_length = None
        self._data_set = None

        self._rf_models_trained = False

    def add_raw_xy_data(self, data):
        """ Add data to train models on

        Args:
            data: dict that must contain:
            ['x'], ['x_labels'], ['x_ticks'], ['y'], ['name'], ['params']['dt']
        Returns:

        Raises:
            Exception if the selected basis function is unknown
        """

        self._x_org = data['x'].copy()
        self._x_labes = data['x_labels']
        self._x_ticks = data['x_ticks']
        self._y = np.vstack(data['y'].copy())
        self._data_set = data['name']
        self._dt = np.float64(data['params']['dt'])  # time bin length (ms)

        # Use basis functions to add a new level dimensions or simply to
        # squash the input values to the interval [0, 1].
        if self._basis_fun == 'original_basis':
            self._x_basis = scaled_original_basis(self._x_org)
        elif self._basis_fun == 'negative_basis':
            self._x_basis = negated_scaled_original_basis(self._x_org)
        elif self._basis_fun == 'binary_basis':
            self._x_basis = binary_basis(self._x_org)
        elif self._basis_fun == 'pyramid_basis':
            self._x_basis = pyramid_basis(self._x_org)
        elif self._basis_fun == 'radial_basis':
            self._x_basis = radial_basis_fun(self._x_org)
        elif self._basis_fun == 'binned_basis':
            self._x_basis = binned_basis_fun(self._x_org)
        else:
            raise Exception("Unknown basis: {}".format(self._basis_fun))

        print "\nData set added"
        print "Name: {}".format(self._data_set)
        print "Basis: {}".format(self._basis_fun)
        print "N_samples: {}".format(self._y.shape[0])
        firing_rate = self._y.mean() * 1e3 / self._dt
        print "Mean firing rate: {} Hz".format(firing_rate)

    def check_model_gradients(self):
        """Performs gradient checking on all models that uses gradients

        Args:

        Returns:

        Raises:
            Exception if data has not been created first
        """

        if len(self._x_basis) == 0:
            raise Exception("No data added yet!!!")

        x_train = self._x_basis[:, :, :]
        y_train = self._y[self._rf_win_size - 1:]

        # Assume we want to check gradients on the first data set
        rf_models.gradient_checking(x_train, y_train, self._rf_win_size)

    def check_cf_gradients(self):
        """ Compares the results of both CF gradient functions

                Args:

                Returns:

                Raises:
                    Exception if data has not been created first
                """

        if len(self._x_train) == 0:
            raise Exception("No data created yet!!!")

        # Assume we want to check gradients on the first data set
        rf_models.cf_der_checking(self._x_train[0],
                                  self._y_train[0],
                                  self._rf_win_size)

    def check_non_convexity(self):
        """ Rudimentary check for non-convexity by plotting the objective
            function for lines in the parameter space

        Args:

        Returns:

        Raises:
            Exception if data has not been created first
        """

        if len(self._x_train) == 0:
            raise Exception("No data created yet!!!")

        # Assume we want to check for non-convexity on the first data set
        rf_models.lines_in_parameter_space(self._x_train[0],
                                           self._y_train[0],
                                           self._rf_win_size)

    def train_models(self,
                     cf_shape = [],
                     n_splits=5,
                     first_fold_only=True,
                     load_path=None,
                     verbose=1):
        """ Trains all rf models

        Training is either done once using a given train/test fraction or n
        times where the train/test fraction is obtained as 1 - 1/n_folds

        Args:
            train_frac: train/test set fraction
            n_splits:
        Returns:

        Raises:
            Exception if data has not been added first
        """

        if self._x_basis.size == 0:
            raise Exception("No data added yet!!!")

        train_frac = 1 - 1. / n_splits if n_splits > 0 else 1
        print "\nTraining models, training frac.: {}".format(train_frac)

        for model in self._models:

            # Set meta data
            model.set_meta_data(self._data_set,
                                self._basis_fun,
                                self._x_labes,
                                self._x_ticks,
                                self._dt)

            # Initialize model parameters
            win_size = self._rf_win_size
            rf_shape = [self._rf_win_size] + list(self._x_basis.shape[1:])
            if len(cf_shape) == 0:
                cf_shape = copy.deepcopy(rf_shape)

            # Use KFold for a separation into training and test sets
            if n_splits > 0:

                kf = KFold(n_splits=n_splits)
                split_idx = 0
                n_samples = self._x_basis.shape[0]

                for train_idx, test_idx in kf.split(range(n_samples)):
                    # Initialize split parameters
                    model.initialize_params(rf_shape, cf_shape, load_path)

                    # Divide up into training and test set
                    x_train = self._x_basis[train_idx, :, :]
                    x_test = self._x_basis[test_idx, :, :]
                    # Throw away the first rf_win_size -1 elements from y as
                    # these lack a full time window of inputs.
                    y_train = self._y[train_idx[win_size - 1:], :]
                    y_test = self._y[test_idx[win_size - 1:], :]

                    # Train and evaluate the model
                    model.train(x_train, y_train, win_size, split_idx, verbose)
                    model.evaluate(x_train, y_train, 'train', split_idx)
                    model.evaluate(x_test, y_test, 'test', split_idx)

                    # Print progress
                    # if verbose:
                    print "Fold {} of {} done.".format(split_idx + 1, n_splits)

                    # Break the if we only want to evaluate the first fold
                    if first_fold_only:
                        break
                    else:
                        split_idx += 1
            # Otherwise we use all available data for training
            else:
                model.initialize_params(rf_shape, cf_shape, load_path)

                # Divide up into training and test set
                x_train = self._x_basis[:, :, :]
                # Throw away the first rf_win_size -1 elements from y as these
                # lack a full time window of inputs.
                y_train = self._y[win_size - 1:, :]

                # Train and evaluate the model
                model.train(x_train, y_train, win_size, 0, verbose)
                model.evaluate(x_train, y_train, 'train', 0)

            print "{} trained.".format(model.__str__())

        self._rf_models_trained = True

    def predict(self, split_idx=0):
        """

        :param split_idx:
        :return:
        """

        y_hats = []
        for model in self._models:
            y_hats.append(model.predict(self._x_basis, split_idx))

        return y_hats

    def reevaluate_models(self, n_splits=5, first_fold_only=True):

        for model in self._models:

            model.eval_train = []
            model.eval_test = []

            # Use KFold for a separation into training and test sets
            kf = KFold(n_splits=n_splits)
            split_idx = 0
            for train_idx, test_idx in kf.split(range(self._x_basis.shape[0])):

                # Divide up into training and test set
                x_train = self._x_basis[train_idx, :, :]
                x_test = self._x_basis[test_idx, :, :]
                # Throw away the first rf_win_size -1 elements from y as these
                # lack a full time window of inputs.
                y_train = self._y[train_idx[self._rf_win_size - 1:]]
                y_test = self._y[test_idx[self._rf_win_size - 1:]]

                # Train and evaluate the model
                model.evaluate(x_train, y_train, 'train', split_idx)
                model.evaluate(x_test, y_test, 'test', split_idx)

                # Break the if we only want to evaluate the first fold
                if first_fold_only:
                    break
                else:
                    split_idx += 1

    def compare_predictions(self):

        if self._x_basis.size == 0:
            raise Exception("No data added yet!!!")

        y_hats = []
        for model in self._models:
            if model.trained[0]:
                y_hats.append(model.predict(self._x_basis))

        import matplotlib.pyplot as plt
        colors = ['r', 'b']
        for idx in range(len(y_hats)):
            plt.plot(y_hats[idx], colors[idx], marker='.', ls='-')
        y = self._y.copy()
        y[y > 0] = 1
        plt.plot(y[self._rf_win_size - 1:], 'ko')
        plt.ylim([-0.1, 1.1])
        plt.show()

    def plot_models(self, fold=None, path=None):

        one_bar_tail = False
        save_dir = self._get_save_directory(path)

        for model in self._models:

            if fold is not None:
                params = [model.params[fold]]
            else:
                params = model.params

            # Count the total number of plots
            n_plots = 0
            n_rf_parts = len(params[0].rfs[0].parts) if \
                params[0].rfs[0].multilin else 1
            n_plots += len(params[0].rfs) * n_rf_parts
            if params[0].cfs:
                n_cf_parts = len(params[0].cfs[0].parts) if \
                    params[0].cfs[0].multilin else 1
                n_plots += len(params[0].cfs) * n_cf_parts
            n_plots += 1 if model.nonlinearity else 0
            n_plots += 1 if model.rf_significance else 0
            if hasattr(params[0], 'cf_act_funs'):
                n_plots += len(params[0].cf_act_funs)

            n_cols = n_plots
            n_rows = len(params)
            ax_id = 1

            fig_win_scaling = {'width': 2,
                               'height': 1 if n_rows == 1 else 2}
            fig = plot_fun.create_fig_window(fig_win_scaling)

            # Different folds on each row
            for fold_idx in range(len(params)):

                # Receptive fields
                for rf in params[fold_idx].rfs:
                    # if rf.multilin:
                    #     for part in rf.parts:
                    #         ax = fig.add_subplot(n_rows, n_cols, ax_id)
                    #         plot_fun.plot_field(ax, part)
                    #         ax_id += 1
                    # else:
                    rf_dims = len([i for i in rf.shape if i > 1])
                    rf_proj = None if rf_dims < 3  else '3d'
                    ax = fig.add_subplot(n_rows, n_cols, ax_id,
                                         projection=rf_proj)
                    plot_fun.plot_field(ax, rf.field,
                                        labels=model.labels,
                                        dt=model.dt)
                    ax.set_title('RF, (min: %2.2f, max: %2.2f, bias: %2.2f)' %
                                 (rf.field.min(), rf.field.max(), rf.bias))
                    ax_id += 1

                # CF activation function
                if hasattr(params[fold_idx], 'cf_act_funs'):
                    for act_fun in params[fold_idx].cf_act_funs:
                        ax = fig.add_subplot(n_rows, n_cols, ax_id)
                        plot_fun.plot_line(
                            ax, act_fun.base_peaks, act_fun.alpha, marker='o')
                        ax.set_title('CF activation function')
                        ax_id += 1

                # Context fields
                for cf in params[fold_idx].cfs:
                    cf_dims = len([i for i in cf.shape if i > 1])
                    cf_proj = None if cf_dims < 3  else '3d'
                    ax = fig.add_subplot(n_rows, n_cols, ax_id,
                                         projection=cf_proj)
                    plot_fun.plot_field(ax, cf.field,
                                        labels=model.labels,
                                        dt=model.dt)
                    ax.set_title('CF, (min: %2.2f, max: %2.2f, bias: %2.2f)' %
                                 (cf.field.min(), cf.field.max(), cf.bias))
                    ax_id += 1

                # Nonlinearity
                if model.nonlinearity:
                    ax = fig.add_subplot(n_rows, n_cols, ax_id)
                    plot_fun.plot_nonlinearity(
                        ax, model.nonlinearity[fold_idx])
                    ax_id += 1

                if model.rf_significance:
                    ax = fig.add_subplot(n_rows, n_cols, ax_id)
                    if hasattr(model, 'ln_significance'):
                        plot_fun.plot_ln_significance(
                            ax, model.ln_significance[fold_idx])
                    else:
                        plot_fun.plot_ln_significance(
                            ax, model.rf_significance[fold_idx])
                    ax_id += 1

            plot_fun.tight()

            if save_dir is not None:
                file_name = save_dir + model.name + '.png'
                plot_fun.save(fig, file_name)
                plot_fun.clear(fig)
                plot_fun.close(fig)

        if save_dir is None:
            plot_fun.show()

        return one_bar_tail

    def plot_evaluation(self, path=None):

        save_dir = self._get_save_directory(path)

        for model in self._models:

            for key in model.eval_train[0].mi_keys:

                # Count the total number of fields
                n_folds = len(model.eval_train)
                n_cols = 2 if n_folds == 1 else n_folds
                n_rows = 1 if n_folds == 1 else 2

                fig_win_scaling = {'width': 2,
                                   'height': 1 if n_rows == 1 else 1.5}

                # Training folds
                fig = plot_fun.create_fig_window(fig_win_scaling)

                ax_id = 1
                for eval in model.eval_train:
                    ax = fig.add_subplot(n_rows, n_cols, ax_id)
                    plot_fun.plot_mi_estimation(ax, eval, key)
                    ax_id += 1

                for eval in model.eval_test:
                    ax = fig.add_subplot(n_rows, n_cols, ax_id)
                    plot_fun.plot_mi_estimation(ax, eval, key)
                    ax_id += 1

                plot_fun.tight()
                if save_dir is not None:
                    file_name = save_dir + model.name + '_' + key + '_eval.png'
                    plot_fun.save(fig, file_name)
                    plot_fun.clear(fig)
                    plot_fun.close(fig)
                else:
                    plot_fun.show()

    def print_obj_fun_values(self):

        header = "\n{:25s}{:>15s}{:>7s}".format(
            'Name', 'value', 'std')
        print header

        for model in self._models:
            info = "{:25s}{:>15.2f}{:>7.2f}".format(model.name,
                                                    model.obj_fun_val.mean(),
                                                    model.obj_fun_val.std())
            print info

    def print_mi_values(self):

        header = "\n{:25s}{:15s}{:>15s}{:>7s}{:>15s}{:>7s}".format(
            'Name', 'method', 'mi_train', 'std', 'mi_test', 'std')
        print header

        for model in self._models:
            for key in model.eval_train[0].mi_keys:
                mi_values_train = [eval.mi[key] for eval in model.eval_train]
                mi_train_mu = np.array(mi_values_train).mean()
                mi_train_std = np.array(mi_values_train).std()
                mi_values_test = [eval.mi[key] for eval in model.eval_test]
                mi_test_mu = np.array(mi_values_test).mean()
                mi_test_std = np.array(mi_values_test).std()

                info = "{:25s}{:15s}{:>15.2f}{:>7.2f}{:>15.2f}{:>7.2f}".format(
                    model.name, key, mi_train_mu, mi_train_std,
                    mi_test_mu, mi_test_std)
                print info

    def print_r_values(self):

        header = "\n{:25s}{:>15s}{:>7s}{:>15s}{:>7s}".format(
            'Name', 'r_train', 'std', 'r_test', 'std')
        print header

        for model in self._models:
            r_values_train = [eval.r for eval in model.eval_train]
            r_train_mu = np.array(r_values_train).mean()
            r_train_std = np.array(r_values_train).std()
            r_values_test = [eval.r for eval in model.eval_test]
            r_test_mu = np.array(r_values_test).mean()
            r_test_std = np.array(r_values_test).std()

            info = "{:25s}{:>15.2f}{:>7.2f}{:>15.2f}{:>7.2f}".format(
                model.name, r_train_mu, r_train_std,
                r_test_mu, r_test_std)
            print info

        header = "\n{:25s}{:>15s}{:>7s}{:>15s}{:>7s}".format(
            'Name', 'r_train_norm', 'std', 'r_test_norm', 'std')
        print header

        for model in self._models:
            if model.eval_train[0].r_norm is not None:
                r_values_train = [eval.r_norm for eval in model.eval_train]
                r_train_mu = np.array(r_values_train).mean()
                r_train_std = np.array(r_values_train).std()
                r_values_test = [eval.r_norm for eval in model.eval_test]
                r_test_mu = np.array(r_values_test).mean()
                r_test_std = np.array(r_values_test).std()

                info = "{:25s}{:>15.2f}{:>7.2f}{:>15.2f}{:>7.2f}".format(
                    model.name, r_train_mu, r_train_std,
                    r_test_mu, r_test_std)
                print info

    def save_models(self, path):
        """ Save all models

        :param save_dir:
        :return:
        """

        save_dir = self._get_save_directory(path)

        if save_dir is not None:
            for model in self._models:
                file_name = save_dir + model.name + '.dat'
                pickle.dump(model, open(file_name, 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)

    def load_models(self, path, tag=None):
        """ Load models from given directory

        :param load_dir:
        :return:
        """

        models = load_saved_models(path, tag=tag)

        if len(models) > 0:
            # Take meta data from the first model
            self._rf_win_size = models[0].params[0].rfs[0].shape[0]
            self._basis_fun = models[0].basis_fun
            self._models = models
            self._rf_models_trained = True

    def _get_save_directory(self, path):
        """ Create a new subdirectory named after the data set

        :param save_dir:
        :return:
        """

        save_dir = None

        if path is not None and self._data_set is not None:
            save_dir = path + self._data_set + "/"
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)

        return save_dir
