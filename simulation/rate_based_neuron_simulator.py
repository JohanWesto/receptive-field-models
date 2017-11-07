#!/usr/bin/python
"""
" @section DESCRIPTION
" Rate based model simulator which can generate data representative of:
"   - traditional LN models,
"   - multi-filter LN models,
"   - context models,
"   - position invariant RF models (shifted filters).
"""

import numpy as np
from scipy.stats import multivariate_normal
from rf_models.rf_parameters import Parameters
from rf_models.rf_helper import inner_product
from rf_models.rf_obj_funs_and_grads import z_rf_der
import plotting.plotting_functions as plot_fun


class RateBasedModel:
    """Class for creating and simulating rate based neural networks"""

    def __init__(self):
        """ Initializes the rate based model """

        self._type = None
        self._cf_act_fun = None
        self._params = None
        self._x = np.array([])
        self._y = np.array([])
        self._built = False
        self._simulated = False

    def build_model(self, stimuli, sim_params):
        """ Builds the rate based model network

        Network types:
        -'gabor'
        -'energy'
        -'shifted'
        -'context'

        Args:
            stimuli:
            sim_params: simulation parameters
        Returns:

        Raises:

        """

        net_params = sim_params['network_model']
        stimulus_dims = stimuli['values'].shape[1:]

        self._type = net_params['type']
        self._cf_act_fun = net_params['cf_act_fun']

        self._params = _generate_filters(stimulus_dims, type=self._type)
        self._x = stimuli['values']
        self._built = True

    def simulate(self):
        """ Simulate the model

        :return:
        """

        if not self._built:
            raise Exception("Build the model first!!!")

        rf_win_size = self._params.rfs[0].shape[0]

        x_nd_full = z_rf_der(self._x, self._params)
        z = inner_product(x_nd_full, self._params.rfs)
        z = z - z.mean(axis=0)
        if self._cf_act_fun == 'rectify':
            z[z < 0] = 0
        elif self._cf_act_fun == 'square':
            z *= z

        # Sum contributions from all subunits and shift the distribution
        # z_sum = z.sum(axis=1)
        z_sum = z.max(axis=1)
        z_sum -= z_sum.mean()
        z_sum -= 1.5*z_sum.std()
        # Apply a sigmoid nonlinearity and generate spikes probabilistically
        y = np.zeros([self._x.shape[0], 1], dtype=np.int64)
        spike_prob = 1 / (1 + np.exp(-10*z_sum))
        spike = np.random.rand(spike_prob.size) < spike_prob
        y[rf_win_size - 1:, 0] = spike

        self._y = y
        self._simulated = True

    def get_spike_counts(self,):
        """ Return spike counts
        :return y:
        """

        if not self._simulated:
            raise Exception("Simulate the model first!!!")

        return self._y.copy()

    def view_filters(self):
        """ Plot model filters """

        n_cols = len(self._params.rfs) + len(self._params.cfs)
        fig_win_scaling = {'width': 1 if n_cols == 1 else 2,
                           'height': 1}
        fig = plot_fun.create_fig_window(fig_win_scaling)

        ax_id = 1
        for rf in self._params.rfs:
            ax = fig.add_subplot(1, n_cols, ax_id)
            plot_fun.plot_field(ax, rf.field)
            ax.set_title('RF')
            ax_id += 1

        for cf in self._params.cfs:
            ax = fig.add_subplot(1, n_cols, ax_id)
            plot_fun.plot_field(ax, cf.field)
            ax.set_title('CF')
            ax_id += 1

        plot_fun.tight()
        plot_fun.show()

    def save_filters(self):

        fig = plot_fun.create_fig_window(size=[4, 4])
        position = [0.0, 0.0, 1.0, 1.0]
        ax = fig.add_axes(position)

        id = 0
        for rf in self._params.rfs:
            plot_fun.plot_field(ax, rf.field, aspect=1)
            ax.axis('off')
            ax.set_xticks([])
            ax.set_yticks([])
            plot_fun.save(fig, 'rf%1d.eps' % id)
            id += 1

        id = 0
        for cf in self._params.cfs:
            plot_fun.plot_field(ax, cf.field, aspect=1)
            ax.axis('off')
            plot_fun.save(fig, 'cf%1d.eps' % id)
            id += 1


def _generate_filters(stimulus_dims, type='gabor'):
    """ Generate filters of varying types

    :param field_shape:
    :param type: gabor | energy | shifted | context
    :return:
    """

    params = Parameters()
    if stimulus_dims[1] == 1:
        res = stimulus_dims[0]
        field_shape = [res, res, 1]
    else:
        res = min(stimulus_dims)
        field_shape = [1] + list(stimulus_dims)

    rfs_tmp = []
    cfs_tmp = []
    if type == 'gabor':
        params.init_rfs(field_shape, reg_c=None)
        rfs_tmp.append(_get_gabor_filter(res, offset=45, scale=3))

    elif type == 'energy':
        params.init_rfs(field_shape, reg_c=None, n_rfs=2)
        rfs_tmp.append(_get_gabor_filter(
            res, offset=140, n_periods=1.75, scale=3))
        rfs_tmp.append(_get_gabor_filter(
            res, offset=50, n_periods=1.75, scale=3))

    elif type == 'shifted':
        params.init_rfs(field_shape, reg_c=None, n_rfs=6)

        for shift in range(-3, 3, 1):
            rfs_tmp.append(_get_gabor_filter(
                res, dir=0, offset=-120, n_periods=2.5, scale=4.0))
            rfs_tmp[-1] = np.roll(rfs_tmp[-1], shift, axis=1)

    elif type == 'context':
        params.init_rfs(field_shape, reg_c=None)
        rfs_tmp.append(_get_gabor_filter(
            res, dir=0, offset=0, n_periods=0.25, scale=4.0))

        params.init_cfs(field_shape, 'same', 'ctx', None, alignment='center')
        cfs_tmp.append(_get_gabor_filter(
            res, dir=0, offset=180, n_periods=5, scale=5.0))
        cfs_tmp[-1][res / 2, res / 2] = 0

    else:
        raise Exception("Unknown LN-model type: {]".format(type))

    for rf, rf_tmp in zip(params.rfs, rfs_tmp):
        if stimulus_dims[1] == 1:
            rf.field[:res, :res, 0] = rf_tmp
        else:
            rf.field[0, :res, :res] = rf_tmp

    for cf, cf_tmp in zip(params.cfs, cfs_tmp):
        if stimulus_dims[1] == 1:
            cf.field[:res, :res, 0] = cf_tmp
        else:
            cf.field[0, :res, :res] = cf_tmp

    return params


def _get_gabor_filter(res, dir=1, offset=0, n_periods=1, scale=1.5):
    """ Generates Gabour filters with selected tilt and offset

    :param res: resolution in pixels
    :param dir: direction (1 or -1)
    :param offset: offset in degrees
    :return gabor: Gabor filter
    """

    sigma = res / scale
    offset_scaled = offset / 360. * (res-1) / n_periods

    x, y = np.meshgrid(np.arange(res), np.arange(res))

    grid = np.vstack((x.ravel(), y.ravel())).T
    mean = [res / 2, res / 2]
    cov = [[sigma, 0.], [0., sigma]]
    gauss = multivariate_normal.pdf(grid, mean=mean, cov=cov)
    gauss = gauss.reshape(res, res)

    ripple = np.empty([res, res])

    scaling = 2 * np.pi / (res-1)*n_periods
    for row in range(res):
        for col in range(res):
            ripple[row, col] = \
                np.cos((col + dir * row + offset_scaled) * scaling)

    gabor = ripple * gauss
    gabor /= np.linalg.norm(gabor)

    return gabor
