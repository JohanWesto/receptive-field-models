#!/usr/bin/python
"""
" @section DESCRIPTION
" Class for describing an activation function that links CFs to RFs
"""

import numpy as np
from scipy import sparse
from rf_helper import smooth_reg_l


class ActFun(object):
    """Piecewise linear activation function"""

    def __init__(self, reg_c, init_shape='linear'):
        """ Create parameter object

        """
        self.init_shape = init_shape
        self.n_tents = None
        self.base_peaks = np.array([])
        self.alpha = np.array([])
        self.half_width = None
        self.slope = None
        self.limits_set = False
        self.reg_c = reg_c
        self.reg_c = 1e-3
        self.reg_c_from_cv = False

        # Set the functions initial shape
        self._set_init_shape(init_shape)

        # Smooth regularization using the discrete Laplace operator
        reg_l = smooth_reg_l([self.n_tents])
        reg_l = sparse.csr_matrix(reg_l)
        self.reg_l = reg_l

    def act_fun(self, z_ctx):
        """ Evaluates the nonlinearity

        :param z_ctx: similarity score between the CF and the stimulus context
        :return f_ctx: f(z_ctx)
        """

        if not self.limits_set:
            self.set_limits(z_ctx)

        f_ctx = np.zeros_like(z_ctx)
        for i in range(self.n_tents):
            # Upward slope
            if i > 0:
                if i == 1:
                    mask_up = (z_ctx <= self.base_peaks[i])
                elif i == self.n_tents - 1:
                    mask_up = (z_ctx > self.base_peaks[i] - self.half_width)
                else:
                    mask_up = (z_ctx > self.base_peaks[i] - self.half_width) & \
                              (z_ctx <= self.base_peaks[i])
                f_ctx[mask_up] += \
                    self.alpha[i] * self.slope * \
                    (z_ctx[mask_up] - (self.base_peaks[i] - self.half_width))
            # Downward slope
            if i < self.n_tents - 1:
                if i == 0:
                    mask_down = (z_ctx <= self.base_peaks[i] + self.half_width)
                elif i == self.n_tents - 2:
                    mask_down = (z_ctx > self.base_peaks[i])
                else:
                    mask_down = (z_ctx > self.base_peaks[i]) & \
                                (z_ctx <= self.base_peaks[i] + self.half_width)
                f_ctx[mask_down] += \
                    self.alpha[i] * self.slope * \
                    ((self.base_peaks[i] + self.half_width) - z_ctx[mask_down])

        return f_ctx

    def single_tent_fun(self, z_ctx, t_id):
        """ Evaluates a single tent function

        :param z_ctx: similarity score between the CF and the stimulus context
        :param t_id: id of the tent function to evaluate
        :return t_ctx: T(z_ctx)
        """

        assert t_id < self.n_tents

        if not self.limits_set:
            self.set_limits(z_ctx)

        t_ctx = np.zeros_like(z_ctx)
        # Upward slope
        if t_id > 0:
            if t_id == 1:
                mask_up = (z_ctx <= self.base_peaks[t_id])
            elif t_id == self.n_tents - 1:
                mask_up = (z_ctx > self.base_peaks[t_id] - self.half_width)
            else:
                mask_up = (z_ctx <= self.base_peaks[t_id]) & \
                          (z_ctx > self.base_peaks[t_id] - self.half_width)
            t_ctx[mask_up] += self.slope * \
                (z_ctx[mask_up] - (self.base_peaks[t_id] - self.half_width))
        # Downward slope
        if t_id < self.n_tents - 1:
            if t_id == 0:
                mask_down = (z_ctx <= self.base_peaks[t_id] + self.half_width)
            elif t_id == self.n_tents - 2:
                mask_down = (z_ctx > self.base_peaks[t_id])
            else:
                mask_down = (z_ctx > self.base_peaks[t_id]) & \
                            (z_ctx <= self.base_peaks[t_id] + self.half_width)
            t_ctx[mask_down] += self.slope * \
                ((self.base_peaks[t_id] + self.half_width) - z_ctx[mask_down])

        return t_ctx

    def act_fun_der(self, z_ctx):
        """

        :param z_ctx: similarity score between the CF and the stimulus context
        :return f_ctx_der: f'(z_ctx)
        """

        if not self.limits_set:
            self.set_limits(z_ctx)

        f_ctx_der = np.zeros_like(z_ctx)
        diff = np.diff(self.alpha) / self.half_width
        for i in range(diff.size):
            mask_tmp = (z_ctx > self.base_peaks[i]) & \
                       (z_ctx <= self.base_peaks[i + 1])
            f_ctx_der[mask_tmp] = diff[i]

        f_ctx_der[z_ctx < self.base_peaks[0]] = diff[0]
        f_ctx_der[z_ctx > self.base_peaks[-1]] = diff[-1]

        return f_ctx_der

    def set_limits(self, z_ctx):
        """ Fix limits for the piecewise linear function

        :param z_ctx:
        :return:
        """
        min_val = z_ctx.min()
        max_val = z_ctx.max()

        self.base_peaks = np.linspace(min_val, max_val, self.n_tents)
        self.half_width = self.base_peaks[1] - self.base_peaks[0]
        self.slope = 1. / self.half_width

        if self.init_shape == 'rectified':
            self.alpha[-1] = max_val

        self.limits_set = True

    def _set_init_shape(self, init_shape):

        if init_shape == 'linear':
            self.n_tents = 15
            self.alpha = np.linspace(-1, 1, self.n_tents)
        elif init_shape == 'random':
            self.n_tents = 15
            self.alpha = np.random.randn(self.n_tents)
        elif init_shape == 'rectified':
            self.n_tents = 3
            self.alpha = np.array([0., 0., 1.])
        elif init_shape == 'adaptive':
            self.n_tents = 15
            self.alpha = np.zeros(self.n_tents)