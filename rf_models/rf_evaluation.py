#!/usr/bin/python
"""
" @section DESCRIPTION
" Model evaluation, Pearson's r and mutual information
"""

import numpy as np
from rf_obj_funs_and_grads import mutual_information
from rf_helper import calculate_r


class Evaluation(object):
    """Estimator for determining mutual information between z and spikes (y)"""

    def __init__(self):
        """ Create field object

            Args:

            Raise

        """
        self.r = None
        self.r_norm = None
        self.mi_hist_resolutions = np.arange(6, 50, 1)
        self.mi = {}
        self.mi_values = {}
        self.mi_bias = {}
        self.mi_keys = []
        self.mi_nr_frac = None

    def evaluate_mi_stc(self, z, z_null, y):
        """
        Mutual information varies depending on the histogram resolution used
        due to an increased finite sample bias with increasing resolution. This
        function estimates the finite sample bias by calculating the mutual
        information value for null vectors, which should carry zero information.
        The obtained value for the null vectors are therefore an estimate of the
        finite sample bias.

        see Fairhall et al. (2006)

        :param z: similarity scores
        :param z_null: null similarity score
        :param y: spike counts
        :return:
        """

        # Estimated bias
        mi_values = np.zeros(self.mi_hist_resolutions.size)
        mi_bin_counts = np.zeros(self.mi_hist_resolutions.size)
        mi_values_null = np.zeros(self.mi_hist_resolutions.size)
        mi_bin_counts_null = np.zeros(self.mi_hist_resolutions.size)
        mi_values_cmp = np.zeros(self.mi_hist_resolutions.size)
        mi_bias = np.zeros(self.mi_hist_resolutions.size)
        for z_idx in range(min(z.shape[1], 2)):

            for hist_res in self.mi_hist_resolutions:

                mi_tmp, mi_bin_count_tmp = \
                    mutual_information(z[:, 0:z_idx+1], y, hist_res)

                mi_null_tmp = 0
                mi_null_bin_count_tmp = 0
                for null_idx in range(z_null.shape[1]):

                    if z_idx == 0:
                        z_null_tmp = \
                            z_null[:, null_idx].reshape(z_null.shape[0], 1)
                    else:
                        z_null_tmp = np.vstack(
                            [z[:, 0].copy(), z_null[:, null_idx].flatten()]).T

                    mi_null_info = mutual_information(z_null_tmp, y, hist_res)
                    mi_null_tmp += mi_null_info[0]
                    mi_null_bin_count_tmp += mi_null_info[1]

                mi_null_tmp /= z_null.shape[1]
                mi_null_bin_count_tmp /= z_null.shape[1]

                res_idx = np.where(self.mi_hist_resolutions == hist_res)[0][0]
                mi_values[res_idx] = mi_tmp
                mi_bin_counts[res_idx] = mi_bin_count_tmp
                mi_values_null[res_idx] = mi_null_tmp
                mi_bin_counts_null[res_idx] = mi_null_bin_count_tmp

            # Interpolate bias values to account for differences in bin counts
            mi_values_null_interp = np.interp(mi_bin_counts, mi_bin_counts_null,
                                              mi_values_null)

            mi_bias[:] = mi_values_null_interp - mi_values_cmp
            mi_values_cmp[:] = mi_values - mi_bias

        mi_values_true = mi_values - mi_bias

        # True value as an average over resulutions with 25 to 35 bins per dim.
        mask = (25 <= self.mi_hist_resolutions) & \
               (self.mi_hist_resolutions <= 35)
        mi_value = mi_values_true[mask].mean()

        # Estimate  N / R at 30 bin resolution
        nr_frac = y.sum() / mi_bin_counts[self.mi_hist_resolutions == 30]

        self.mi_nr_frac = nr_frac
        self.mi['stc'] = mi_value
        self.mi_values['stc'] = mi_values
        self.mi_bias['stc'] = mi_bias
        self.mi_keys.append('stc')

    def evaluate_mi_qe(self, z, y):
        """
        Mutual information varies depending on the histogram resolution used
        due to an increased finite sample bias with increasing resolution. This
        function fits a quadratic function on the form:
        I_naive = I_true + a/N + b/N^2
        and hence estimates the true I value as the height at which the
        function crosses the y-axis. Essentially extra-polating to an infinite
        sample size.

        see Treves and Panzeri (1995) and Panzeri et al. (2007)

        :param z: similarity scores
        :param y: spike counts
        :return:
        """
        n_reps = 50
        n_fracs = 10

        mi_values = np.zeros(self.mi_hist_resolutions.size)
        mi_bias = np.zeros(self.mi_hist_resolutions.size)
        mi_bin_counts = np.zeros(self.mi_hist_resolutions.size)

        for hist_res in self.mi_hist_resolutions:

            set_fracs = 1. / np.linspace(1, 4, n_fracs)
            mi_values_tmp = np.zeros([n_fracs, n_reps])
            for frac_id in range(n_fracs):
                frac_size = np.int64(set_fracs[frac_id]*y.size)
                for rep_id in range(n_reps):
                    r_perm = np.random.permutation(y.size)
                    idx_tmp = r_perm[:frac_size]
                    mi_tmp, _ = mutual_information(z[idx_tmp],
                                                   y[idx_tmp],
                                                   hist_res)
                    mi_values_tmp[frac_id, rep_id] = mi_tmp

            # a = np.vstack([np.ones(n_fracs), 1 / set_fracs, (1 / set_fracs)**2])
            a = np.vstack([np.ones(n_fracs), 1 / set_fracs])
            b = mi_values_tmp.mean(axis=1)
            coef, _, _, _ = np.linalg.lstsq(a.T, b)

            # import matplotlib.pyplot as plt
            # plt.plot(1 / set_fracs, mi_values_tmp.mean(axis=1), 'ko')
            # x_lin = np.linspace(0, 4, 100)
            # # plt.plot(x_lin, coef[0] + x_lin * coef[1] + x_lin**2 * coef[2], 'b')
            # plt.plot(x_lin, coef[0] + x_lin * coef[1], 'b')
            # plt.show()

            mi_tmp, mi_bin_count_tmp = mutual_information(z, y, hist_res)
            res_idx = np.where(self.mi_hist_resolutions == hist_res)[0][0]
            mi_values[res_idx] = mi_tmp
            mi_bias[res_idx] = mi_tmp - coef[0]
            mi_bin_counts[res_idx] = mi_bin_count_tmp

        # True value as an average over resulutions with 25 to 35 bins per dim.
        mi_values_true = mi_values - mi_bias
        mask = (25 <= self.mi_hist_resolutions) & \
               (self.mi_hist_resolutions <= 35)
        mi_value = mi_values_true[mask].mean()

        # Estimate  N / R at 30 bin resolution
        nr_frac = y.sum() / mi_bin_counts[self.mi_hist_resolutions == 30]

        self.mi_nr_frac = nr_frac
        self.mi['qe'] = mi_value
        self.mi_values['qe'] = mi_values
        self.mi_bias['qe'] = mi_bias
        self.mi_keys.append('qe')

    def evaluate_mi_raw(self, z, y):
        """
        Mutual information varies depending on the histogram resolution used
        due to an increased finite sample bias with increasing resolution. This
        function estimates ignores the bias and provides raw estimates.

        see Fairhall et al. (2006)

        :param z: similarity scores
        :param y: spike counts
        :return:
        """

        mi_values = np.zeros(self.mi_hist_resolutions.size)
        mi_bias = np.zeros(self.mi_hist_resolutions.size)
        mi_bin_counts = np.zeros(self.mi_hist_resolutions.size)

        for hist_res in self.mi_hist_resolutions:
            mi_tmp, mi_bin_count_tmp = mutual_information(z, y, hist_res)
            res_idx = np.where(self.mi_hist_resolutions == hist_res)[0][0]
            mi_values[res_idx] = mi_tmp
            mi_bin_counts[res_idx] = mi_bin_count_tmp

        # True value as an average over resulutions with 25 to 35 bins per dim.
        mi_values_true = mi_values - mi_bias
        mask = (25 <= self.mi_hist_resolutions) & \
               (self.mi_hist_resolutions <= 35)
        mi_value = mi_values_true[mask].mean()

        # Estimate  N / R at 30 bin resolution
        nr_frac = y.sum() / mi_bin_counts[self.mi_hist_resolutions == 30]

        self.mi_nr_frac = nr_frac
        self.mi['raw'] = mi_value
        self.mi_values['raw'] = mi_values
        self.mi_bias['raw'] = mi_bias
        self.mi_keys.append('raw')

    def evaluate_r(self, y_hat, y):
        """ Calculates Pearson's r and its normalized version

        See Schoppe et al. (2016)

        :param y_hat: Predicted responses
        :param y: Recorded responses, each column represent one trial
        :return:
        """

        # Pearson correlation
        y_mean = np.vstack(y.mean(axis=1))
        y_hat_var = y_hat.var()
        y_var = y_mean.var()
        cov = np.dot(y_mean.ravel()-y_mean.mean(), y_hat.ravel()-y_hat.mean())
        cov /= (y_hat.size - 1)

        r_val = cov / np.sqrt(y_hat_var*y_var)
        # r_val = calculate_r(y_mean.ravel(), y_hat.ravel())
        self.r = r_val

        if y.shape[1] > 1:
            sp = (y.sum(axis=1).var() - y.var(axis=0).sum()) / \
                 (y.shape[1] ** 2 - y.shape[1])
            r_val_norm = cov / np.sqrt(y_hat_var * sp)
            self.r_norm = r_val_norm

