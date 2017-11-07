#!/usr/bin/python
"""
" @section DESCRIPTION
" Generic 1- or 2-D nonlinearity for RF models
"""

import numpy as np
from rf_helper import z_dist
from rf_obj_funs_and_grads import inv_bernoulli_link, inv_poisson_link


class Nonlinearity(object):
    """Generic 1- or 2-D nonlinearity for LN or context models"""

    def __init__(self, dist_res=11):
        """ Create nonliearity object

            Args:
                dist_res: resolution along each dimension when
                          approximating probability distributions
            Raise

        """
        self.distribution_res = dist_res
        self.mean_z = None
        self.std_z = None
        self.z_edges = []
        self.firing_prob = None
        self.firing_prob_nonlin = None
        self.learned = False

    def estimate(self, z, y, solver=None):
        """ Estimating the probability distribution p(spike|z)

            This function estimates a non-linear function that maps similarity
            scores (z) into spiking probabilities as:
            P(spike|z) = P(spike) * P(z|spike) / P(z)

            Args:
                z: similarity score array
                y: spike count array
            Returns:

            Raise:

        """

        # Projection z and statistics
        self.mean_z = np.mean(z, axis=0)
        self.std_z = np.std(z, axis=0)

        # P(spike)
        p_spike = np.mean(y)

        # P(z) and P(z|spike)
        p_z, p_z_spike, self.z_edges = \
            z_dist(z, y, self.distribution_res)

        # P(spike|z) = p(z|spike) p(spike) / p(z)
        prob_frac = np.zeros(p_z.shape)
        no_zero = p_z > 0
        prob_frac[no_zero] = p_z_spike[no_zero] / p_z[no_zero]
        self.firing_prob = p_spike * prob_frac

        if solver is not None:
            dz = np.diff(self.z_edges[0][1:3])[0] / 2
            z = self.z_edges[0][1:] - dz
            z[-1] = z[-2] + 2 * dz
            if solver == 'linreg':
                self.firing_prob_nonlin = z
            elif solver == 'logreg':
                self.firing_prob_nonlin = inv_bernoulli_link(z)
            elif solver == 'poireg':
                self.firing_prob_nonlin = inv_poisson_link(z)
            elif solver == 'mne':
                self.firing_prob_nonlin = inv_bernoulli_link(z)

        self.learned = True

    def predict(self, z):
        """ Probability for one or more spikes as a function of z

            Args:
                z: similarity score array
            Returns:
                y_hat:
            Raise:
                Exception if not learned first

        """

        if not self.learned:
            raise Exception("Nonlinearity not learned yet!")

        # Anything outside the right edge should fall into the
        # right most bin.
        bin_idx = []
        for dim in range(len(self.z_edges)):
            bin_idx.append(
                np.digitize(z[:, dim].flatten(),
                            self.z_edges[dim][:-1]) - 1
            )
            # Anything outside the left edge should fall into the
            # left most bin.
            if np.min(bin_idx[dim]) < 0:
                bin_idx[dim][bin_idx[dim] < 0] = 0

        if len(self.z_edges) == 1:
            y_hat = self.firing_prob[bin_idx[0]]
        else:
            y_hat = self.firing_prob[bin_idx[0], bin_idx[1]]

        return y_hat
