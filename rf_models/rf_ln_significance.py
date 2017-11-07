#!/usr/bin/python
"""
" @section DESCRIPTION
" Filter significance tester for multi-filter LN models
"""

import numpy as np


class Significance(object):
    """Filter significance tester for LN models

    Originally intended for automatic test of the number of relevant
    filters in LN-models. However, all test tried seemed very arbitrary in that
    the number of filters can be heavily modified through arbitrary parameters.
    These have therefore been removed and this class now mainly
    stores the results from the solver used to find LN-model filters.

    """

    def __init__(self, solver='stc'):
        """ Create significance object

            Args:
                solver: 'stc', 'istac', or 'mne'
            Raise

        """
        self.values = None
        self.significant_idxs = None
        self.solver = solver

        if solver[:3] == 'stc' or solver == 'mne':
            self.unit_label = 'Eigen value'
        elif solver == 'istac':
            self.unit_label = 'Mutual information'
        else:
            raise Exception("Unknown solver: {}".format(solver))

    def add_values(self, values):
        """ Store filter specific values (eigen-values or MI)

        :param values:
        :return:
        """
        self.values = values.copy()

    def get_most_sig_idxs(self, n_dims):
        """ Returns indices for the n-most significant filters

        :param n_dims:
        :return:
        """

        if n_dims > 0:
            # Select indices corresponding to the n_dims eigne values that
            # deviate the most from the mean eigen value
            if self.solver[:3] == 'stc' or self.solver == 'mne':
                mu = self.values.mean()
                abs_diff = np.abs(self.values - mu)
                sort_idxs = np.argsort(abs_diff)
                most_sig_idxs = sort_idxs[-n_dims:]

            # Select indices with the top n_dims mutual information values
            elif self.solver == 'istac':
                sort_idxs = np.argsort(self.values)
                most_sig_idxs = sort_idxs[-n_dims:]
        else:
            most_sig_idxs = np.array([])

        # Revert the array so that the most significant
        # indices are first instead of last.
        most_sig_idxs = most_sig_idxs[::-1]
        most_sig_vals = self.values[most_sig_idxs]

        return most_sig_idxs, most_sig_vals