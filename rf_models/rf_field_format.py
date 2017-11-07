#!/usr/bin/python
"""
" @section DESCRIPTION
" Field class for representing data in a field format
"""

import numpy as np
from operator import mul
from rf_helper import outer_product, gaussian_field, smooth_reg_l
from scipy import sparse



class Field(object):
    """Field class representing data in field format"""

    def __init__(self,
                 shape,
                 reg_c,
                 multilin=False,
                 init_values='zeros',
                 origin=[],
                 zero_origin=False,
                 reg_type='norm'):

        """ Create field object

        :param shape: list with field dimensions
        :param multilin: multi-linear model (True) / full matrix (False)
        :param init_values: initial values: 'zeros' | 'ones' | 'randn'
        :param origin: field origin when doing conv|corr
        :param zero_origin: keep the origin at zero
        :param reg_type: 'norm' | 'smooth'
        """
        self.field = np.array([])  # Field matrix
        self.parts = []  # Field parts, if multilinear
        self.bias = 0
        self.reg_l = np.array([])  # regularization matrix
        self.shape = shape
        self.multilin = multilin
        self.init_values = init_values
        self.zero_origin = zero_origin
        self.reg_type = reg_type
        self.reg_c = reg_c
        self.reg_c_from_cv = False
        self.qn_lambda = 1  # used with low-rank QN models only
        self.qn_square = False  # used with low-rank QN models only

        # Default origin if not provided (only used by CFs)
        self.origin = origin
        if len(self.origin) == 0:
            self.origin = [0 for i in range(len(self.shape))]

        self._initialize_field()
        self.initialize_reg_l()

    def make_separable(self, reg_c, reg_type):
        """ TESTING PHASE STILL

        :param reg_c:
        :param reg_type:
        :return:
        """
        u, s, v = np.linalg.svd(self.field[:, :, 0])
        self.parts.append(u[:, 0])
        self.parts.append(v[0, :].reshape([v.shape[0], 1]))
        self.field = outer_product(self.parts)
        self.multilin = True
        self.initialize_reg_l()
        self.reg_c = reg_c
        self.reg_type = reg_type
        self.reg_c_from_cv = False

    def get_unraveled_origin_idx(self):
        """ Returns the unraveled index of the field's origin """

        origin_pos = [self.shape[0] / 2 + self.origin[0],
                      self.shape[1] / 2 + self.origin[1],
                      self.shape[2] / 2 + self.origin[2]]
        origin_idx = np.ravel_multi_index(origin_pos, self.shape)

        return origin_idx

    def get_raveled_origin_pos(self):
        """ Returns the raveled origin position """

        origin_pos = [self.shape[0] / 2 + self.origin[0],
                      self.shape[1] / 2 + self.origin[1],
                      self.shape[2] / 2 + self.origin[2]]

        return origin_pos

    def initialize_reg_l(self):
        """Determine the Tikhonov regularization matrix (Lambda)

        Creates an ideintity matrix or the discrete Laplace operator for
        "norm" and "smooth" regularization respectively.

        Observe that it only includes field parameters (not the bias term)

        """

        # For functionality with older saved models!!!
        if hasattr(self, 'regularization'):
            self.reg_type = self.regularization

        n_params = reduce(mul, self.shape)

        # Standard L2 norm regularization
        if self.reg_type == 'norm':
            if self.multilin:
                reg_l = [np.eye(part.size) for part in self.parts]
            else:
                reg_l = np.eye(n_params)

        # Smooth regularization
        elif self.reg_type == 'smooth':
            if self.multilin:
                reg_l = []
                for part in self.parts:
                    reg_l.append(smooth_reg_l(part.shape))
            else:
                reg_l = smooth_reg_l(self.shape)
                # Remove connections to the origin
                if self.zero_origin:
                    origin_idx = self.get_unraveled_origin_idx()
                    reg_l[origin_idx, :] = 0
                    reg_l[:, origin_idx] = 0
                    reg_l[origin_idx, origin_idx] = 1

        else:
            if self.multilin:
                reg_l = []
                for part in self.parts:
                    reg_l.append(np.array([]))
            else:
                reg_l = np.array([])

        # Make the matrix sparse
        if self.multilin:
            self.reg_l = [sparse.csr_matrix(l) for l in reg_l]
        else:
            self.reg_l = sparse.csr_matrix(reg_l)

    def _initialize_field(self):
        """ Initializes field values"""

        # Scaling of initial values
        beta = 1e-2

        # Multi-linear field structure
        field_size = reduce(mul, self.shape)
        if self.multilin:
            if self.init_values == 'zeros':
                self.parts = [np.zeros(self.shape[0])]
                # self.parts = self.parts + \
                #              [np.ones(n) for n in self.shape[1:]]
                self.parts = self.parts + [np.ones(self.shape[1:])]
                self.bias = 0
            elif self.init_values == 'ones':
                self.parts = [np.ones(self.shape[0]) * beta]
                # self.parts = self.parts + \
                #              [np.ones(n) for n in self.shape[1:]]
                self.parts = self.parts + [np.ones(self.shape[1:])]
                self.bias = 1 * beta
            elif self.init_values == 'randn':
                self.parts = [np.random.randn(self.shape[0]) * beta]
                # self.parts = self.parts + \
                #              [np.random.randn(n) for n in self.shape[1:]]
                self.parts = self.parts + [np.random.randn(self.shape[1],
                                                           self.shape[2])]
                self.bias = np.random.randn(1)[0] * beta
            self.field = outer_product(self.parts)

        # Full field matrix structure
        else:
            self.parts = []
            if self.init_values == 'zeros':
                self.field = np.zeros(field_size).reshape(self.shape)
                self.bias = 0
            elif self.init_values == 'ones':
                self.field = np.ones(field_size).reshape(self.shape) * beta
                self.bias = 1 * beta
            elif self.init_values == 'randn':
                self.field = np.random.randn(field_size).reshape(self.shape) \
                             * beta
                self.bias = np.random.randn(1)[0] * beta
            elif self.init_values == 'gaussian':
                self.field = gaussian_field(self.shape, self.origin) * beta
                self.bias = 1 * beta
