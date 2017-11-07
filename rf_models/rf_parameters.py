#!/usr/bin/python
"""
" @section DESCRIPTION
" Class for collecting all parameters contained in an RF model
"""

import copy
import numpy as np
from scipy import sparse
from rf_field_format import Field
from rf_piecewise_lin_act_fun import ActFun
from rf_helper import gaussian_field, smooth_reg_l


class Parameters(object):
    """Parameter class representing all field parameters STRF models"""

    def __init__(self):
        """ Create parameter object

        """
        self.rfs = []  # receptive fields
        self.cfs = []  # context fields
        self.cf_act_funs = []
        self.rf_type = None  # 'lin_rfs' | 'max_rfs' | 'qn_rfs' | 'pos_inv_rfs'
        self.cf_type = None  # 'ctx' | 'ctx_nl' | 'subunit'
        self.context_map = np.array([])
        self.mapping_method = None
        self.qn_poly = np.array([])
        self.solution_checked = False

    def init_rfs(self, shape, reg_c,
                 type='lin_rfs',
                 n_rfs=1,
                 multilin=False,
                 reg_type='norm',
                 init_values='zeros'):
        """ Initialize receptive field parameters

        :param shape: rf shape [time, freq, level]
        :param n_rfs: number of receptive fields
        :param multilin: multilinear or full matrix True/False
        :param reg_type: 'norm' | 'smooth'
        :param init_values: initial values 'zeros'/'ones'/'randn'
        :return:
        """

        self.rfs = [Field(shape, reg_c,
                          multilin=multilin,
                          reg_type=reg_type,
                          init_values=init_values) for i in range(n_rfs)]
        self.rf_type = type

    def init_cfs(self, shape, method, type, reg_c,
                 multilin=False,
                 reg_type='norm',
                 init_values='zeros',
                 zero_origin=False,
                 alignment='edge'):
        """ Initializes context field parameters and the rf-->cf mapping

        :param shape: cf shape [time, freq, level]
        :param method: mapping method 'none'/'same'/'sign'/
        :param type: # 'ctx' | 'ctx_nl' | 'subunit'
        :param multilin: multilinear or full matrix True/False
        :param init_values: initial values 'zeros'/'ones'/'randn'
        :return:
        """

        assert len(self.rfs) > 0

        # Set CF type
        self.cf_type = type

        # Define a mapping between RF elements and the CFs
        rf_shape = self.rfs[0].shape
        if method is None:
            n_cfs = 0
            context_map = -1 * np.ones(rf_shape)
        elif method == 'same':
            n_cfs = 1
            context_map = np.zeros(rf_shape)
        elif method == 'dim0_split':
            n_cfs = 1
            context_map = np.zeros(rf_shape)
        elif method == 'dim1_split':
            n_cfs = 1
            context_map = np.zeros(rf_shape)
        elif method == 'dim2_split':
            n_cfs = 1
            context_map = np.zeros(rf_shape)
        elif method == 'sign': # mainly used for multiple CFs gradient checking
            n_cfs = 2
            context_map = np.zeros(rf_shape)
            context_map[self.rfs[0].field > 0] = 1
        elif method == 'temporal':
            n_cfs = self.rfs[0].shape[0]
            context_map = np.zeros(rf_shape)
            for i in range(n_cfs):
                context_map[i, :, :] = i
        else:
            raise Exception("Unknown mapping method: {}".format(method))

        self.context_map = context_map
        self.mapping_method = method
        if multilin:
            zero_origin = False

        # Initialize the CFs
        for cf_idx in range(n_cfs):
            origin = [0 for i in shape]
            if alignment == 'edge':
                origin[0] = np.int64(np.ceil((shape[0]) / 2.)) - 1 - 3
            self.cfs.append(Field(shape, reg_c,
                                  multilin=multilin,
                                  init_values=init_values,
                                  origin=origin,
                                  zero_origin=zero_origin,
                                  reg_type=reg_type))

    def init_act_funs(self, cf_act_fun, reg_c):

        if cf_act_fun is not None and self.cf_type != 'ctx':
            # Go with a linear initial guess if the activation function is to
            # be learned, otherwise initialize to the selected shape
            if cf_act_fun == 'adaptive':
                init_shape = 'linear'
            else:
                init_shape = cf_act_fun

            for i in range(len(self.cfs)):
                self.cf_act_funs.append(ActFun(reg_c, init_shape=init_shape))

    def update_context_map(self):
        """ Splits the RF in predefined ways so as to obtain more than one CF

        (Testing stage still!)

        :return:
        """

        added_cf = False

        if self.mapping_method == 'same':
            pass

        elif self.mapping_method == 'dim0_split':
            if len(self.cfs) == 1:
                # Reset previous fields and add a new one
                self._reset_context_fields()
                # self._add_context_field(cf_origin=[0, 0, 0])
                self._add_context_field(cf_origin=self.cfs[0].origin)
                # Split the RF into two context along the 0th dimension based on
                # the location of the maximal RF element
                max_idx = np.unravel_index(self.rfs[0].field.argmax(),
                                           self.rfs[0].shape)
                self.context_map[:max_idx[0] - 1, :, :] = 1

                added_cf = True

        elif self.mapping_method == 'dim1_split':
            if len(self.cfs) == 1:
                # Find the maximal RF element
                max_idx = np.unravel_index(self.rfs[0].field.argmax(),
                                           self.rfs[0].shape)

                # Divide up the region around the maxima into different CFs
                for cf_id in range(2):
                    if cf_id > 0:
                        self._add_context_field()

                    if cf_id == 0:
                        self.context_map[:, :max_idx[1], :] = cf_id
                    else:
                        self.context_map[:, max_idx[1]:, :] = cf_id

                    cf_id += 1

                # plt.ioff()
                # fig = plt.figure()
                # ax1 = fig.add_subplot(1, 2, 1)
                # ax1.imshow(self.rfs[0].field[:, :, 0].T)
                # ax2 = fig.add_subplot(1, 2, 2)
                # cax = ax2.imshow(self.context_map[:, :, 0].T)
                # fig.colorbar(cax)
                # plt.pause(0.1)
                # plt.show()
                # plt.ion()

                added_cf = True

        elif self.mapping_method == 'dim2_split':
            if len(self.cfs) == 1:
                n_cfs = self.rfs[0].shape[-1]
                for i in range(1, n_cfs):
                    self._add_context_field()
                    if self.cfs[0].multilin:
                        self.cfs[i].parts = copy.deepcopy(self.cfs[0].parts)
                    self.cfs[i].field[:] = self.cfs[0].field
                    self.cfs[i].bias = self.cfs[0].bias
                    self.context_map[:, :, i] = i
                    self.cf_act_funs.append(copy.deepcopy(self.cf_act_funs[0]))

                added_cf = True

        return added_cf

    def truncate_rf(self):
        """ Truncates the RF to one point in time and shift the CF accordingly

        Testing stage still!
        Only intended for subunit models

        :return:
        """

        old_shape = self.rfs[0].field.shape
        new_shape = [1] + [i for i in old_shape[1:]]
        max_idx = np.abs(self.rfs[0].field).argmax()
        max_pos = np.unravel_index(max_idx, old_shape)

        # Truncate the RF
        self.rfs[0].field = self.rfs[0].field[max_pos[0], :, :].reshape(new_shape)
        self.rfs[0].shape = self.rfs[0].field.shape

        # New RF regularization matrix
        reg_l = smooth_reg_l(new_shape)
        reg_l = sparse.csr_matrix(reg_l)
        self.rfs[0].reg_l = reg_l

        # Time shift the CF
        n_shifts = old_shape[0] - max_pos[0] - 1
        for cf in self.cfs:
            origin_shift = cf.shape[0] / 2 - cf.origin[0] - 1
            cf.origin[0] = cf.shape[0] / 2 - 1
            n_shifts_tmp = n_shifts - origin_shift
            cf.field = np.roll(cf.field, -n_shifts_tmp, axis=0)

    def check_rf(self):
        """ Flips arbitrary signs to make the solutions conform to a standard

        (Testing stage still!)

        :return:
        """

        flipped = False

        # Get a mainly positive RF
        rf_abs_pos = np.abs(self.rfs[0].field.max())
        rf_abs_min = np.abs(self.rfs[0].field.min())
        if rf_abs_min > rf_abs_pos and self.cf_type == 'subunit':
            self.rfs[0].field *= -1
            flipped = True

        return flipped

    def mask_cf(self):
        """ Masks out the CF surround (around the origin)

        (Testing stage still!)

        :return:
        """

        if self.cf_type == 'subunit':
            for cf in self.cfs:
                mask = gaussian_field(cf.shape, cf.origin)
                cf.field *= mask



    def _reset_context_fields(self):
        """ Reset to all CFs to zero with a unitary bias """
        for cf in self.cfs:
            cf.field[:, :, :] = 0
            cf.bias = 1

    def _add_context_field(self, cf_origin=[]):
        """ Add a new empty context field

        :return cf_id: the index of the newly created field.
        """

        # Copy field structure data from a previous CF
        cf_shape = self.cfs[0].shape
        reg_c = self.cfs[0].reg_c
        multilin = self.cfs[0].multilin
        if len(cf_origin) == 0: cf_origin = self.cfs[0].origin

        # Create and append a new context field
        new_cf = Field(cf_shape, reg_c,
                       multilin=multilin,
                       init_values='zeros',
                       origin=cf_origin,
                       zero_origin=True)
        new_cf.reg_c_from_cv = self.cfs[0].reg_c_from_cv
        self.cfs.append(new_cf)

        # Return the index of the newly created field
        return len(self.cfs)
