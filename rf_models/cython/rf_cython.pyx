#!/usr/bin/python
"""
" @section DESCRIPTION
" Cython code for speeding up certain RF model functions
"""

import numpy as np
import multiprocessing as mp
cimport scipy.linalg.cython_blas as blas
cimport cython
from cython.parallel import prange, parallel
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cross_corr_c(double[:] x, double[:] rf, int n_samples, int stride, int win_size):
    """ Cross-correlation

    Calculates the cross correlation between an input and a receptive field
    without having to create a larger input matrix with fake dimensions.

    Args:
        x: input array (1-D), use x.ravel() if x is multidimensional
        rf: receptive field (1-D), use rf.ravel() if rf is multidimensional
        n_samples: output size, x.shape[0] - rf.shape[0] + 1
        stride: strides along x, reduce(mul, x.shape[1:])
        win_size: number of elements in the receptive field
    Returns:
        z: similarity score (cross-correlation)
    Raise

    """

    cdef int i, j, offset
    cdef int shift = 1
    cdef int nr_cpu = mp.cpu_count()
    cdef double[:] z = np.empty(n_samples)

    # offset = 0
    # for i in range(n_samples):
    #     z[i] = blas.ddot( & win_size, & x[offset], & shift, & rf[0], & shift)
    #     offset += stride

    # Multicore version
    with nogil, parallel(num_threads=nr_cpu):
        for i in prange(n_samples, schedule='static'):
            offset = i * stride
            z[i] = blas.ddot(&win_size, &x[offset], &shift, &rf[0], &shift)

    return z


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def cf_mat_der_c(double[:] x, double[:] e, double[:] rf, int n_samples, int stride, int win_size):

    cdef int i, j, offset
    cdef int shift = 1
    cdef double[:] cf_der_sum = np.zeros(win_size)

    offset = 0
    for i in range(n_samples):
        for j in range(win_size):
            cf_der_sum[j] += e[i]*x[offset+j]*rf[j]
        offset += stride

    return cf_der_sum