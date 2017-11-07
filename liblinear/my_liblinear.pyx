"""
Wrapper for liblinear

Author: fabian.pedregosa@inria.fr

Original file (liblinear.pyx) from skicit learn
modified by: johan.westo@gmail.com

Modified to specifically solve modified logistic and poisson 
regression problems on dense matrices, see Westo & May (2017).

"""

import  numpy as np
cimport numpy as np
cimport my_liblinear

np.import_array()


def my_train_wrap(np.ndarray[np.float64_t, ndim=2, mode='c'] X,
                  np.ndarray[np.float64_t, ndim=1, mode='c'] Y,
		          np.ndarray[np.float64_t, ndim=2, mode='c'] reg_ltl,
                  np.ndarray[np.float64_t, ndim=1, mode='c'] weights,
                  np.ndarray[np.float64_t, ndim=1, mode='c'] bias_i,
                  np.ndarray[np.float64_t, ndim=1, mode='c'] init_sol,
                  int solver_type,
                  double eps,
                  double C,
                  int nr_thread):

    cdef parameter *param
    cdef problem *problem
    cdef model *model
    cdef char_const_ptr error_msg
    cdef int len_w

    # Define the problem
    # The bias is explicitely set to zero and it therefore has to be
    # included in the input array x at this stage already
    problem = set_problem(
        X.data,
        Y.data,
	    reg_ltl.data,
        weights.data,
        bias_i.data,
        X.shape,
        -1.0)

    # Define parameters
    cdef int warm_start
    if init_sol.size > 0:
        warm_start = 1
    else:
        warm_start = 0
    param = set_parameter(solver_type, eps, C, nr_thread,
                          init_sol.data, warm_start)

    # Verify parameters
    error_msg = check_parameter(problem, param)
    if error_msg:
        free_parameter(param)
        raise ValueError(error_msg)

    # early return
    with nogil:
        model = train(problem, param)

    # coef matrix holder created as fortran since that's what's used in liblinear
    cdef np.ndarray[np.float64_t, ndim=2, mode='fortran'] w
    cdef int nr_feature = get_nr_feature(model)
    w = np.empty((1, nr_feature),order='F')
    copy_w(w.data, model, nr_feature)

    ### FREE
    free_and_destroy_model(&model)
    free_parameter(param)

    return w


def set_verbosity_wrap(int verbosity):
    """
    Control verbosity of libsvm library
    """
    set_verbosity(verbosity)

