"""
Wrapper for liblinear

Author: fabian.pedregosa@inria.fr

Original file (liblinear.pxd) from skicit learn
modified by: johan.westo@gmail.com

Modified to specifically solve modified logistic and poisson 
regression problems on dense matrices, see Westo & May (2017).
"""

cimport numpy as np

cdef extern from "src/linear.h":
    cdef struct problem
    cdef struct model
    cdef struct parameter
    ctypedef problem* problem_const_ptr "problem const *"
    ctypedef parameter* parameter_const_ptr "parameter const *"
    ctypedef char* char_const_ptr "char const *"
    char_const_ptr check_parameter(problem_const_ptr prob, parameter_const_ptr param)
    model *train(problem_const_ptr prob, parameter_const_ptr param) nogil
    int get_nr_feature (model *model)
    void free_and_destroy_model (model **)

cdef extern from "src/liblinear_helper.c":
    void copy_w(void *, model *, int)
    parameter *set_parameter(int, double, double, int, char *, int)
    problem *set_problem (char *, char *, char *, char *, char *, np.npy_intp *, double)

    model *set_model(parameter *, char *, np.npy_intp *, char *, double)

    double get_bias(model *)
    void free_problem (problem *)
    void free_parameter (parameter *)
    void set_verbosity(int)

