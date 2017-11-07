#include <stdlib.h>
#include <numpy/arrayobject.h>
#include "linear.h"

/*
 * This code was originally part of scikit-learn (liblinear_helper.c)
 *
 * It has been cleande modified to only support dense matrices and to solve 
 * the problems in Westo & May (2017)
 *
 */

/* Create a problem struct with and return it */
struct problem * set_problem(char *X, char *Y, char *reg_ltl, char *weights, char *bias_i, npy_intp *dims, double bias)
{
    struct problem *problem;
    /* not performant but simple */
    problem = malloc(sizeof(struct problem));
    if (problem == NULL) return NULL;
    problem->l = (int) dims[0];

    // NOTE, the bias term is currently not implemented!!!
    if (bias > 0) {
        problem->n = (int) dims[1] + 1;
    } else {
        problem->n = (int) dims[1];
    }
    problem->x = (double *) X;
    problem->y = (double *) Y;
    problem->reg_ltl = (double *) reg_ltl;
    problem->weights = (double *) weights;
    problem->bias_i = (double *) bias_i;
    problem->bias = bias;
    if (problem->x == NULL) { 
        free(problem);
        return NULL;
    }

    return problem;
}


/* Create a paramater struct with and return it */
struct parameter *set_parameter(int solver_type, double eps, double C, int nr_thread, char* init_sol, int warm_start)
{
    struct parameter *param = malloc(sizeof(struct parameter));
    if (param == NULL)
        return NULL;

    param->solver_type = solver_type;
    param->eps = eps;
    param->C = C;
    param->nr_thread = nr_thread;
    if (warm_start > 0)
        param->init_sol = (double *) init_sol;
    else
        param->init_sol = NULL;
    return param;
}

void copy_w(void *data, struct model *model, int len)
{
    memcpy(data, model->w, len * sizeof(double)); 
}

double get_bias(struct model *model)
{
    return model->bias;
}

void free_parameter(struct parameter *param)
{
    free(param);
}

/* rely on built-in facility to control verbose output */
static void print_null(const char *s) {}

static void print_string_stdout(const char *s)
{
    fputs(s ,stdout);
    fflush(stdout);
}

/* provide convenience wrapper */
void set_verbosity(int verbosity_flag){
    if (verbosity_flag)
        set_print_string_function(&print_string_stdout);
    else
        set_print_string_function(&print_null);
}
