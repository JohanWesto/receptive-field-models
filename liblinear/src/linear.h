#ifndef _LIBLINEAR_H
#define _LIBLINEAR_H

/*
 * This code is originally taken from Mutli-core LIBLINEAR (linear.h)
 *
 * It has been modified to specifically solve modified logistic and poisson 
 * regression problems on dense matrices, see Westo & May (2017).
 *
 */

#ifdef __cplusplus
extern "C" {
#endif

struct feature_node
{
	int index;
	double value;
};

struct problem
{
	int l, n;
	double *y;
	double *x;
	double bias;            /* < 0 if no bias term */  
	// Additions: see Westo & May (2017)
	double *reg_ltl;  // Tikhonov regularization matrix, LTL
	double *bias_i;
	double *weights;
};

enum {CTX_LR, CTX_PR, MNE_LR}; /* solver_type */

struct parameter
{
	int solver_type;

	/* these are for training only */
	double eps;	        /* stopping criteria */
	double C;
	int nr_thread;
	double *init_sol;
};

struct model
{
	struct parameter param;
	int nr_feature;
	double *w;
	double bias;
};

struct model* train(const struct problem *prob, const struct parameter *param);

int get_nr_feature(const struct model *model_);

void free_model_content(struct model *model_ptr);
void free_and_destroy_model(struct model **model_ptr_ptr);
void destroy_param(struct parameter *param);

const char *check_parameter(const struct problem *prob, const struct parameter *param);
void set_print_string_function(void (*print_func) (const char*));

#ifdef __cplusplus
}
#endif

#endif /* _LIBLINEAR_H */

