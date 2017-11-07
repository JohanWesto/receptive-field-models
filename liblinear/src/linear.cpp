#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <locale.h>
#include "linear.h"
#include "tron.h"
#include <omp.h>
typedef signed char schar;
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

#ifdef __cplusplus
extern "C" {
#endif
extern double ddot_(int *, double *, int *, double *, int *);
extern int daxpy_(int *, double *, double *, int *, double *, int *);
#ifdef __cplusplus
}
#endif

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif

static inline int rand_int(const int max)
{
	static int seed = omp_get_thread_num();
	seed = ((seed * 1103515245) + 12345) & 0x7fffffff;
	return seed%max;
}

/*
 * This code is originally taken from Mutli-core LIBLINEAR (linear.cpp)
 *
 * It has been modified to specifically solve modified logistic and poisson 
 * regression problems on dense matrices, see Westo & May (2017).
 *
 */

class Reduce_Vectors
{
public:
	Reduce_Vectors(int size);
	~Reduce_Vectors();

	void init(void);
	void sum_scale_x(int n, double scalar, double *x);
	void sum_scale_x_poly(int n, double scalar, double *x);
	void reduce_sum(double* v);

private:
	int nr_thread;
	int size;
	double **tmp_array;
};

Reduce_Vectors::Reduce_Vectors(int size)
{
	nr_thread = omp_get_max_threads();
	this->size = size;
	tmp_array = new double*[nr_thread];
	for(int i = 0; i < nr_thread; i++)
		tmp_array[i] = new double[size];
}

Reduce_Vectors::~Reduce_Vectors(void)
{
	for(int i = 0; i < nr_thread; i++)
		delete[] tmp_array[i];
	delete[] tmp_array;
}

void Reduce_Vectors::init(void)
{
#pragma omp parallel for schedule(static)
	for(int i = 0; i < size; i++)
		for(int j = 0; j < nr_thread; j++)
			tmp_array[j][i] = 0.0;
}

void Reduce_Vectors::sum_scale_x(int n, double scalar, double *x)
{
	int inc = 1;
	int thread_id = omp_get_thread_num();
	daxpy_(&n, &scalar, x, &inc, tmp_array[thread_id], &inc);
}

void Reduce_Vectors::sum_scale_x_poly(int n, double scalar, double *x)
{
	int inc = 1;
	int count = 0;
	int thread_id = omp_get_thread_num();

	daxpy_(&n, &scalar, x, &inc, tmp_array[thread_id], &inc);

	// Indices start from one, so as to ignore the bias term
	for(int i = 1; i < n; i++){
		for(int j = 1; j <= i; j++){
			tmp_array[thread_id][n+count] += scalar*x[i]*x[j];
			count += 1;
		}	
	}
}

void Reduce_Vectors::reduce_sum(double* v)
{
#pragma omp parallel for schedule(static)
	for(int i = 0; i < size; i++)
	{
		v[i] = 0;
		for(int j = 0; j < nr_thread; j++)
			v[i] += tmp_array[j][i];
	}
}


// Definition of a modified Tikhonov-regularized logistic regression problem
//
//  min_w 1/2 w^T w + C \sum log(1+exp(-yi w^T xi + biasi)),
//
// Given: 
// x, y, C, biasi
//
// The additional term biasi have been added to the original liblinear formulation
// See Westo and May (2017)

class ctx_lr_fun: public function
{
public:
	ctx_lr_fun(const problem *prob, double *C);
	~ctx_lr_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	Reduce_Vectors *reduce_vectors;

	const problem *prob;
};

ctx_lr_fun::ctx_lr_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	reduce_vectors = new Reduce_Vectors(get_nr_variable());

	this->C = C;
}

ctx_lr_fun::~ctx_lr_fun()
{
	delete[] z;
	delete[] D;
	delete reduce_vectors;
}


double ctx_lr_fun::fun(double *w)
{
	int i;
	int inc = 1;
	int l=prob->l;
	int n = prob->n;
	double f = 0;
	double *y = prob->y;
	double *reg_ltl = prob->reg_ltl;
	double *tmp_array = new double[n];

	Xv(w, z);

#pragma omp parallel for private(i) schedule(static)
	for(i=0;i<n;i++)
		tmp_array[i] = ddot_(&n, w, &inc, &reg_ltl[i*n], &inc);
	
	f += ddot_(&n, w, &inc, tmp_array, &inc);
	f /= 2.0;

#pragma omp parallel for private(i) reduction(+:f) schedule(static)
	for(i=0;i<l;i++)
	{
		double yz = y[i]*z[i];
		if (yz >= 0)
			f += C[i]*log(1 + exp(-yz));
		else
			f += C[i]*(-yz+log(1 + exp(yz)));
	}
	
	delete[] tmp_array;

	return(f);
}

void ctx_lr_fun::grad(double *w, double *g)
{
	int i;
	int inc = 1;
	int l = prob->l;
	int n = prob->n;
	double *y = prob->y;
	double *reg_ltl = prob->reg_ltl;

#pragma omp parallel for private(i) schedule(static)
	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i]*(z[i]-1)*y[i];
	}
	XTv(z, g);

#pragma omp parallel for private(i) schedule(static)
	for(i = 0; i < n; i++)
		g[i] = g[i] + ddot_(&n, w, &inc, &reg_ltl[i*n], &inc);;
}

int ctx_lr_fun::get_nr_variable(void)
{
	return prob->n;
}

void ctx_lr_fun::Hv(double *s, double *Hs)
{
	int i;
	int inc = 1;
	int l=prob->l;
	int n = prob->n;
	double *wa = new double[l];
	double *x = prob->x;
	double *reg_ltl = prob->reg_ltl;

	reduce_vectors->init();

#pragma omp parallel for private(i) schedule(guided)
	for(i=0;i<l;i++)
	{
		wa[i] = ddot_(&n, s, &inc, &x[i*n], &inc);
		wa[i] = C[i]*D[i]*wa[i];
		reduce_vectors->sum_scale_x(n, wa[i], &x[i*n]);
	}

	reduce_vectors->reduce_sum(Hs);

#pragma omp parallel for private(i) schedule(static)
	for(i=0;i<n;i++)
		Hs[i] = ddot_(&n, s, &inc, &reg_ltl[i*n], &inc) + Hs[i];

	delete[] wa;
}

void ctx_lr_fun::Xv(double *v, double *Xv)
{
	int i;
	int inc = 1;
	int n = prob->n;
	int l=prob->l;
	double *x = prob->x;
	double *bias_i = prob->bias_i;

#pragma omp parallel for private (i) schedule(guided)
	for(i=0;i<l;i++){
		Xv[i] = ddot_(&n, v, &inc, &x[i*n], &inc);
		Xv[i] += bias_i[i];
	}
}

void ctx_lr_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int n = prob->n;
	double *x = prob->x;

	reduce_vectors->init();

#pragma omp parallel for private(i) schedule(guided)
	for(i=0;i<l;i++)
		reduce_vectors->sum_scale_x(n, v[i], &x[i*n]);

	reduce_vectors->reduce_sum(XTv);
}

// Definition of a modified Tikhonov-regularized poisson regression problem
//
//  min_w 1/2 w^T w - C \sum yi (w^T xi + biasi) + exp(w^T xi + biasi),
//
// Given: 
// x, y, C, biasi
//
// The additional term biasi distinguishes the modified problem from original
// See Westo and May (2017)


class ctx_pr_fun: public function
{
public:
	ctx_pr_fun(const problem *prob, double *C);
	~ctx_pr_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	Reduce_Vectors *reduce_vectors;

	const problem *prob;
};

ctx_pr_fun::ctx_pr_fun(const problem *prob, double *C)
{
	int l=prob->l;

	this->prob = prob;

	z = new double[l];
	D = new double[l];

	reduce_vectors = new Reduce_Vectors(get_nr_variable());

	this->C = C;
}

ctx_pr_fun::~ctx_pr_fun()
{
	delete[] z;
	delete[] D;
	delete reduce_vectors;
}


double ctx_pr_fun::fun(double *w)
{
	int i;
	int inc = 1;
	int l = prob->l;
	int n = prob->n;
	double f=0;
	double *y=prob->y;
	double *reg_ltl = prob->reg_ltl;
	double *tmp_array = new double[n];

	Xv(w, z);

#pragma omp parallel for private(i) schedule(static)
	for(i=0;i<n;i++)
		tmp_array[i] = ddot_(&n, w, &inc, &reg_ltl[i*n], &inc);
	
	f += ddot_(&n, w, &inc, tmp_array, &inc);
	f /= 2.0;

#pragma omp parallel for private(i) reduction(+:f) schedule(static)
	for(i=0;i<l;i++)
		f += -C[i]*y[i]*z[i] + C[i]*exp(z[i]);

	delete[] tmp_array;

	return(f);
}

void ctx_pr_fun::grad(double *w, double *g)
{
	int i;
	int inc = 1;
	int l = prob->l;
	int n = prob->n;
	double *y=prob->y;
	double *reg_ltl = prob->reg_ltl;

#pragma omp parallel for private(i) schedule(static)
	for(i=0;i<l;i++)
	{
		D[i] = exp(z[i]);
		z[i] = C[i]*(-y[i] + D[i]);
	}
	XTv(z, g);

#pragma omp parallel for private(i) schedule(static)
	for(i = 0; i < n; i++)
		g[i] = g[i] + ddot_(&n, w, &inc, &reg_ltl[i*n], &inc);;
}

int ctx_pr_fun::get_nr_variable(void)
{
	return prob->n;
}

void ctx_pr_fun::Hv(double *s, double *Hs)
{
	int i;
	int inc = 1;
	int l=prob->l;
	int n = prob->n;
	double *wa = new double[l];
	double *x = prob->x;
	double *reg_ltl = prob->reg_ltl;

	reduce_vectors->init();

#pragma omp parallel for private(i) schedule(guided)
	for(i=0;i<l;i++)
	{
		wa[i] = ddot_(&n, s, &inc, &x[i*n], &inc);
		wa[i] = C[i]*D[i]*wa[i];
		reduce_vectors->sum_scale_x(n, wa[i], &x[i*n]);
	}

	reduce_vectors->reduce_sum(Hs);

#pragma omp parallel for private(i) schedule(static)
	for(i=0;i<n;i++)
		Hs[i] = ddot_(&n, s, &inc, &reg_ltl[i*n], &inc) + Hs[i];

	delete[] wa;
}

void ctx_pr_fun::Xv(double *v, double *Xv)
{
	int i;
	int inc = 1;
	int l=prob->l;
	int n = prob->n;
	double *bias_i = prob->bias_i;
	double *x = prob->x;

#pragma omp parallel for private (i) schedule(guided)
	for(i=0;i<l;i++){
		Xv[i] = ddot_(&n, v, &inc, &x[i*n], &inc);
		Xv[i] += bias_i[i];
	}
}

void ctx_pr_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int n = prob->n;
	double *x = prob->x;

	reduce_vectors->init();

#pragma omp parallel for private(i) schedule(guided)
	for(i=0;i<l;i++)
		reduce_vectors->sum_scale_x(n, v[i], &x[i*n]);
	
	reduce_vectors->reduce_sum(XTv);
}

// Definition of a modified L2-regularized logistic regression problem
//
//  min_w 1/2 w^T w + C \sum log(1+exp(-yi w^T xi + biasi)),
//
// Given: 
// x, y, C
//
// Intended to solve MNE models
// See Westo and May (2017)

class mne_lr_fun: public function
{
public:
	mne_lr_fun(const problem *prob, double *C);
	~mne_lr_fun();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);

private:
	void Xv(double *v, double *Xv);
	void XTv(double *v, double *XTv);

	double *C;
	double *z;
	double *D;
	Reduce_Vectors *reduce_vectors;

	const problem *prob;
};

mne_lr_fun::mne_lr_fun(const problem *prob, double *C)
{
	int l = prob->l;
	int n = prob->n;
	int w_size = 1+(n-1)*(n-1+3)/2;

	this->prob = prob;

	z = new double[l];
	D = new double[l];
	reduce_vectors = new Reduce_Vectors(w_size);

	this->C = C;
	info("mne_lr_fun created\n");
}

mne_lr_fun::~mne_lr_fun()
{
	delete[] z;
	delete[] D;
	delete reduce_vectors;
}


double mne_lr_fun::fun(double *w)
{
	int i;
	int l=prob->l;
	int n = prob->n;
	int w_size = 1+(n-1)*(n-1+3)/2;
	double f = 0;
	double *y = prob->y;

	Xv(w, z);

#pragma omp parallel for private(i) reduction(+:f) schedule(static)
	for(i=0;i<w_size;i++)
		f += w[i]*w[i];

	f /= 2.0;

#pragma omp parallel for private(i) reduction(+:f) schedule(static)
	for(i=0;i<l;i++)
	{
		double yz = y[i]*z[i];
		if (yz >= 0)
			f += C[i]*log(1 + exp(-yz));
		else
			f += C[i]*(-yz+log(1 + exp(yz)));
	}

	return(f);
}

void mne_lr_fun::grad(double *w, double *g)
{
	int i;
	int l = prob->l;
	int n = prob->n;
	int w_size = 1+(n-1)*(n-1+3)/2;
	double *y = prob->y;

#pragma omp parallel for private(i) schedule(static)
	for(i=0;i<l;i++)
	{
		z[i] = 1/(1 + exp(-y[i]*z[i]));
		D[i] = z[i]*(1-z[i]);
		z[i] = C[i]*(z[i]-1)*y[i];
	}
	XTv(z, g);

//#pragma omp parallel for private(i) schedule(static)
	for(i = 0; i < w_size; i++){
		g[i] = g[i] + w[i];
	}
}

int mne_lr_fun::get_nr_variable(void)
{
	int n = prob->n;
	int w_size = 1+(n-1)*(n-1+3)/2;
	return w_size;
}

void mne_lr_fun::Hv(double *s, double *Hs)
{
	int i, j, k, count;
	int inc = 1;
	int l=prob->l;
	int n = prob->n;
	int w_size = 1+(n-1)*(n-1+3)/2;
	double *wa = new double[l];
	double *x = prob->x;

	reduce_vectors->init();

#pragma omp parallel for private(i, j, k, count) schedule(static)
	for(i=0;i<l;i++)
	{
		// Bias and linear terms
		wa[i] = ddot_(&n, s, &inc, &x[i*n], &inc);
		// Quadratic terms
		count = 0;
		// Indices start from one, so as to ignore the bias term
		for (j=1;j<n;j++){
			for (k=1;k<=j;k++){
				wa[i] += x[i*n+j] * x[i*n+k] * s[n+count];
				count += 1;
			}
		}

		wa[i] = C[i]*D[i]*wa[i];
		reduce_vectors->sum_scale_x_poly(n, wa[i], &x[i*n]);
	}

	reduce_vectors->reduce_sum(Hs);

#pragma omp parallel for private(i) schedule(static)
	for(i=0;i<w_size;i++)
		Hs[i] = s[i] + Hs[i];


	delete[] wa;
}

void mne_lr_fun::Xv(double *v, double *Xv)
{
	int i, j, k, count;
	int inc = 1;
	int l = prob->l;
	int n = prob->n;
	double *x = prob->x;

#pragma omp parallel for private (i, j, k, count) schedule(static)
	for(i=0;i<l;i++){
		// Bias and linear terms
		Xv[i] = ddot_(&n, v, &inc, &x[i*n], &inc);
		// Quadratic terms
		count = 0;
		// Indices start from one, so as to ignore the bias term
		for (j=1;j<n;j++){
			for (k=1;k<=j;k++){
				Xv[i] += x[i*n+j] * x[i*n+k] * v[n+count];
				count += 1;
			}
		}
	}
}

void mne_lr_fun::XTv(double *v, double *XTv)
{
	int i;
	int l=prob->l;
	int n = prob->n;
	double *x = prob->x;

	reduce_vectors->init();

#pragma omp parallel for private(i) schedule(guided)
	for(i=0;i<l;i++)
		reduce_vectors->sum_scale_x_poly(n, v[i], &x[i*n]);

	reduce_vectors->reduce_sum(XTv);
}

static void train_one(const problem *prob, const parameter *param, double *w)
{
	//inner and outer tolerances for TRON
	double eps = param->eps;
	double eps_cg = 0.1;
	if(param->init_sol != NULL)
		eps_cg = 0.5;

	int pos = 0;
	int neg = 0;
	for(int i=0;i<prob->l;i++)
		if(prob->y[i] > 0)
			pos++;
	neg = prob->l - pos;
	double primal_solver_tol = eps*max(min(pos,neg), 1)/prob->l;

	double *C = new double[prob->l];
	for(int i = 0; i < prob->l; i++)
	{
		C[i] = param->C * prob->weights[i];
	}

	function *fun_obj=NULL;
	switch(param->solver_type)
	{
		case CTX_LR:
		{
			fun_obj=new ctx_lr_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol, eps_cg);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;
		}
		case CTX_PR:
		{
			fun_obj=new ctx_pr_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol, eps_cg);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;
		}
		case MNE_LR:
		{
			fun_obj=new mne_lr_fun(prob, C);
			TRON tron_obj(fun_obj, primal_solver_tol, eps_cg);
			tron_obj.set_print_string(liblinear_print_string);
			tron_obj.tron(w);
			delete fun_obj;
			delete[] C;
			break;
		}
		default:
			fprintf(stderr, "ERROR: unknown solver_type\n");
			break;
	}
}

//
// Interface functions
//
model* train(const problem *prob, const parameter *param)
{
	int i, w_size;
	int n = prob->n;
	model *model_ = Malloc(model,1);

	if (param->solver_type == MNE_LR)
		w_size = 1+(n-1)*(n-1+3)/2;
	else
		w_size = n;

	if(prob->bias>=0)
		model_->nr_feature=w_size-1;
	else
		model_->nr_feature=w_size;

	model_->param = *param;
	model_->bias = prob->bias;

	// Mutlicore
	omp_set_num_threads(param->nr_thread);
	
	model_->w=Malloc(double, w_size);
	if(param->init_sol != NULL)
		for(i=0;i<w_size;i++)
			model_->w[i] = param->init_sol[i];
	else
		for(i=0;i<w_size;i++)
			model_->w[i] = 0;

	train_one(prob, param, model_->w);

	
	return model_;
}

int get_nr_feature(const model *model_)
{
	return model_->nr_feature;
}

void free_model_content(struct model *model_ptr)
{
	if(model_ptr->w != NULL)
		free(model_ptr->w);
}

void free_and_destroy_model(struct model **model_ptr_ptr)
{
	struct model *model_ptr = *model_ptr_ptr;
	if(model_ptr != NULL)
	{
		free_model_content(model_ptr);
		free(model_ptr);
	}
}

const char *check_parameter(const problem *prob, const parameter *param)
{
	if(param->eps <= 0)
		return "eps <= 0";

	if(param->C <= 0)
		return "C <= 0";

	if(param->solver_type != CTX_LR && param->solver_type != CTX_PR && param->solver_type != MNE_LR)
		return "unknown solver type";

	return NULL;
}

void set_print_string_function(void (*print_func)(const char*))
{
	if (print_func == NULL)
		liblinear_print_string = &print_string_stdout;
	else
		liblinear_print_string = print_func;
}

