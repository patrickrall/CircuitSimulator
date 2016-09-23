#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_linalg.h>

int mod(int a, int b);

//define K from RandomStabilizerState algorithm (page 16)
struct StabilizerState {
	int n;
	int k;
	gsl_vector *h;		//in \mathbb{F}^n_2
	gsl_matrix *G;		//in \mathbb{F}^{n\times n}_2
	gsl_matrix *Gbar;	//= (G^-1)^T

	//define q to be zero for all x
	int Q;				//in \mathbb{Z}_8
	gsl_vector *D;		//in {0,2,4,6}^k
	gsl_matrix *J;		//in {0,4}^{k\times k}, symmetric
};

void deepCopyState(struct StabilizerState *dest, struct StabilizerState *src);

void exponentialSum(struct StabilizerState *state, int *eps, int *p, int *m, gsl_complex *ans, int exact);

int shrink(struct StabilizerState *state, gsl_vector *xi, int alpha, int lazy);

void innerProduct(struct StabilizerState *state1, struct StabilizerState *state2, int *eps, int *p, int *m, gsl_complex *ans, int exact);

void randomStabilizerState(struct StabilizerState *state, int n);

void extend(struct StabilizerState *state, gsl_vector *xi);

double measurePauli(struct StabilizerState *state, int m, gsl_vector *zeta, gsl_vector *xi);

void freeStabilizerState(struct StabilizerState *state);
