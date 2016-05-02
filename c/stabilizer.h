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

double randDouble(double min, double max);

//define K from RandomStabilizerState algorithm (page 16)
struct StabilizerStates {
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

void deepCopyState(struct StabilizerStates *dest, struct StabilizerStates *src);

void updateDJ(struct StabilizerStates *state, gsl_matrix *R);

void updateQD(struct StabilizerStates *state, gsl_vector *y);

void evalW(gsl_complex *ans, int eps, int p, int m);

void Gamma(int *eps, int *p, int *m, int A, int B);

void partialGamma(int *eps, int *p, int *m, int A);

void Wsigma(struct StabilizerStates *state, int *eps, int *p, int *m, gsl_complex *ans, 
	int exact, int sigma, int s, int *M, int Mlength, int *Dimers, int DimersLength);

void exponentialSum(struct StabilizerStates *state, int *eps, int *p, int *m, gsl_complex *ans, int exact);

int shrink(struct StabilizerStates *state, gsl_vector *xi, int alpha, int lazy);

void innerProduct(struct StabilizerStates *state1, struct StabilizerStates *state2, int *eps, int *p, int *m, gsl_complex *ans, int exact);

double logeta(int d, int n);

void randomStabilizerState(struct StabilizerStates *state, int n);

void extend(struct StabilizerStates *state, gsl_vector *xi);

double measurePauli(struct StabilizerStates *state, int m, gsl_vector *zeta, gsl_vector *xi);
