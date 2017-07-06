#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

int mod(int a, int b);

//define K from RandomStabilizerState algorithm (page 16)
struct StabilizerState {
	int n;
	int k;
	gsl_vector *h;		//in \mathbb{F}^n_2
	gsl_matrix *G;		//in \mathbb{F}^{n\times n}_2
	gsl_matrix *Gbar;	//= (G^-1)^T

	int Q;				//in \mathbb{Z}_8
	gsl_vector *D;		//in {0,2,4,6}^k
	gsl_matrix *J;		//in {0,4}^{k\times k}, symmetric
};


struct StabilizerState* allocStabilizerState(int n, int k);

void deepCopyState(struct StabilizerState *dest, struct StabilizerState *src);

void exponentialSum(struct StabilizerState *state, int *eps, int *p, int *m, gsl_complex *ans, int exact);

int shrink(struct StabilizerState *state, gsl_vector *xi, int alpha, int lazy);

void innerProduct(struct StabilizerState *state1, struct StabilizerState *state2, int *eps, int *p, int *m, gsl_complex *ans, int exact);

struct StabilizerState* randomStabilizerState(int n);

void extend(struct StabilizerState *state, gsl_vector *xi);

double measurePauli(struct StabilizerState *state, int m, gsl_vector *zeta, gsl_vector *xi);

void freeStabilizerState(struct StabilizerState *state);

void printStabilizerState(struct StabilizerState *state);
