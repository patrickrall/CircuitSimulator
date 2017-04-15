#include "../utils/matrix.h"


//define K from RandomStabilizerState algorithm (page 16)
struct StabilizerState {
	int n;
	int k;
    struct BitVector* h;		//in \mathbb{F}^n_2
	struct BitMatrix* G;		//in \mathbb{F}^{n\times n}_2
	struct BitMatrix* Gbar;	//= (G^-1)^T

	int Q;				//in \mathbb{Z}_8
	struct BitMatrix *D2  // in {0,2,4,6}^k
	struct BitMatrix *D1; // D[i] = 2*D2[i] + D1[i]
	struct BitMatrix* J; //in {0,4}^{k\times k}, symmetric
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
