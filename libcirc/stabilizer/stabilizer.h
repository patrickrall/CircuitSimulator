#include "../utils/comms.h"


//define K from RandomStabilizerState algorithm (page 16)
struct StabilizerState {
	int n;
	int k;
    struct BitVector* h;		//in \mathbb{F}^n_2
	struct BitMatrix* G;		//in \mathbb{F}^{n\times n}_2
	struct BitMatrix* Gbar;	//= (G^-1)^T

	int Q;				//in \mathbb{Z}_8
	struct BitVector* D2; // in {0,2,4,6}^k
	struct BitVector* D1; // D[i] = 2*D2[i] + D1[i]
	struct BitMatrix* J; //in {0,4}^{k\times k}, symmetric
};


struct StabilizerState* allocStabilizerState(int n, int k);
void deepCopyState(struct StabilizerState *dest, struct StabilizerState *src);
void freeStabilizerState(struct StabilizerState* state);
void printStabilizerState(struct StabilizerState* state);  // prints python code for easy debugging

void exponentialSumExact(struct StabilizerState* state, int* eps, int* p, int* m);
Complex exponentialSum(struct StabilizerState* state);

int shrink(struct StabilizerState* state, struct BitVector* xi, int alpha, int lazy);

void innerProductExact(struct StabilizerState* state1, struct StabilizerState* state2, int* eps, int* p, int* m);
Complex innerProduct(struct StabilizerState* state1, struct StabilizerState* state2);

struct StabilizerState* randomStabilizerState(int n);

double measurePauli(struct StabilizerState* state, int m, struct BitVector* zeta, struct BitVector* xi);
void extend(struct StabilizerState* state, struct BitVector* xi);

int getD(struct StabilizerState* state, int i);
void setD(struct StabilizerState* state, int i, int val);
