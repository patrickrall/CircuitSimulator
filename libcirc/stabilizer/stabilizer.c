#include "stabilizer.h"
#include <assert.h>

// --------------------------------- memory and debug -------------------------------
// Create an empty stabilizer state as used by
// the RandomStabilizerState function. It has
// K =\mathbb{F}^n_2 and has q(x) = 0 for all x.
struct StabilizerState* allocStabilizerState(int n, int k) {
    struct StabilizerState *state = (struct StabilizerState *)malloc(sizeof(struct StabilizerState));

    state->n = n;
    state->k = k;

    state->h = newBitVector(n);
    state->G = newBitMatrixIdentity(n, n);
    state->Gbar = newBitMatrixIdentity(n, n);

	state->Q = 0;
	state->D2 = newBitVector(n);
	state->D1 = newBitVector(n);
	state->J = newBitMatrixZero(n, n);

    return state;
}

void deepCopyState(struct StabilizerState *dest, struct StabilizerState *src) {
	dest->n = src->n;
	dest->k = src->k;
	
    BitVectorCopy(dest->h, src->h);
	BitMatrixCopy(dest->G, src->G);
	BitMatrixCopy(dest->Gbar, src->Gbar);

    dest->Q = src->Q;
	BitVectorCopy(dest->D1, src->D1);
	BitVectorCopy(dest->D2, src->D2);
	BitMatrixCopy(dest->J, src->J);
}

void freeStabilizerState(struct StabilizerState* state) {
    BitVectorFree(state->h);
    BitMatrixFree(state->G);
    BitMatrixFree(state->Gbar);
    
    BitVectorFree(state->D1);
    BitVectorFree(state->D2);
    BitMatrixFree(state->J);

    free(state);
}


// -------------------- D getters and setters ---------------------
int getD(struct StabilizerState* state, int i) {
    return 2*(BitVectorGet(state->D2, i)*2 + BitVectorGet(state->D1, i))
}

void setD(struct StabilizerState* state, int i, int val) {
    BitVectorSet(state->D1, i, (val/2) % 2);
    BitVectorSet(state->D2, i, (val/2 - (val/2)%2)/2 % 2);
}


void printStabilizerState(struct StabilizerState* state) {
    printf("state.n = %d\n", state->n);
    printf("state.k = %d\n", state->k);
    
    printf("state.h = np.array([");
    for (int i = 0; i<state->n; i++) {
        printf("%d", BitVectorGet(state->h, i));
        if (i+1 != state->n) printf(",");
    }
    printf("])\n");

    printf("state.G = np.array([");
    for (int i = 0; i<state->n; i++) {
        printf("[");
        for (int j = 0; j<state->n; j++) {
            printf("%d", BitMatrixGet(state->G, i, j));
            if (j+1 != state->n) printf(",");
        }
        printf("]");
        if (i+1 != state->n) printf(",");
    }
    printf("])\n");
   
    printf("state.Gbar = np.array([");
    for (int i = 0; i<state->n; i++) {
        printf("[");
        for (int j = 0; j<state->n; j++) {
            printf("%d", BitMatrixGet(state->Gbar, i, j));
            if (j+1 != state->n) printf(",");
        }
        printf("]");
        if (i+1 != state->n) printf(",");
    }
    printf("])\n");

    printf("state.Q = %d\n", state->Q);

    printf("state.D = np.array([");
    for (int i = 0; i<state->k; i++) {
        printf("%d", getD(state->D, i));
        if (i+1 != state->k) printf(",");
    }
    printf("])\n");

    if (state->k == 0) {
        printf("state.J = np.zeros((0,0))\n");
    } else {
        printf("state.J = np.array([");
        for (int i = 0; i<state->k; i++) {
            printf("[");
            for (int j = 0; j<state->k; j++) {
                printf("%d", 4*BitMatrixGet(state->J, i, j));
                if (j+1 != state->k) printf(",");
            }
            printf("]");
            if (i+1 != state->k) printf(",");
        }
        printf("])\n");
}

// --------------------------------- helper functions -------------------------

// helper to update D, J using equations 49, 50 on page 10
void updateDJ(struct StabilizerState* state, struct BitMatrix* R) {
    // equation 49
    BitMatrixMulVectorSet(R, state->D1);
    BitMatrixMulVectorSet(R, state->D2);

    struct BitMatrix* prod = BitMatrixMulMatrix(R, state->J);
    //TODO: finish 


    // equation 50
    //self.J = np.dot(np.dot(R, self.J), R.T) % 8 
    BitMatrixMulMatrixLeft(R, state->J);
    BitMatrixTranspose(R);
    BitMatrixMulMatrixRight(state->J, R);
}

// helper to update Q, D using equations 51, 52 on page 10
void updateQD(struct StabilizerState* state, struct BitVector* y);

// --------------------------------- exponential sum -------------------------------

// -- Helpers for evaluating equations like 63, 68. For even A,B only! --
// Evaluates 1 + e^{A*i*pi/4} + e^{A*i*pi/4} - e^{(A + B)*i*pi/4}
void Gamma(int* eps, int* p, int* m, int A, int B) {
    assert(A % 2 == 0 && B % 2 == 0); 

    int* lookupre = {1, 0, -1, 0};
    int* lookupim = {0, 1, 0, -1};

    Complex gamma = {1,0}

    // + e^{A*i*pi/4}
    gamma.re += lookupre[(A % 8)/2]
    gamma.im += lookupim[(A % 8)/2]

    // + e^{B*i*pi/4}
    gamma.re += lookupre[(B % 8)/2]
    gamma.im += lookupim[(B % 8)/2]

    // + e^{(A+B)*i*pi/4}
    gamma.re += lookupre[((A + B) % 8)/2]
    gamma.im += lookupim[((A + B) % 8)/2]

    if (gamma.re == 0 && gamma.im == 0) {
        *eps = 0;
        *p = 0;
        *m = 0;
        return;
    }

    *eps = 1;
    *p = 2;
   
    // lookup = {1: 0, 1+1j: 1, 1j: 2, -1: 4, -1j: 6, 1-1j: 7}
    if (gamma.re == 0) { // cases 1j, -1j
        if (gamma.im > 0) *m = 2;
        else *m = 6;
    } else { // cases 1, 1+1j, -1, 1-1j
        if (gamma.re == -1) *m = 4
        else {
            if (gamma.im == 1) *m = 1;
            if (gamma.im == 0) *m = 0;
            if (gamma.im == -1) *m = 7;
        }
    }
}

// Evaluates 1 + e^{A*i*pi/4}
void partialGamma(int *eps, int *p, int *m, int A) {
	assert(A % 2 == 0);
	
	//lookup = {0: (1, 2, 0), 2: (1, 1, 1), 4: (0, 0, 0), 6: (1, 1, 7)}
	//return lookup[A % 8]
	*eps = 1;
    if (A % 8 == 4) *eps = 0;

	switch(A % 8) {
		case 0:
			*p = 2;
			*m = 0;
			break;
		case 2:
			*p = 1;
			*m = 1;
			break;
		case 4:
			*p = 0;
			*m = 0;
			break;
		case 6:
			*p = 1;
			*m = 7;
			break;
	}
}

// helper for exponentialsum
void Wsigma(struct StabilizerState *state, int *eps, int *p, int *m, int sigma, int s, int *M, int Mlength, int *Dimers, int DimersLength) {
	if(state->k == 0){
        *eps = 1;
		*p = 0;
		*m = state->Q;
        return;
	}
	
	//W = (1, 0, self.Q + sigma*self.D[s])
	int tempEps = 1;
	int tempP = 0;
	int tempM = state->Q + sigma*(getD(state, s));
	for(int i=0;i<Mlength;i++){
		partialGamma(eps, p, m, getD(state, *(M+i)) + sigma*(4*BitMatrixGet(state->J, *(M + i), s)));
		if(*eps == 0){
			*p = 0;
			*m = 0;
			return;
		}
		tempEps = 1;
		tempP += *p;
		tempM = mod((tempM + *m), 8);
	}
	for(int i=0;i<DimersLength;i++){
		Gamma(eps, p, m, getD(state, *(Dimers + 2*i)) + sigma*4*BitMatrixGet(state->J, *(Dimers + 2*i), s),
			getD(state, *(Dimers + 2*i + 1)) + sigma*4*BitMatrixGet(state->J, *(Dimers + 2*i + 1), s));
		if(*eps == 0){
			*p = 0;
			*m = 0;
			return;
		}
		tempEps = 1;
		tempP += *p;
		tempM = mod((tempM + *m), 8);
	}
	
	*eps = tempEps;
	*p = tempP;
	*m = tempM;
}


void exponentialSumExact(struct StabilizerState* state, int* eps, int* p, int* m) {
 
	//define matrix R for later usage
	struct BitMatrix* R = newBitMatrixIdentity(state->n, state->n);
	
	//S = [a for a in range(self.k) if self.D[a] in [2, 6]]
	int *S;
	S = malloc(state->k * sizeof(int));
	//number of elements in S
	int Slength = 0;
	for(int a=0;a<state->k;a++){
		switch(getD(state, a)){
			case 2:
			case 6:
				*(S + Slength++) = a;
				break;
		}
	}

    if(Slength > 0){
		int a = *(S);
		
		//Construct R as in comment on page 12
		for(int i=1;i<Slength;i++){
			BitMatrixFlip(R, *(S+i), a);
		}

		updateDJ(state, R);
		
		//swap a and k, such that we only need to focus
		//on (k-1)*(k-1) submatrix of J later
	    BitMatrixSetIdentity(R);
		BitMatrixSwapCols(R, a, state->k - 1);

		updateDJ(state, R);

		*(S) = state->k - 1;
		Slength = 1;
	}

    //Now J[a, a] = 0 for all a not in S
	
	//E = [k for k in range(self.k) if k not in S]
	int *E;
	E = malloc(state->k * sizeof(int));
	//number of elements in E
	int Elength = 0;
	for(int k=0;k<state->k;k++){
		if(Slength==0 || k != *(S)){
			*(E + Elength++) = k;
		}
	}

	//tempE used later on to delete values from E
	int *tempE;
	tempE = malloc(state->k * sizeof(int));
	int tempElength = 0;


    int *M;
	M = malloc(state->k * sizeof(int));
	int Mlength = 0;
	int *Dimers;
	
    Dimers = malloc(2 * state->k * sizeof(int));
    //maintain list of dimers rather than r
	//each dimer is two consecutive numbers in the array
	//DimersLength is the number of dimers, not individual elements!
	
    int DimersLength = 0;
	int *K;
	K = malloc(state->k * sizeof(int));
	int Klength = 0;

    while(Elength > 0){
		int a = *(E);
		
		Klength = 0;
		//K = [b for b in E[1:] if self.J[a, b] == 4]
		for(int i=1;i<Elength;i++){
			if(BitMatrixGet(state->J, a, *(E + i)) == 1){
				*(K + Klength++) = *(E + i);
			}
		}
				
		if(Klength == 0){	//found a new monomer {a}
			*(M + Mlength++) = a;
			for(int i=0;i<Elength-1;i++){
				*(E + i) = *(E + i + 1);
			}
			Elength--;
		}
		else{
			int b = *(K);
			
			//Construct R for basis change
			BitMatrixSetIdentity(R);
			for(int i=0;i<Elength;i++){
				if(*(E + i) != a && *(E + i) != b){
					if(BitMatrixGet(state->J, a, *(E + i)) == 1) BitMatrixFlip(R, *(E + i), b);
					if(BitMatrixGet(state->J, b, *(E + i)) == 1) BitMatrixFlip(R, *(E + i), a);
				}
			}
			
			updateDJ(state, R);

			//{a, b} form a new dimer
			*(Dimers + 2*DimersLength) = a;
			*(Dimers + 2*DimersLength++ + 1) = b;
			
			//E = [x for x in E if x not in [a, b]]
			memcpy(tempE, E, Elength * sizeof(int));
			tempElength = Elength;
			Elength = 0;
			for(int i=0;i<tempElength;i++){
				if(*(tempE + i) != a && *(tempE + i) != b){
					*(E + Elength++) = *(tempE + i);
				}
			}
		}
	}

    BitMatrixFree(R);
    free(K);
    free(tempE);
    free(E);

	if(Slength == 0){
		//Compute W(K,q) from Eq. 63
		Wsigma(state, eps, p, m, 0, 0, M, Mlength, Dimers, DimersLength);
        free(Dimers);
        free(M);
        free(S);
		return;
	} else {
        int eps0, p0, m0, eps1, p1, m1;
        
        Wsigma(state, &eps0, &p0, &m0, 0, *(S), M, Mlength, Dimers, DimersLength);
        Wsigma(state, &eps1, &p1, &m1, 1, *(S), M, Mlength, Dimers, DimersLength);

        free(Dimers);
        free(M);
        free(S);
        
        if(eps0 == 0){
            *eps = eps1;
            *p = p1;
            *m = m1;
            return;
        }
        if(eps1 == 0){
            *eps = eps0;
            *p = p0;
            *m = m0;
            return;
        }
        
        //Now eps1 == eps0 == 1
        if(p0 != p1){
            printf("ExponentialSum: p0, p1 must be equal!\n");
            return;
        }
        if(mod((m1-m0), 2) == 1){
            printf("ExponentialSum: m1-m0 must be even!\n");
            return;
        }
        
        //Rearrange 2^{p0/2} e^{i pi m0/4} + 2^{p1/2} e^{i pi m1/4}
        //To 2^(p0/2) ( 1 + e^(i pi (m1-m0)/4)) and use partialGamma
        
        partialGamma(eps, p, m, m1-m0);
        if(*eps == 0){
            *p = 0;
            *m = 0;
        }
        else{
            *p += p0;
            *m = mod(*m + m0, 8);
        }
	}

}

// evaluates the expression in the comment on page 12
Complex evalW(int eps, int p, int m) {
    Complex z = ComplexPolar(1, M_PI * (double)m /4.);
    z = ComplexMulReal(z, eps * pow(2., (double)p /2.));
    return z;
}

Complex exponentialSum(struct StabilizerState* state) {
    int eps, p, m;
    exponentialSumExact(state, &eps, &p, &m);
    return evalW(eps, p, m);
}

// --------------------------------- shrink -------------------------------
int shrink(struct StabilizerState* state, gsl_vector* xi, int alpha, int lazy);

// --------------------------------- inner product -------------------------------
void innerProductExact(struct StabilizerState* state1, struct StabilizerState* state2, int* eps, int* p, int* m);
Complex innerProduct(struct StabilizerState* state1, struct StabilizerState* state2);

// --------------------------------- random state -------------------------------
//helper to generate a random double in a range
double randDouble(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

// helper to compute distribution given by equation 79 on page 15
double logeta(int d, int n){
	if(d == 0) return 0.;
	
	double product = 0;
	for(int a=1;a<=d;a++){
		product += log2(1 - pow(2, d - n - a));
		product -= log2(1 - pow(2, -a));
	}
	
	return (-d*(d+1)/2) + product;
}

struct StabilizerState* randomStabilizerState(int n);

// --------------------------------- measure Pauli -------------------------------
void extend(struct StabilizerState* state, struct BitVector* xi);

double measurePauli(struct StabilizerState* state, int m, struct BitVector* zeta, struct BitVector* xi);
