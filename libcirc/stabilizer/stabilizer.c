#include "stabilizer.h"
#include <math.h>
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
    state->G = newBitMatrixIdentity(n);
    state->Gbar = newBitMatrixIdentity(n);

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
    return 2*(BitVectorGet(state->D2, i)*2 + BitVectorGet(state->D1, i));
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
        printf("%d", getD(state, i));
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
}

// --------------------------------- helper functions -------------------------

// helper to update D, J using equations 49, 50 on page 10
void updateDJ(struct StabilizerState* state, struct BitMatrix* R) {
    
    // equation 49
	int *Dnew;
	Dnew = malloc(state->n * sizeof(int));
	for(int i=0;i<state->n;i++) {
        Dnew[i] = 0;
	    for(int j=0;j<state->k;j++) {
            Dnew[i] += getD(state, j) * BitMatrixGet(R, i, j);
        }
	}
	for(int i=0;i<state->n;i++) setD(state, i, Dnew[i]);
    free(Dnew);
    
    // D += R J R^T
    for (int a = 0; a < state->n; a++) {
        int val = 0;
        for (int b = 0; b < state->n; b++) {
            for (int c = 0; c < b; c++) {
                val += BitMatrixGet(state->J, b,c) * BitMatrixGet(R, a,b) * BitMatrixGet(R, a,c);
            }
        }
        if (val % 2 == 1) BitVectorFlip(state->D2, a);
    }

    // equation 50
    //self.J = np.dot(np.dot(R, self.J), R.T) % 8 
    BitMatrixMulMatrixLeft(R, state->J);
    struct BitMatrix* RT = BitMatrixTranspose(R);
    BitMatrixMulMatrixRight(state->J, RT);
    BitMatrixFree(RT);
}

// helper to update Q, D using equations 52, 53 on page 10
void updateQD(struct StabilizerState* state, struct BitVector* y) {
    // eqn 52
    for (int a = 0; a < state->k; a++) {
        state->Q += getD(state, a) * BitVectorGet(y, a);
        for (int b = a+1; b < state->k; b++) {
            state->Q += 4 * BitMatrixGet(state->J, a,b) * BitVectorGet(y, a) * BitVectorGet(y, b);
        }   
    }
    state->Q = state->Q % 8;

    // eqn 53
    struct BitVector* tempVector = BitMatrixMulVector(state->J, y);
    BitVectorXorSet(state->D2, tempVector);
    BitVectorFree(tempVector);
}

// --------------------------------- exponential sum -------------------------------

// -- Helpers for evaluating equations like 63, 68. For even A,B only! --
// Evaluates 1 + e^{A*i*pi/4} + e^{A*i*pi/4} - e^{(A + B)*i*pi/4}
void Gamma(int* eps, int* p, int* m, int A, int B) {
    assert(A % 2 == 0 && B % 2 == 0); 

    int lookupre[] = {1, 0, -1, 0};
    int lookupim[] = {0, 1, 0, -1};

    Complex gamma = {1,0};

    // + e^{A*i*pi/4}
    gamma.re += lookupre[(A % 8)/2];
    gamma.im += lookupim[(A % 8)/2];

    // + e^{B*i*pi/4}
    gamma.re += lookupre[(B % 8)/2];
    gamma.im += lookupim[(B % 8)/2];

    // + e^{(A+B)*i*pi/4}
    gamma.re -= lookupre[((A + B) % 8)/2];
    gamma.im -= lookupim[((A + B) % 8)/2];

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
        if (gamma.re/2 == -1) *m = 4;
        else {
            if (gamma.im/2 == 1) *m = 1;
            if (gamma.im/2 == 0) *m = 0;
            if (gamma.im/2 == -1) *m = 7;
        }
    }
}

// Evaluates 1 + e^{A*i*pi/4}
void partialGamma(int *eps, int *p, int *m, int A) {
    while (A < 0) A += 8;
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
		tempM =(tempM + *m) % 8;
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
		tempM = (tempM + *m) % 8;
	}
	
	*eps = tempEps;
	*p = tempP;
	*m = tempM;
}


void exponentialSumExact(struct StabilizerState* state, int* eps, int* p, int* m) {
 
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
		}
	}

	struct BitMatrix* R = newBitMatrixIdentity(state->n);

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
			tempElength = Elength;
			for(int i=0;i<tempElength;i++) *(tempE +i) = *(E + i);
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
        if((m1-m0) % 2 == 1){
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
            *m = *m + m0 % 8;
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

// 0 -> Empty
// 1 -> Same
// 2 -> Success
// --------------------------------- shrink -------------------------------
int shrink(struct StabilizerState* state, struct BitVector* xi, int alpha, int lazy) {
    //xi is of length n

	//S = [a for a in range(self.k) if np.inner(self.G[a], xi) % 2 == 1]
	int *S;
	S = malloc(state->k * sizeof(int));
	int Slength = 0;
	for(int a=0; a<state->k; a++) {
		struct BitVector* tempVector = BitMatrixGetRow(state->G, a);
		if(BitVectorInner(tempVector, xi) % 2 == 1) {
            *(S + Slength++) = a;
        }
        BitVectorFree(tempVector);
	}

	int beta = (alpha + BitVectorInner(xi, state->h)) % 2;

    if (Slength == 0){
        free(S);
        return 1-beta; // 0-> Empty, 1 -> Same
	}

    int i = *(S + --Slength);
	
    struct BitVector* gbar_i = BitMatrixGetRow(state->Gbar, i);
	for (int t=0;t<Slength;t++){
		int a = *(S + t);
	        
		//g^a <- g^a \oplus g^i
		//self.G[a] = (self.G[a] + self.G[i]) % 2
		struct BitVector* g_a = BitMatrixGetRow(state->G, a);
		struct BitVector* g_i = BitMatrixGetRow(state->G, i);
		BitVectorXorSet(g_a, g_i);
	    BitMatrixSetRow(state->G, g_a, a);

        BitVectorFree(g_a);
        BitVectorFree(g_i);

		//update D, J using equations 48, 49 on page 10
		//compute k*k basis change matrix R (equation 47)
		if (lazy != 1){
            struct BitMatrix* R = newBitMatrixIdentity(state->n);
			BitMatrixSet(R, a, i, 1);
			updateDJ(state, R);
            BitMatrixFree(R);
		}

		//gbar^i <- gbar^i + \sum_a gbar^a
		struct BitVector* gbar_a = BitMatrixGetRow(state->Gbar, a);
		BitVectorXorSet(gbar_i, gbar_a);
        BitVectorFree(gbar_a);
	}
	BitMatrixSetRow(state->Gbar, gbar_i, i);
    free(S);

	//swap g^i and g^k, gbar^i and gbar^k
	//remember elements are zero-indexed, so we use k-1
	BitMatrixSwapRows(state->G, i, state->k-1);
	BitMatrixSwapRows(state->Gbar, i, state->k-1);
	
	//update D, J using equations 48, 49 on page 10
	if(lazy != 1){
        struct BitMatrix* R = newBitMatrixIdentity(state->n);
		BitMatrixSwapRows(R, i, state->k-1);
		updateDJ(state, R);
        BitMatrixFree(R);
	}

    //h <- h \oplus beta*g^k
    if (beta == 1) {
        struct BitVector* tempVector = BitMatrixGetRow(state->G, state->k-1);
        BitVectorXorSet(state->h, tempVector);
        BitVectorFree(tempVector);
    }

	if(lazy != 1){
		//update Q, D using equations 51, 52 on page 10
		struct BitVector *y = newBitVector(state->n);
		BitVectorSet(y, state->k-1, beta);
		updateQD(state, y);
        BitVectorFree(y);
	}

	state->k--;
	return 2; // 2 -> Success
}


// --------------------------------- inner product -------------------------------
void innerProductExact(struct StabilizerState* state1, struct StabilizerState* state2, int* eps, int* p, int* m) {
    assert(state1->n == state2->n);
   
    struct StabilizerState *state = allocStabilizerState(state1->n, state1->k);
    deepCopyState(state, state1);
    
    for(int b=state2->k; b<state2->n; b++){
		struct BitVector* tempVector = BitMatrixGetRow(state2->Gbar, b);
            
		int alpha = BitVectorInner(state2->h, tempVector) % 2;
		
        *eps = shrink(state, tempVector, alpha, 0);
		if(*eps == 0){
			*p = 0;
			*m = 0;
            return;
		}

        BitVectorFree(tempVector);
	}

	//Now K = K_1 \cap K_2
    struct BitMatrix* R = newBitMatrixZero(state->n, state->n);

	struct BitVector* h_plus_h2 = newBitVector(state->n);
    BitVectorCopy(h_plus_h2, state->h);
    BitVectorXorSet(h_plus_h2, state2->h);

    struct BitVector* y = newBitVector(state->n);

	for (int a=0; a < state2->n; a++) {
		struct BitVector* G2bar_a = BitMatrixGetRow(state2->Gbar, a);
		BitVectorSet(y, a, BitVectorInner(h_plus_h2, G2bar_a));
       
        for (int b = 0; b < state2->n; b++) {
		    struct BitVector* G_b = BitMatrixGetRow(state->G, b);
            BitMatrixSet(R, a, b, BitVectorInner(G_b, G2bar_a)) ;
            BitVectorFree(G_b);
        }
        
        BitVectorFree(G2bar_a);
	}

    BitVectorFree(h_plus_h2);

    struct StabilizerState *state2temp = allocStabilizerState(state2->n, state2->k);
    deepCopyState(state2temp, state2);

    updateQD(state2temp, y);
    updateDJ(state2temp, R);

    BitVectorFree(y);
    BitMatrixFree(R);

    state->Q -= state2temp->Q;
    if (state->Q < 0) state->Q += 8;
    
    for (int i = 0; i < state->n; i++) {
        int val = getD(state, i) - getD(state2temp, i);
        if (val < 0) val += 8;
        setD(state, i, val);
    }

    BitMatrixXorSet(state->J, state2temp->J);
    
    exponentialSumExact(state, eps, p, m);
    *p -= state1->k + state2->k;

    freeStabilizerState(state);
    freeStabilizerState(state2temp);
}

Complex innerProduct(struct StabilizerState* state1, struct StabilizerState* state2) {
    int eps, p, m;
    innerProductExact(state1, state2, &eps, &p, &m);
    return evalW(eps, p, m);
}

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

struct StabilizerState* randomStabilizerState(int n) {
    assert(n >= 1);
	
	int i, j, d;
	double *dist;
	dist = malloc((n+1)*sizeof(double));
	double sum = 0;
	for(d=0;d<=n;d++){
		*(dist + d) = pow(2, logeta(d, n));
		sum += *(dist + d);
	}
	//normalize
	for(d=0;d<=n;d++){
		*(dist + d) /= sum;
	}
	
	double *cumulative;
	cumulative = calloc((n+1), sizeof(double));
	for(i=0;i<=n;i++){
		for(d=0;d<=i;d++){
			*(cumulative + i) += *(dist + d);
		}
	}

	//normalize
	for(i=0;i<=n;i++){
		*(cumulative + i) /= *(cumulative + n);
	}
	
	//sample d from distribution
	double sample = randDouble(0., 1.);
	while(sample == 0.){
		sample = randDouble(0., 1.);
	}
	
	for(d=0;d<=n;d++){
		if(sample <= *(cumulative + d)){
			break;
		}
	}
	int k = n - d;

    free(dist);
    free(cumulative);

    struct StabilizerState* state = allocStabilizerState(n, n);
	
    while (state->k > k){
	    struct BitVector* tempVector = newBitVectorRandom(n);
		shrink(state, tempVector, 0, 1);
        BitVectorFree(tempVector);
	}
    // Now K is a random k-dimensional subspace

    BitVectorSetRandom(state->h);
    BitVectorSetRandom(state->D1);
    BitVectorSetRandom(state->D2);

	for(i=0;i<k;i++){
	    BitMatrixSet(state->J, i, i, (unsigned int)(((getD(state, i)*2) % 8)/4));
		for(j=0;j<i;j++){
            unsigned int val = rand();
			BitMatrixSet(state->J, i, j, val);
			BitMatrixSet(state->J, j, i, val);
		}
	}
    return state;
}

// --------------------------------- measure Pauli -------------------------------
void extend(struct StabilizerState* state, struct BitVector* xi) {
	//S = [a for a in range(self.n) if np.dot(xi, self.Gbar[a]) % 2 == 1]
	int *S;
	S = malloc(state->n * sizeof(int));
	int Slength = 0;
	for(int a=0;a<state->n;a++){
        struct BitVector* GBar_a = BitMatrixGetRow(state->Gbar, a);
		if(BitVectorInner(xi, GBar_a) % 2 == 1) *(S + Slength++) = a;
        BitVectorFree(GBar_a);
	}
	
	//T = [a for a in S if self.k <= a and self.k < self.n]
	int *T;
	T = malloc(Slength * sizeof(int));
	int Tlength = 0;
	for(int i=0;i<Slength;i++){
		if(state->k <= *(S + i) && state->k < state->n){
			*(T + Tlength++) = *(S + i);
		}
	}
	
	if(Tlength == 0) {
        free(S);
        free(T);
		return;	//xi in L(K)
	}
	
	int i = *T;
	
	//S = [a for a in S if a != i]
	int *newS;
	newS = malloc(Slength * sizeof(int));
	int newSlength = 0;
	for(int j=0;j<Slength;j++){
		if(*(S + j) != i){
			*(newS + newSlength++) = *(S + j);
		}
	}
	
	int a;
	struct BitVector* GBar_i = BitMatrixGetRow(state->Gbar, i);
	struct BitVector* G_i = BitMatrixGetRow(state->G, i);
	for(int j=0;j<newSlength;j++){
		a = *(newS + j);
		struct BitVector* GBar_a = BitMatrixGetRow(state->Gbar, a);
        BitVectorXorSet(GBar_a, GBar_i);
		BitMatrixSetRow(state->Gbar, GBar_a, a);
        BitVectorFree(GBar_a);
	
        struct BitVector* G_a = BitMatrixGetRow(state->G, a);
        BitVectorXorSet(G_i, G_a);
        BitVectorFree(G_a);
	}
	BitMatrixSetRow(state->G, G_i, i);
    BitVectorFree(GBar_i);
    BitVectorFree(G_i);
	
	//Swap g^i and g^k (not g^{k+1} because zero indexing)
    BitMatrixSwapRows(state->G, i, state->k);
	BitMatrixSwapRows(state->Gbar, i, state->k);
	
	state->k++;

    free(S);
    free(T);
    free(newS);
}

double measurePauli(struct StabilizerState* state, int m, struct BitVector* zeta, struct BitVector* xi) {
    assert(state->n == (int)zeta->size);
    assert(state->n == (int)xi->size);


	//write zeta, xi in basis of K
    struct BitVector* vecXi = newBitVector(state->n);
    struct BitVector* vecZeta = newBitVector(state->n);

	for(int a=0; a<state->k; a++){
        struct BitVector* Gbar_a = BitMatrixGetRow(state->Gbar, a);
        BitVectorSet(vecXi, a, BitVectorInner(Gbar_a, xi));
        BitVectorFree(Gbar_a);

        struct BitVector* G_a = BitMatrixGetRow(state->G, a);
        BitVectorSet(vecZeta, a, BitVectorInner(G_a, zeta));
        BitVectorFree(G_a);
	}

    struct BitVector* xiPrime = newBitVector(state->n);


	for(int a=0;a<state->k;a++){
        if (BitVectorGet(vecXi, a)) {
            struct BitVector* G_a = BitMatrixGetRow(state->G, a);
            BitVectorXorSet(xiPrime, G_a);
            BitVectorFree(G_a);
        }
	}
	
	//compute w in {0, 2, 4, 6} using eq. 88
	int w = 2*m;
	w += 4*(BitVectorInner(zeta, state->h) % 2);
	w += 2*BitVectorInner(state->D1, vecXi);
	w += 4*BitVectorInner(state->D2, vecXi);

	for(int b=0; b<state->k; b++){
		for(int a=0;a<b;a++){
			w += 4*BitMatrixGet(state->J, a, b)*BitVectorGet(vecXi, a)*BitVectorGet(vecXi, b);
		}
	}
	w = w % 8;

	//Compute eta_0, ..., eta_{k-1} using eq. 94
    struct BitVector* eta = BitMatrixMulVector(state->J, vecXi);
    BitVectorXorSet(eta, vecZeta);

	if(BitVectorSame(xi, xiPrime)){
		if(w==0 || w==4){
            struct BitVector* gamma = newBitVector(state->n);
			
			for(int a=0; a<state->k; a++){
                if (BitVectorGet(eta, a) == 1) {
                    struct BitVector* Gbar_a = BitMatrixGetRow(state->Gbar, a);
                    BitVectorXorSet(gamma, Gbar_a);
                    BitVectorFree(Gbar_a);
                }
			}
			
			int omegaPrime = w/4;
			int alpha = omegaPrime + BitVectorInner(gamma, state->h);
            alpha = alpha % 2;

			int eps = shrink(state, gamma, alpha, 0);

            BitVectorFree(vecZeta);
            BitVectorFree(vecXi);
            BitVectorFree(xiPrime);
            BitVectorFree(eta);
            BitVectorFree(gamma);

			switch(eps){
				case 0:
					return 0;
					break;
				case 1:
					return 1;
					break;
				case 2:
					return pow(2, -0.5);
					break;
			}
		}
		else if(w==2 || w==6){
			int sigma = 2 - w/2;
			//Update Q, D, J using equations 100, 101
			state->Q = (state->Q + sigma) % 8;
			for(int a=0;a<state->k;a++) {
                int val = getD(state, a) - 2*sigma*BitVectorGet(eta, a);
                while (val < 0) {
                    val += 8;
                }
				setD(state, a, val);
			}
			
			//ignore a != b for some reason, MATLAB code does it too
			//still satisfies J[a,a] = 2 D[a] mod 8
			for(int i=0;i<state->n;i++){
				for(int j=0;j<state->n;j++){
                    if (BitVectorGet(eta, i) * BitVectorGet(eta, j) == 1) BitMatrixFlip(state->J, i, j);
				}
			}

            BitVectorFree(vecZeta);
            BitVectorFree(vecXi);
            BitVectorFree(xiPrime);
            BitVectorFree(eta);
			return pow(2, -0.5);
		}
	}
	
	//remaining case: xiPrime != xi
	extend(state, xi);
	
	//update D
    int newDval = 2*m + 4*(BitVectorInner(zeta, xi) % 2) + 4*(BitVectorInner(zeta, state->h) % 2);
    setD(state, state->k-1, newDval);
	
	//update J 
	BitMatrixSetRow(state->J, vecZeta, state->k-1);
	BitMatrixSetCol(state->J, vecZeta, state->k-1);
	BitMatrixSet(state->J, state->k-1, state->k-1, m);
    
    BitVectorFree(vecZeta);
    BitVectorFree(vecXi);
    BitVectorFree(xiPrime);
    BitVectorFree(eta);

	return pow(2, -0.5);
}
