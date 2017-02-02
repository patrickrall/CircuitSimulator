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
#include "stabilizer.h"

//define modulo function
//needed because the % operator is for remainder, not modulo
//those two have a difference when it comes to negative numbers
//http://stackoverflow.com/questions/11720656/modulo-operation-with-negative-numbers
int mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

//helper to generate a random double in a range
double randDouble(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}


struct StabilizerState* allocStabilizerState(int n, int k) {
	struct StabilizerState *state = (struct StabilizerState *)malloc(sizeof(struct StabilizerState));

    state->n = n;
    state->k = k;

    state->h = gsl_vector_alloc(n);
    gsl_vector_set_zero(state->h);
    state->G = gsl_matrix_alloc(n, n);
    gsl_matrix_set_identity(state->G);
    state->Gbar = gsl_matrix_alloc(n, n);
    gsl_matrix_set_identity(state->Gbar);

	state->Q = 0;
	state->D = gsl_vector_alloc(n);
    gsl_vector_set_zero(state->D);
	state->J = gsl_matrix_alloc(n, n);
    gsl_matrix_set_zero(state->J);

    return state;
}


void deepCopyState(struct StabilizerState *dest, struct StabilizerState *src){
	dest->n = src->n;
	dest->k = src->k;
	gsl_vector_memcpy(dest->h, src->h);
	gsl_matrix_memcpy(dest->G, src->G);
	gsl_matrix_memcpy(dest->Gbar, src->Gbar);
	dest->Q = src->Q;
	gsl_vector_memcpy(dest->D, src->D);
	gsl_matrix_memcpy(dest->J, src->J);
}

//helper to update D, J using equations 48, 49 on page 10		
void updateDJ(struct StabilizerState *state, gsl_matrix *R){
	
	//temporary variables for storing intermediary results
	gsl_vector *tempVector, *tempVector1;
	gsl_matrix *tempMatrix;
	tempVector = gsl_vector_alloc(state->n);
	tempVector1 = gsl_vector_alloc(state->n);
	tempMatrix = gsl_matrix_alloc(state->n, state->n);
	
	//equation 48
	//tempVector <- RD
	gsl_blas_dgemv(CblasNoTrans, 1., R, state->D, 0., tempVector);
	//D <- tempVector
	gsl_vector_memcpy(state->D, tempVector);
	//TODO: convert loops into matrix form
	for(int b=0;b<state->k;b++){
		for(int c=0;c<b;c++){
			//tempVector <- R[:,b]
			gsl_matrix_get_col(tempVector, R, b);
			//tempVector1 <- R[:,c]
			gsl_matrix_get_col(tempVector1, R, c);
			//tempVector <- tempVector .* tempVector1
			gsl_vector_mul(tempVector, tempVector1);
			//tempVector <- tempVector * J[b,c]
			gsl_vector_scale(tempVector, gsl_matrix_get(state->J,b,c));
			//D <- tempVector
			gsl_vector_add(state->D, tempVector);
		}
	}
	//D = D % 8
	//TODO: find a better way to % 8 a vector or matrix
	for(int i=0;i<state->k;i++){
		gsl_vector_set(state->D, i, mod((int)gsl_vector_get(state->D, i), 8));
	}
	
	//equation 49
	//tempMatrix <- R * J
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, R, state->J, 0, tempMatrix);
	//J <- tempMatrix * R'
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, tempMatrix, R, 0, state->J);
	//J = J % 8
	for(int i=0;i<state->k;i++){
		for(int j=0;j<state->k;j++){
			gsl_matrix_set(state->J, i, j, mod((int)gsl_matrix_get(state->J, i, j), 8));
		}
	}
	
	//free memory
	gsl_vector_free(tempVector);
	gsl_vector_free(tempVector1);
	gsl_matrix_free(tempMatrix);
}

//helper to update Q, D using equations 51, 52 on page 10
void updateQD(struct StabilizerState *state, gsl_vector *y){
	
	
	//temporary variables for storing intermediary results
	double tempInt;
	gsl_vector *tempVector;
	tempVector = gsl_vector_alloc(state->n);
	
	//equation 51
	//tempInt <- D dot y
	gsl_blas_ddot(state->D, y, &tempInt);
	state->Q += (int)tempInt;
	for(int a=0;a<state->k;a++){
		for(int b=0;b<a;b++){
			//Q += J[a,b]*y[a]*y[b]
			//TODO: replace loops with matrix operations
			state->Q += gsl_matrix_get(state->J,a,b)*gsl_vector_get(y,a)*gsl_vector_get(y,b);
		}
	}
	state->Q = mod(state->Q, 8);
	
	//equation 52
	//D_a += J[a,:] dot y
	for(int a=0;a<state->k;a++){
		gsl_matrix_get_row(tempVector, state->J, a);
		gsl_blas_ddot(tempVector, y, &tempInt);
		gsl_vector_set(state->D, a, mod((int)(gsl_vector_get(state->D, a) + tempInt), 8));
	}
	
	//free memory
	gsl_vector_free(tempVector);
}

//helper that evaluates the expression in the comment on page 12
void evalW(gsl_complex *ans, int eps, int p, int m){
	//imaginary unit
	gsl_complex eye = gsl_complex_rect(0,1);
	
	*ans = gsl_complex_mul_real(eye, M_PI*m/4);
	*ans = gsl_complex_exp(*ans);
	//*ans = gsl_complex_exp(gsl_complex_mul_real(eye, M_PI*m/4));
	*ans = gsl_complex_mul_real(*ans, eps*pow(2,p/2.));
}

//Helpers for evaluating equations like 63, 68. For even A,B only!

//Evaluates 1 + e^{A*i*pi/4} + e^{A*i*pi/4} - e^{(A + B)*i*pi/4}
void Gamma(int *eps, int *p, int *m, int A, int B){
	if(mod(A,2)==1 || mod(B,2)==1){
		printf("Gamma: A and B must be even!\n");
		return;
	}
	
	//lookup = {0: 1, 2: 1j, 4: -1, 6: -1j}
	gsl_vector_complex *lookup;
	lookup = gsl_vector_complex_alloc(8);
	gsl_vector_complex_set(lookup, 0, gsl_complex_rect(1,0));
	gsl_vector_complex_set(lookup, 2, gsl_complex_rect(0,1));
	gsl_vector_complex_set(lookup, 4, gsl_complex_rect(-1,0));
	gsl_vector_complex_set(lookup, 6, gsl_complex_rect(0,-1));
	
	//gamma = 1 + 0j + lookup[A % 8] + lookup[B % 8] - lookup[(A + B) % 8]
	gsl_complex gamma = gsl_complex_add(gsl_vector_complex_get(lookup, mod(A,8)), gsl_vector_complex_get(lookup, mod(B,8)));
	gamma = gsl_complex_sub(gamma, gsl_vector_complex_get(lookup, mod(A+B,8)));
	gamma = gsl_complex_add_real(gamma, 1);
	
	if(GSL_REAL(gamma) == 0 && GSL_IMAG(gamma) == 0){
		*eps = 0;
		*p = 0;
		*m = 0;
	}
	else{
		//lookup = {1: 0, 1+1j: 1, 1j: 2, -1: 4, -1j: 6, 1-1j: 7}
		//return(1, 2, lookup[gamma/2])
		*eps = 1;
		*p = 2;
		gamma = gsl_complex_div_real(gamma,2);
		if(GSL_REAL(gamma) == 1 && GSL_IMAG(gamma) == 0){
			*m = 0;
		}
		else if(GSL_REAL(gamma) == 1 && GSL_IMAG(gamma) == 1){
			*m = 1;
		}
		if(GSL_REAL(gamma) == 0 && GSL_IMAG(gamma) == 0){
			*m = 2;
		}
		if(GSL_REAL(gamma) == -1 && GSL_IMAG(gamma) == 0){
			*m = 4;
		}
		if(GSL_REAL(gamma) == 0 && GSL_IMAG(gamma) == -1){
			*m = 6;
		}
		if(GSL_REAL(gamma) == 1 && GSL_IMAG(gamma) == -1){
			*m = 7;
		}
	}

    gsl_vector_complex_free(lookup);
}

//Evaluates 1 + e^{A*i*pi/4}
void partialGamma(int *eps, int *p, int *m, int A){
	if(mod(A,2)==1){
		printf("partialGamma: A must be even!\n");
		return;
	}
	
	//lookup = {0: (1, 2, 0), 2: (1, 1, 1), 4: (0, 0, 0), 6: (1, 1, 7)}
	//return lookup[A % 8]
	switch(mod(A,8)){
		case 0:
			*eps = 1;
			*p = 2;
			*m = 0;
			break;
		case 2:
			*eps = 1;
			*p = 1;
			*m = 1;
			break;
		case 4:
			*eps = 0;
			*p = 0;
			*m = 0;
			break;
		case 6:
			*eps = 1;
			*p = 1;
			*m = 7;
			break;
	}
}

void Wsigma(struct StabilizerState *state, int *eps, int *p, int *m, gsl_complex *ans, 
	int exact, int sigma, int s, int *M, int Mlength, int *Dimers, int DimersLength){
	
	if(state->k == 0){
        *eps = 1;
		*p = 0;
		*m = state->Q;

		if(exact == 0) {
			evalW(ans, 1, 0, state->Q);
		}
        return;
	}
	
	//W = (1, 0, self.Q + sigma*self.D[s])
	int tempEps = 1;
	int tempP = 0;
	int tempM = state->Q + sigma*((int)gsl_vector_get(state->D, s));
	for(int i=0;i<Mlength;i++){
		partialGamma(eps, p, m, gsl_vector_get(state->D, *(M+i)) + sigma*((int)gsl_matrix_get(state->J, *(M + i), s)));
		if(*eps == 0){
			*p = 0;
			*m = 0;
			*ans = gsl_complex_rect(0, 0);
			return;
		}
		tempEps = 1;
		tempP += *p;
		tempM = mod((tempM + *m), 8);
	}
	for(int i=0;i<DimersLength;i++){
		Gamma(eps, p, m, (int)gsl_vector_get(state->D, *(Dimers + 2*i)) + sigma*((int)gsl_matrix_get(state->J, *(Dimers + 2*i), s)),
			(int)gsl_vector_get(state->D, *(Dimers + 2*i + 1)) + sigma*((int)gsl_matrix_get(state->J, *(Dimers + 2*i + 1), s)));
		if(*eps == 0){
			*p = 0;
			*m = 0;
			*ans = gsl_complex_rect(0, 0);
			return;
		}
		tempEps = 1;
		tempP += *p;
		tempM = mod((tempM + *m), 8);
	}
	
	*eps = tempEps;
	*p = tempP;
	*m = tempM;
	if(exact == 0){ 
		evalW(ans, *eps, *p, *m);
	}
}

//Helper required for InnerProduct and MeasurePauli.
//Depends only on Q, D, J. Manipulates integers p, m, eps
//to avoid rounding error then evaluates to a real number.
void exponentialSum(struct StabilizerState *state, int *eps, int *p, int *m, gsl_complex *ans, int exact){

	//define matrix R for later usage
	gsl_matrix *R;
	R = gsl_matrix_alloc(state->n, state->n);
	
	//S = [a for a in range(self.k) if self.D[a] in [2, 6]]
	int *S;
	S = malloc(state->k * sizeof(int));
	//number of elements in S
	int Slength = 0;
	for(int a=0;a<state->k;a++){
		switch((int)gsl_vector_get(state->D, a)){
			case 2:
			case 6:
				*(S + Slength++) = a;
				break;
		}
	}
	
	if(Slength > 0){
		int a = *(S);
		
		//Construct R as in comment on page 12
		gsl_matrix_set_identity(R);
		for(int i=1;i<Slength;i++){
			gsl_matrix_set(R, *(S+i), a, mod(((int)gsl_matrix_get(R, *(S+i), a) + 1), 2));
		}

		updateDJ(state, R);
		
		//swap a and k, such that we only need to focus
		//on (k-1)*(k-1) submatrix of J later
		gsl_matrix_set_identity(R);
		gsl_matrix_swap_columns(R, a, state->k - 1);

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
	Dimers = malloc(2 * state->k * sizeof(int));	//maintain list of dimers rather than r
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
			if((int)gsl_matrix_get(state->J, a, *(E + i)) == 4){
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
			gsl_matrix_set_identity(R);
			for(int i=0;i<Elength;i++){
				if(*(E + i) != a && *(E + i) != b){
					if((int)gsl_matrix_get(state->J, a, *(E + i)) == 4){
						gsl_matrix_set(R, *(E + i), b, mod(((int)gsl_matrix_get(R, *(E + i), b) + 1), 2));
					}
					if((int)gsl_matrix_get(state->J, b, *(E + i)) == 4){
						gsl_matrix_set(R, *(E + i), a, mod(((int)gsl_matrix_get(R, *(E + i), a) + 1), 2));
					}
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

    gsl_matrix_free(R);
    free(K);
    free(tempE);
    free(E);

	if(Slength == 0){
		//Compute W(K,q) from Eq. 63
		Wsigma(state, eps, p, m, ans, exact, 0, 0, M, Mlength, Dimers, DimersLength);
        free(Dimers);
        free(M);
        free(S);
		return;
	}
	else{
		//Compute W_0, W_1 from Eq. 68
		if(exact == 0){
			//return Wsigma(0, s) + Wsigma(1, s)
			Wsigma(state, eps, p, m, ans, exact, 0, *(S), M, Mlength, Dimers, DimersLength);
			gsl_complex tempAns = gsl_complex_rect(GSL_REAL(*ans), GSL_IMAG(*ans));
			Wsigma(state, eps, p, m, ans, exact, 1, *(S), M, Mlength, Dimers, DimersLength);
			*ans = gsl_complex_add(*ans, tempAns);
            free(Dimers);
            free(M);

            free(S);
			return;
		}
		else{
			int eps0, p0, m0, eps1, p1, m1;
			Wsigma(state, &eps0, &p0, &m0, ans, exact, 0, *(S), M, Mlength, Dimers, DimersLength);
			Wsigma(state, &eps1, &p1, &m1, ans, exact, 1, *(S), M, Mlength, Dimers, DimersLength);
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
}

//possible outputs of function shrink:
//EMPTY == 0
//SAME == 1
//SUCCESS = 2
int shrink(struct StabilizerState *state, gsl_vector *xi, int alpha, int lazy){
	//xi is of length n
	
	int a;
	double tempInt;
	gsl_vector *tempVector, *tempVector1;
	tempVector = gsl_vector_alloc(state->n);
	tempVector1 = gsl_vector_alloc(state->n);
	gsl_matrix *R;
	R = gsl_matrix_alloc(state->n, state->n);
	
	//S = [a for a in range(self.k) if np.inner(self.G[a], xi) % 2 == 1]
	int *S;
	S = malloc(state->k * sizeof(int));
	int Slength = 0;
	for(a=0;a<state->k;a++){
		gsl_matrix_get_row(tempVector, state->G, a);
		gsl_blas_ddot(tempVector, xi, &tempInt);
		if(mod((int)tempInt, 2) == 1){
			*(S + Slength++) = a;
		}
	}
	
	gsl_blas_ddot(xi, state->h, &tempInt);
	int beta = mod(alpha + (int)tempInt, 2);
	
	if(Slength == 0){
		if(beta == 1){
            gsl_vector_free(tempVector);
            gsl_vector_free(tempVector1);
            gsl_matrix_free(R);
            free(S);
			return 0;
		}
		if(beta == 0){
            gsl_vector_free(tempVector);
            gsl_vector_free(tempVector1);
            gsl_matrix_free(R);
            free(S);
			return 1;
		}
	}
	
	int i = *(S + --Slength);
	
	for(int t=0;t<Slength;t++){
		a = *(S + t);
		
		//g^a <- g^a \oplus g^i
		//self.G[a] = (self.G[a] + self.G[i]) % 2
		gsl_matrix_get_row(tempVector, state->G, a);
		gsl_matrix_get_row(tempVector1, state->G, i);
		gsl_vector_add(tempVector, tempVector1);
		for(int t1=0;t1<state->k;t1++){
			gsl_vector_set(tempVector, t1, mod((int)gsl_vector_get(tempVector, t1), 2));
		}
		gsl_matrix_set_row(state->G, a, tempVector);
		
		//update D, J using equations 48, 49 on page 10
		//compute k*k basis change matrix R (equation 47)
		if(lazy != 1){
			gsl_matrix_set_identity(R);
			gsl_matrix_set(R, a, i, 1);
			
			updateDJ(state, R);
		}
		
		//gbar^i <- gbar^i + \sum_a gbar^a
		gsl_matrix_get_row(tempVector, state->Gbar, i);
		gsl_matrix_get_row(tempVector1, state->Gbar, a);
		gsl_vector_add(tempVector, tempVector1);
		gsl_matrix_set_row(state->Gbar, i, tempVector);
	}
	
	//self.Gbar = self.Gbar % 2
	gsl_matrix_get_row(tempVector, state->Gbar, i);
	for(int t1=0;t1<state->n;t1++){
		gsl_vector_set(tempVector, t1, mod((int)gsl_vector_get(tempVector, t1), 2));
	}
	gsl_matrix_set_row(state->Gbar, i, tempVector);
	
	//swap g^i and g^k, gbar^i and gbar^k
	//remember elements are zero-indexed, so we use k-1
	gsl_matrix_swap_rows(state->G, i, state->k-1);
	gsl_matrix_swap_rows(state->Gbar, i, state->k-1);
	
	//update D, J using equations 48, 49 on page 10
	if(lazy != 1){
		gsl_matrix_set_identity(R);
		gsl_matrix_swap_rows(R, i, state->k-1);
		updateDJ(state, R);
	}
	
	//h <- h \oplus beta*g^k
	gsl_matrix_get_row(tempVector, state->G, state->k-1);
	gsl_vector_scale(tempVector, beta);
	gsl_vector_add(tempVector, state->h);
	for(int t1=0;t1<state->n;t1++){
		gsl_vector_set(tempVector, t1, mod((int)gsl_vector_get(tempVector, t1), 2));
	}
	gsl_vector_memcpy(state->h, tempVector);
	
	if(lazy != 1){
		//update Q, D using equations 51, 52 on page 10
		gsl_vector *y;
		y = gsl_vector_calloc(state->n);
		gsl_vector_set(y, state->k-1, beta);
		updateQD(state, y);
        gsl_vector_free(y);
		
		//remove last row and column from J
		//gsl_matrix_view newJ = gsl_matrix_submatrix(state->J, 0, 0, state->k-1, state->k-1);
		//gsl_matrix_free(state->J);
		//state->J = &newJ.matrix;
		for(int i=state->k-1;i<state->n;i++){
			for(int j=state->k-1;j<state->n;j++){
				gsl_matrix_set(state->J, i, j, 0);
			}
			gsl_vector_set(state->D, i, 0);
		}
		
		//remove last element from D
		//gsl_vector_view newD = gsl_vector_subvector(state->D, 0, state->k-1);
		//gsl_vector_free(state->D);
		//state->D = &newD.vector;)
	}
	
    gsl_vector_free(tempVector);
    gsl_vector_free(tempVector1);
    gsl_matrix_free(R);
    free(S);
	state->k--;
	
	return 2;
}

void innerProduct(struct StabilizerState *state1, struct StabilizerState *state2, int *eps, int *p, int *m, gsl_complex *ans, int exact){
	if(state1->n != state2->n){
		printf("innerProduct: States do not have same dimension.\n");
		return;
	}
	
	int i, j, b, alpha;
	double tempInt;
	gsl_vector *tempVector, *tempVector1;
	tempVector = gsl_vector_alloc(state2->n);
	tempVector1 = gsl_vector_alloc(state2->n);
	
	//K <- K_1, (also copy q_1)
    struct StabilizerState *state = allocStabilizerState(state1->n, state1->k);
	deepCopyState(state, state1);

	for(b=state2->k;b<state2->n;b++){
		gsl_matrix_get_row(tempVector, state2->Gbar, b);
		gsl_blas_ddot(state2->h, tempVector, &tempInt);
		alpha = mod((int)tempInt, 2);
		*eps = shrink(state, tempVector, alpha, 0);
		if(*eps == 0){
			*eps = 0;
			*p = 0;
			*m = 0;
			*ans = gsl_complex_rect(0,0);
            return;
		}
	}
	
	//Now K = K_1 \cap K_2
	gsl_vector *y;
	y = gsl_vector_alloc(state2->n);
	gsl_vector_memcpy(tempVector, state->h);
	gsl_vector_add(tempVector, state2->h);
	for(i=0;i<state2->n;i++){
		gsl_matrix_get_row(tempVector1, state2->Gbar, i);
		gsl_blas_ddot(tempVector, tempVector1, &tempInt);
		gsl_vector_set(y, i, mod((int)tempInt, 2));
	}
	
	gsl_matrix *smallR, *R, *Rtemp;
	smallR = gsl_matrix_calloc(state->k, state2->k);
	gsl_vector *smallRrow;
	smallRrow = gsl_vector_alloc(state2->k);
	Rtemp = gsl_matrix_alloc(state->k+1, state2->k);
	gsl_matrix_view Gk = gsl_matrix_submatrix(state->G, 0, 0, state->k, state->n);
	gsl_matrix_view Gk2 = gsl_matrix_submatrix(state2->Gbar, 0, 0, state2->k, state2->n);

    if (state->k>0) {
        gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, &Gk.matrix, &Gk2.matrix, 0, smallR);
        for(i=0;i<state->k;i++){
            gsl_matrix_get_row(smallRrow, smallR, i);
            gsl_matrix_set_row(Rtemp, i, smallRrow);
        }
    }

	//TODO: more efficient modulo of a matrix..
	R = gsl_matrix_alloc(state->n, state->n);
	gsl_matrix_set_zero(R);	//maybe identity?
	for(i=0;i<state->k+1;i++){
		for(j=0;j<state2->k;j++){
			gsl_matrix_set(R, i, j, mod((int)gsl_matrix_get(Rtemp, i, j), 2));
		}
	}
	
    struct StabilizerState *state2temp = allocStabilizerState(state2->n, state2->k);
	deepCopyState(state2temp, state2);

    updateQD(state2temp, y);
	updateDJ(state2temp, R);
    gsl_vector_free(y);
    gsl_matrix_free(R);
    gsl_matrix_free(Rtemp);
    gsl_matrix_free(smallR);
    gsl_vector_free(smallRrow);
    gsl_vector_free(tempVector);
    gsl_vector_free(tempVector1);

	//now q, q2 are defined in the same basis
	state->Q = state->Q - mod(state2temp->Q, 8);
	for(i=0;i<state->k;i++){
		gsl_vector_set(state->D, i, mod((int)gsl_vector_get(state->D, i) - (int)gsl_vector_get(state2temp->D, i), 8));
		for(j=0;j<state->k;j++){
			gsl_matrix_set(state->J, i, j, mod((int)gsl_matrix_get(state->J, i, j) - (int)gsl_matrix_get(state2temp->J, i, j), 8));
		}
	}	


    if(exact == 0){
        //printStabilizerState(state);
		exponentialSum(state, eps, p, m, ans, 0);
        //printStabilizerState(state);
		//exponentialSum(state, eps, p, m, ans, 0);
		*ans = gsl_complex_mul_real(*ans, pow(2, -((double)state1->k + (double)state2temp->k)/2));
	}
	else{
		exponentialSum(state, eps, p, m, ans, 1);
		*p -= state1->k + state2temp->k;
	}
    freeStabilizerState(state);
    freeStabilizerState(state2temp);
}

//helper to compute distribution given by equation 79 on page 15
double logeta(int d, int n){
	if(d == 0){
		return 0.;
	}
	
	double product = 0;
	for(int a=1;a<=d;a++){
		product += log2(1 - pow(2, d - n - a));
		product -= log2(1 - pow(2, -a));
	}
	
	return (-d*(d+1)/2) + product;
}

struct StabilizerState* randomStabilizerState(int n){
	//not using the dDists caching from python
	
	if(n<1){
		printf("randomStabilizerState: Vector space must have positive nonzero dimension.\n");
	}
	
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

	gsl_vector *tempVector;
	tempVector = gsl_vector_alloc(n);
	
    while (state->k > k){
        for(int i=0;i<n;i++){
		    gsl_vector_set(tempVector, i, rand() % 2);
	    }

		//lazy shrink with a'th row of X
		shrink(state, tempVector, 0, 1);
	}
    gsl_vector_free(tempVector);

    // Now K is a random k-dimensional subspace
	
    for(int i=0;i<n;i++){
		gsl_vector_set(state->h, i, rand() % 2);
	}
	state->Q = rand() % 8;
	for(int i=0;i<k;i++){
		gsl_vector_set(state->D, i, 2*(rand() % 4));
	}
	
	for(i=0;i<k;i++){
		gsl_matrix_set(state->J, i, i, mod(2*(int)(gsl_vector_get(state->D, i)), 8));
		for(j=0;j<i;j++){
			gsl_matrix_set(state->J, i, j, 4*(rand() % 2));
			gsl_matrix_set(state->J, j, i, gsl_matrix_get(state->J, i, j));
		}
	}
    return state;
}

//Helper: if xi not in K, extend it to an affine space that does
//Doesn't return anything, instead modifies state
void extend(struct StabilizerState *state, gsl_vector *xi){
	
	gsl_vector *tempVector, *tempVector1;
	tempVector = gsl_vector_alloc(state->n);
	tempVector1 = gsl_vector_alloc(state->n);
	double tempInt;
	
	//S = [a for a in range(self.n) if np.dot(xi, self.Gbar[a]) % 2 == 1]
	int *S;
	S = malloc(state->n * sizeof(int));
	int Slength = 0;
	for(int a=0;a<state->n;a++){
		gsl_matrix_get_row(tempVector, state->Gbar, a);
		gsl_blas_ddot(tempVector, xi, &tempInt);
		if(mod((int)tempInt, 2) == 1){
			*(S + Slength++) = a;
		}
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
	
	if(Tlength == 0){
        gsl_vector_free(tempVector);
        gsl_vector_free(tempVector1);
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
	for(int j=0;j<newSlength;j++){
		a = *(newS + j);
		gsl_matrix_get_row(tempVector, state->Gbar, a);
		gsl_matrix_get_row(tempVector1, state->Gbar, i);
		gsl_vector_add(tempVector, tempVector1);
		gsl_matrix_set_row(state->Gbar, a, tempVector);
		//TODO: more efficient %2 of vector
		for(int t=0;t<state->n;t++){
			gsl_matrix_set(state->Gbar, a, t, mod((int)gsl_matrix_get(state->Gbar, a, t), 2));
		}
		
		gsl_matrix_get_row(tempVector, state->G, i);
		gsl_matrix_get_row(tempVector1, state->G, a);
		gsl_vector_add(tempVector, tempVector1);
		gsl_matrix_set_row(state->G, i, tempVector);
	}
	//TODO: more efficient %2 of vector
	for(int t=0;t<state->n;t++){
		gsl_matrix_set(state->G, i, t, mod((int)gsl_matrix_get(state->G, i, t), 2));
	}
	//Now g^i = xi
	
	//Swap g^i and g^k (not g^{k+1} because zero indexing)
	gsl_matrix_swap_rows(state->G, i, state->k);
	gsl_matrix_swap_rows(state->Gbar, i, state->k);
	
	state->k++;

    gsl_vector_free(tempVector);
    gsl_vector_free(tempVector1);
    free(S);
    free(T);
    free(newS);
}

//Write a pauli as P = i^m * Z(zeta) * X(xi), m in Z_4
//Returns the norm of the projected state Gamma = ||P_+ |K,q>||
//If Gamma nonzero, projects the state to P_+|K,q>
double measurePauli(struct StabilizerState *state, int m, gsl_vector *zeta, gsl_vector *xi){
	
	//write zeta, xi in basis of K
	gsl_vector *vecZeta, *vecXi, *xiPrime, *tempVector;
	vecZeta = gsl_vector_alloc(state->n);
	gsl_vector_set_zero(vecZeta);
	vecXi = gsl_vector_alloc(state->n);
	gsl_vector_set_zero(vecXi);
	xiPrime = gsl_vector_alloc(state->n);
	gsl_vector_set_zero(xiPrime);
	tempVector = gsl_vector_alloc(state->n);
	double tempInt;


	
	for(int a=0;a<state->k;a++){
		gsl_matrix_get_row(tempVector, state->G, a);
		gsl_blas_ddot(tempVector, zeta, &tempInt);
		gsl_vector_set(vecZeta, a, mod((int)tempInt, 2));
		
		gsl_matrix_get_row(tempVector, state->Gbar, a);
		gsl_blas_ddot(tempVector, xi, &tempInt);
		gsl_vector_set(vecXi, a, mod((int)tempInt, 2));
	}
	for(int a=0;a<state->k;a++){
		gsl_matrix_get_row(tempVector, state->G, a);
		gsl_vector_scale(tempVector, gsl_vector_get(vecXi,a));
		gsl_vector_add(xiPrime, tempVector);
	}
	for(int a=0;a<state->n;a++){
		gsl_vector_set(xiPrime, a, mod((int)gsl_vector_get(xiPrime, a), 2));
	}
	
	//compute w in {0, 2, 4, 6} using eq. 88
	int w = 2*m;
	gsl_blas_ddot(zeta, state->h, &tempInt);
	w += 4*mod((int)tempInt, 2);
	gsl_blas_ddot(state->D, vecXi, &tempInt);
	w += (int)tempInt;
	//TODO: definitely matrix manipulations rather than loops
	for(int b=0;b<state->k;b++){
		for(int a=0;a<b;a++){
			w += (int)gsl_matrix_get(state->J, a, b)*gsl_vector_get(vecXi, a)*gsl_vector_get(vecXi, b);
		}
	}
	w = mod(w, 8);
	
	//Compute eta_0, ..., eta_{k-1} using eq. 94
	gsl_vector *eta;
	eta = gsl_vector_alloc(state->n);
	gsl_vector_memcpy(eta, vecZeta);
	for(int a=0;a<state->k;a++){
		for(int b=0;b<state->k;b++){
			gsl_vector_set(eta, a, gsl_vector_get(eta, a) + gsl_matrix_get(state->J, a, b)*gsl_vector_get(vecXi, b)/4);
		}
	}
	for(int a=0;a<state->k;a++){
		gsl_vector_set(eta, a, mod((int)gsl_vector_get(eta, a), 2));
	}
	
	int areXiXiprimeClose = 1;
	for(int a=0;a<state->n;a++){
		if(fabs(gsl_vector_get(xi, a) - gsl_vector_get(xiPrime, a)) > 0.00001){
			areXiXiprimeClose = 0;
			break;
		}
	}
	if(areXiXiprimeClose == 1){
		if(w==0 || w==4){
			gsl_vector *gamma;
			gamma = gsl_vector_alloc(state->n);
			//gsl_matrix_view Gbark = gsl_matrix_submatrix(state->Gbar, 0, 0, state->k, state->n);
			//gsl_blas_dgemv(CblasNoTrans, 1., &Gbark.matrix, eta, 0., gamma);
			//gsl_vector_view etak = gsl_vector_subvector(eta, 0, state->k);
			
			gsl_blas_dgemv(CblasTrans, 1., state->Gbar, eta, 0., gamma);
			for(int a=0;a<state->n;a++){
				gsl_vector_set(gamma, a, mod((int)gsl_vector_get(gamma, a), 2));
			}
			
			int omegaPrime = w/4;
			gsl_blas_ddot(gamma, state->h, &tempInt);
			int alpha = mod(omegaPrime + (int)tempInt, 2);
			
			int eps = shrink(state, gamma, alpha, 0);
            
            gsl_vector_free(vecZeta);
            gsl_vector_free(vecXi);
            gsl_vector_free(xiPrime);
            gsl_vector_free(tempVector);
            gsl_vector_free(eta);
            gsl_vector_free(gamma);

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
			state->Q = mod(state->Q + sigma, 8);
			for(int a=0;a<state->k;a++){
				gsl_vector_set(state->D, a, mod((int)gsl_vector_get(state->D, a) - 2*sigma*(int)gsl_vector_get(eta, a), 8));
			}
			
			//ignore a != b for some reason, MATLAB code does it too
			//still satisfies J[a,a] = 2 D[a] mod 8
			gsl_matrix *etaMatrix, *tempMatrix;
			etaMatrix = gsl_matrix_alloc(1, state->n);
			tempMatrix = gsl_matrix_alloc(state->n, state->n);
			gsl_matrix_set_row(etaMatrix, 0, eta);
			gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1., etaMatrix, etaMatrix, 0., tempMatrix);
			gsl_matrix_scale(tempMatrix, 4);
			gsl_matrix_add(state->J, tempMatrix);
			//TODO: more efficient modulo of a matrix..
			for(int i=0;i<state->n;i++){
				for(int j=0;j<state->n;j++){
					gsl_matrix_set(state->J, i, j, mod((int)gsl_matrix_get(state->J, i, j), 8));
				}
			}

            gsl_vector_free(vecZeta);
            gsl_vector_free(vecXi);
            gsl_vector_free(xiPrime);
            gsl_vector_free(tempVector);
            gsl_vector_free(eta);
            gsl_matrix_free(etaMatrix);
            gsl_matrix_free(tempMatrix);
			return pow(2, -0.5);
		}
	}
	
	//remaining case: xiPrime != xi
	extend(state, xi);
	
	//update D
	gsl_vector_memcpy(tempVector, xi);
	gsl_vector_add(tempVector, state->h);
	gsl_blas_ddot(zeta, tempVector, &tempInt);
	int newDval = mod(2*m + 4*mod((int)tempInt, 2), 8);
	gsl_vector_set(state->D, state->k-1, newDval);
	
	//update J 
	gsl_vector_scale(vecZeta, 4);
	gsl_matrix_set_col(state->J, state->k-1, vecZeta);
	gsl_matrix_set_row(state->J, state->k-1, vecZeta);
	gsl_matrix_set(state->J, state->k-1, state->k-1, mod(4*m, 8));


    gsl_vector_free(vecZeta);
    gsl_vector_free(vecXi);
    gsl_vector_free(xiPrime);
    gsl_vector_free(tempVector);
    gsl_vector_free(eta);
	return pow(2, -0.5);
}


void freeStabilizerState(struct StabilizerState *state) {
    gsl_vector_free(state->h);
    gsl_matrix_free(state->G);
    gsl_matrix_free(state->Gbar);

    gsl_vector_free(state->D);
    gsl_matrix_free(state->J);

    free(state);
}


void printStabilizerState(struct StabilizerState *state) {
    printf("state.n = %d\n", state->n);
    printf("state.k = %d\n", state->k);
    
    printf("state.h = np.array([");
    for (int i = 0; i<state->n; i++) {
        printf("%d", (int)gsl_vector_get(state->h, i));
        if (i+1 != state->n) printf(",");
    }
    printf("])\n");

    printf("state.G = np.array([");
    for (int i = 0; i<state->n; i++) {
        printf("[");
        for (int j = 0; j<state->n; j++) {
            printf("%d", (int)gsl_matrix_get(state->G, i, j));
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
            printf("%d", (int)gsl_matrix_get(state->Gbar, i, j));
            if (j+1 != state->n) printf(",");
        }
        printf("]");
        if (i+1 != state->n) printf(",");
    }
    printf("])\n");

    printf("state.Q = %d\n", state->Q);

    printf("state.D = np.array([");
    for (int i = 0; i<state->k; i++) {
        printf("%d", (int)gsl_vector_get(state->D, i));
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
                printf("%d", (int)gsl_matrix_get(state->J, i, j));
                if (j+1 != state->k) printf(",");
            }
            printf("]");
            if (i+1 != state->k) printf(",");
        }
        printf("])\n");
    }
}
