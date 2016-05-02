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
	time_t t;
	srand((unsigned) time(&t));
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

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

void deepCopyState(struct StabilizerStates *dest, struct StabilizerStates *src){
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
void updateDJ(struct StabilizerStates *state, gsl_matrix *R){
	
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
	//gsl_vector_free(tempVector);
	//sl_vector_free(tempVector1);
	//gsl_matrix_free(tempMatrix);
}

//helper to update Q, D using equations 51, 52 on page 10
void updateQD(struct StabilizerStates *state, gsl_vector *y){
	
	
	//temporary variables for storing intermediary results
	double tempInt;
	gsl_vector *tempVector;
	tempVector = gsl_vector_alloc(state->k);
	
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
	//gsl_vector_free(tempVector);
}

//helper that evaluates the expression in the comment on page 12
void evalW(gsl_complex *ans, int eps, int p, int m){
	//imaginary unit
	gsl_complex eye = gsl_complex_rect(0,1);
	
	*ans = gsl_complex_exp(gsl_complex_mul_real(eye, M_PI*m/4));
	*ans = gsl_complex_mul_real(*ans, eps*pow(2,p/2.));
}

//Helpers for evaluating equations like 63, 68. For even A,B only!

//Evaluates 1 + e^{A*i*pi/4} + e^{A*i*pi/4} - e^{(A + B)*i*pi/4}
void Gamma(int *eps, int *p, int *m, int A, int B){
	if(mod(A,2)==1 || mod(B,2)==1){
		printf("Gamma: A and B must be even!");
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
}

//Evaluates 1 + e^{A*i*pi/4}
void partialGamma(int *eps, int *p, int *m, int A){
	if(mod(A,2)==1){
		printf("partialGamma: A must be even!");
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

void Wsigma(struct StabilizerStates *state, int *eps, int *p, int *m, gsl_complex *ans, 
	int exact, int sigma, int s, int *M, int Mlength, int *Dimers, int DimersLength){
	
	if(state->k == 0){
		if(exact == 1){
			*eps = 1;
			*p = 0;
			*m = state->Q;
			return;
		}
		else{
			evalW(ans, 1, 0, state->Q);
			return;
		}
	}
	
	//W = (1, 0, self.Q + sigma*self.D[s])
	int tempEps = 1;
	int tempP = 0;
	int tempM = state->Q + sigma*((int)gsl_vector_get(state->D, s));
	for(int i=0;i<Mlength;i++){
		partialGamma(eps, p, m, sigma*((int)gsl_matrix_get(state->J, *(M + i), s)));
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
	if(exact == 1){
		evalW(ans, *eps, *p, *m);
	}
}

//Helper required for InnerProduct and MeasurePauli.
//Depends only on Q, D, J. Manipulates integers p, m, eps
//to avoid rounding error then evaluates to a real number.
void exponentialSum(struct StabilizerStates *state, int *eps, int *p, int *m, gsl_complex *ans, int exact){
	
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
		
	if(Slength == 0){
		//Compute W(K,q) from Eq. 63
		Wsigma(state, eps, p, m, ans, exact, 0, 0, M, Mlength, Dimers, DimersLength);
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
			return;
		}
		else{
			int eps0, p0, m0, eps1, p1, m1;
			Wsigma(state, &eps0, &p0, &m0, ans, exact, 0, *(S), M, Mlength, Dimers, DimersLength);
			Wsigma(state, &eps1, &p1, &m1, ans, exact, 1, *(S), M, Mlength, Dimers, DimersLength);
			
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
				printf("ExponentialSum: p0, p1 must be equal!");
				return;
			}
			if(mod((m1-m0), 2) == 1){
				printf("ExponentialSum: m1-m0 must be even!");
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
int shrink(struct StabilizerStates *state, gsl_vector *xi, int alpha, int lazy){
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
			return 0;
		}
		if(beta == 0){
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
		y = gsl_vector_calloc(state->k);
		gsl_vector_set(y, state->k-1, beta);
		updateQD(state, y);
		
		//remove last row and column from J
		gsl_matrix_view newJ = gsl_matrix_submatrix(state->J, 0, 0, state->k-1, state->k-1);
		//gsl_matrix_free(state->J);
		state->J = &newJ.matrix;
		
		//remove last element from D
		gsl_vector_view newD = gsl_vector_subvector(state->D, 0, state->k-1);
		//gsl_vector_free(state->D);
		state->D = &newD.vector;
	}
	
	state->k--;
	
	return 2;
}

void innerProduct(struct StabilizerStates *state1, struct StabilizerStates *state2, int *eps, int *p, int *m, gsl_complex *ans, int exact){
	if(state1->n != state2->n){
		printf("innerProduct: States do not have same dimension.");
		return;
	}
	
	int i, j, b, alpha;
	double tempInt;
	gsl_vector *tempVector, *tempVector1;
	tempVector = gsl_vector_alloc(state2->n);
	tempVector1 = gsl_vector_alloc(state2->n);
	
	//K <- K_1, (also copy q_1)
	struct StabilizerStates *state;
	deepCopyState(state, state1);
	for(b=state2->k;b<state2->n;b++){
		gsl_matrix_get_row(tempVector, state2->Gbar, b);
		gsl_blas_ddot(state2->h, tempVector, &tempInt);
		alpha = mod((int)tempInt, 2);
		*eps = shrink(state2, tempVector, alpha, 0);
		if(*eps == 0){
			*eps = 0;
			*p = 0;
			*m = 0;
			*ans = gsl_complex_rect(0,0);
		}
	}
	
	//Now K = K_1 \cap K_2
	gsl_vector *y;
	y = gsl_vector_alloc(state2->k);
	gsl_vector_memcpy(tempVector, state->h);
	gsl_vector_add(tempVector, state2->h);
	for(i=0;i<state2->k;i++){
		gsl_matrix_get_row(tempVector1, state2->Gbar, i);
		gsl_blas_ddot(tempVector, tempVector1, &tempInt);
		gsl_vector_set(y, i, mod((int)tempInt, 2));
	}
	
	gsl_matrix *smallR, *R;
	smallR = gsl_matrix_calloc(state->k, state2->k);
	gsl_vector *smallRrow;
	smallRrow = gsl_vector_alloc(state2->k);
	R = gsl_matrix_alloc(state->k+1, state2->k);
	gsl_matrix_view Gk = gsl_matrix_submatrix(state->G, 0, 0, state->k, state->n);
	gsl_matrix_view Gk2 = gsl_matrix_submatrix(state2->Gbar, 0, 0, state2->k, state2->n);
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, &Gk.matrix, &Gk2.matrix, 0, smallR);
	for(i=0;i<state->k;i++){
		gsl_matrix_get_row(smallRrow, smallR, i);
		gsl_matrix_set_row(R, i, smallRrow);
	}
	//TODO: more efficient modulo of a matrix..
	for(i=0;i<state->k;i++){
		for(j=0;j<state2->k;j++){
			gsl_matrix_set(R, i, j, mod((int)gsl_matrix_get(R, i, j), 2));
		}
	}
	
	
	struct StabilizerStates *state2temp;
	deepCopyState(state2temp, state2);
	
	updateQD(state2temp, y);
	updateDJ(state2temp, R);
	
	//now q, q2 are defined in the same basis
	state->Q = state->Q - mod(state2temp->Q, 8);
	for(i=0;i<state->k;i++){
		gsl_vector_set(state->D, i, gsl_vector_get(state->D, i) - mod((int)gsl_vector_get(state2temp->D, i), 8));
		for(j=0;j<state->k;j++){
			gsl_matrix_set(state->J, i, j, gsl_matrix_get(state->J, i, j) - mod((int)gsl_matrix_get(state2temp->J, i, j), 8));
		}
	}	
	
	if(exact == 0){
		exponentialSum(state, eps, p, m, ans, 0);
		*ans = gsl_complex_mul_real(*ans, pow(2, -(state1->k + state2temp->k)/2));
		return;
	}
	else{
		exponentialSum(state, eps, p, m, ans, 1);
		*p -= state1->k + state2temp->k;
	}
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

void randomStabilizerState(struct StabilizerStates *state, int n){
	//not using the dDists caching from python
	
	if(n<2){
		printf("randomStabilizerState: Vector space must have positive nonzero nontrivial dimension.");
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
		for(d=0;d<i;d++){
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
	
	time_t t;
	srand((unsigned) time(&t));
	
	gsl_matrix *X;
	
	if(d > 0){
		//pick random X in \mathbb{F}^{d,n}_2 with rank d
		gsl_matrix *U, *V;
		gsl_vector *S;
		X = gsl_matrix_alloc(n, d); 
		U = gsl_matrix_alloc(n, d);
		V = gsl_matrix_alloc(n, n);  
		S = gsl_vector_alloc(n); 
		gsl_vector *work;
		work = gsl_vector_alloc(n);
		int rank;
		double tempInt;
		//X is d x n but we'll transpose it later
		//need those dimensions to make SVD work since n>=d
		while(1){
			for(i=0;i<n;i++){
				for(j=0;j<d;j++){
					gsl_matrix_set(X, i, j, rand() % 2);
				}
			}
			
			//rank of a matrix is the number of non-zero values in its singular value decomposition
			rank = 0;
			gsl_matrix_memcpy(U, X);
			gsl_linalg_SV_decomp(U, V, S, work);
			for(i=0;i<n;i++){
				if(fabs(gsl_vector_get(S, i)) > 0.00001){
					rank++;
				}
			}
			if(rank == d){
				break;
			}
		}
		gsl_matrix_transpose(X);
		
		//gsl_matrix_free(U);
		//gsl_matrix_free(V);
		//gsl_vector_free(S);
		//gsl_vector_free(work);
	}
	
	state->n = n;
	state->k = k;
	
	gsl_vector *tempVector;
	tempVector = gsl_vector_alloc(n);
	
	for(i=0;i<d;i++){
		//lazy shrink with a'th row of X
		gsl_matrix_get_row(tempVector, X, i);
		shrink(state, tempVector, 0, 1);
	}
	state->k = k;
	
	//now K = ker(X) and is in standard form
	state->h = gsl_vector_alloc(n);
	for(int i=0;i<n;i++){
		gsl_vector_set(state->h, i, rand() % 2);
	}
	state->Q = rand() % 8;
	state->D = gsl_vector_alloc(k);
	for(int i=0;i<k;i++){
		gsl_vector_set(state->D, i, 2*(rand() % 4));
	}
	
	state->J = gsl_matrix_alloc(k, k);
	for(i=0;i<k;i++){
		gsl_matrix_set(state->J, i, i, mod(2*(int)(gsl_vector_get(state->D, i)), 8));
		for(j=0;j<i;j++){
			gsl_matrix_set(state->J, i, j, 4*(rand() % 2));
			gsl_matrix_set(state->J, j, i, gsl_matrix_get(state->J, i, j));
		}
	}
}

//Helper: if xi not in K, extend it to an affine space that does
//Doesn't return anything, instead modifies state
void extend(struct StabilizerStates *state, gsl_vector *xi){
	
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
}

//Write a pauli as P = i^m * Z(zeta) * X(xi), m in Z_4
//Returns the norm of the projected state Gamma = ||P_+ |K,q>||
//If Gamma nonzero, projects the state to P_+|K,q>
double measurePauli(struct StabilizerStates *state, int m, gsl_vector *zeta, gsl_vector *xi){
	
	//write zeta, xi in basis of K
	gsl_vector *vecZeta, *vecXi, *xiPrime, *tempVector;
	vecZeta = gsl_vector_alloc(state->k);
	vecXi = gsl_vector_alloc(state->k);
	xiPrime = gsl_vector_alloc(state->k);
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
	for(int a=0;a<state->k;a++){
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
	eta = gsl_vector_alloc(state->k);
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
	for(int a=0;a<state->k;a++){
		if(fabs(gsl_vector_get(xi, a) - gsl_vector_get(xiPrime, a)) > 0.00001){
			areXiXiprimeClose = 0;
			break;
		}
	}
	if(areXiXiprimeClose == 1){
		if(w==0 || w==4){
			gsl_vector *gamma;
			gamma = gsl_vector_alloc(state->n);
			gsl_matrix_view Gbark = gsl_matrix_submatrix(state->Gbar, 0, 0, state->k, state->n);
			gsl_blas_dgemv(CblasNoTrans, 1., &Gbark.matrix, eta, 0., gamma);
			for(int a=0;a<state->n;a++){
				gsl_vector_set(gamma, a, mod((int)gsl_vector_get(gamma, a), 2));
			}
			
			int omegaPrime = w/4;
			gsl_blas_ddot(gamma, state->h, &tempInt);
			int alpha = mod(omegaPrime + (int)tempInt, 2);
			
			int eps = shrink(state, gamma, alpha, 0);
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
			etaMatrix = gsl_matrix_alloc(1, state->k);
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

			return pow(2, -0.5);
		}
	}
	
	//remaining case: xiPrime != xi
	extend(state, xi);
	gsl_vector_memcpy(tempVector, xi);
	gsl_vector_add(tempVector, state->h);
	gsl_blas_ddot(zeta, tempVector, &tempInt);
	int newDval = mod(2*m + 4*mod((int)tempInt, 2), 8);
	
	//must be a better way to extend a vector/matrix in GSL
	
	//TODO: not sure if that's correct
	gsl_vector *newD;
	newD = gsl_vector_alloc(state->k);
	for(int i=0;i<state->k-1;i++){
		gsl_vector_set(newD, i, gsl_vector_get(state->D, i));
	}
	gsl_vector_set(newD, state->k-1, newDval);
	state->D = newD;
	
	//TODO
	//self.J = np.bmat([[self.J, np.array([4*vecZeta]).T],
	//					[np.array([4*vecZeta]), [[(4*m) % 8]]]])

	return pow(2, -0.5);
}

void testUpdateDJ(){
	printf("\nTest updateDJ:\n");
	struct StabilizerStates state;
	
	state.n = 40;
	state.k = 40;
	double Rdata[] = { 
		1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1 
	};
	double Ddata[] = {
		2, 6, 0, 4, 4, 0, 6, 0, 0, 4, 2, 0, 6, 2, 0, 0, 6, 4, 2, 2, 6, 4, 0, 2, 2, 0, 4, 6, 4, 2, 2, 6, 0, 6, 2, 0, 0, 6, 6, 0
	};
	double Jdata[] = {
		4,0,4,4,0,4,0,4,4,4,4,0,0,4,0,0,4,4,4,4,4,0,0,0,4,4,0,0,4,0,4,0,0,4,0,0,0,0,4,0,0,4,0,4,4,0,0,4,4,4,0,0,4,4,4,4,0,0,0,0,0,0,4,0,4,4,0,0,0,4,4,4,4,0,0,4,0,4,0,0,4,0,0,0,4,4,0,0,4,0,0,0,4,0,0,4,0,4,4,4,0,0,4,4,4,0,0,0,4,0,4,0,0,0,0,0,0,4,4,0,4,4,0,0,4,4,4,0,4,4,0,0,0,4,4,0,0,4,0,0,4,4,4,0,0,0,4,4,0,4,4,0,0,4,4,0,4,0,4,0,0,4,4,4,0,4,4,0,4,0,0,0,0,4,0,0,0,0,0,0,0,0,0,4,4,4,0,0,0,4,4,4,0,0,0,4,4,0,0,0,4,0,4,4,4,0,4,0,0,0,0,0,0,0,4,0,0,4,4,4,4,0,4,0,0,0,0,0,0,4,4,0,0,4,4,4,4,0,0,4,0,0,0,4,4,4,4,4,4,0,4,0,4,4,4,4,0,4,0,4,0,0,4,0,0,4,4,4,0,0,4,0,0,0,4,4,4,4,0,0,4,4,0,0,0,0,4,0,0,4,4,4,4,4,0,0,4,4,0,0,0,0,4,0,0,0,0,0,0,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,4,0,0,4,4,0,4,0,0,0,0,0,4,4,4,4,0,0,4,0,0,0,0,0,0,4,0,4,4,0,4,0,0,4,4,4,0,4,0,0,0,4,4,0,0,4,0,4,0,0,0,4,4,4,0,0,0,4,0,0,0,0,4,4,0,4,0,0,0,4,0,4,4,4,4,0,0,0,0,0,4,4,4,0,4,4,0,0,4,0,0,0,0,0,4,0,0,0,4,0,4,4,4,0,0,0,4,0,4,4,0,4,4,4,0,0,0,0,0,0,0,4,0,4,4,0,4,0,4,0,0,4,4,0,4,4,4,0,4,4,4,4,4,4,0,0,0,0,0,4,0,0,0,4,0,4,4,0,0,0,4,4,4,0,0,4,4,0,4,4,0,4,4,0,0,0,0,4,0,4,0,4,0,0,0,4,0,0,4,4,4,4,0,4,4,4,0,4,4,0,4,4,0,4,0,0,0,4,4,4,0,0,4,0,0,4,0,4,4,0,0,4,0,4,4,0,4,4,0,4,0,4,4,4,0,4,0,4,0,4,4,0,0,0,4,4,4,4,0,0,4,4,4,4,4,0,0,0,4,4,4,4,4,4,4,4,0,0,4,4,4,4,4,0,0,4,4,0,0,0,4,0,0,0,0,0,4,4,0,0,0,0,0,0,0,0,4,4,0,4,4,4,4,4,4,4,0,0,0,0,4,4,4,0,4,0,0,0,0,0,0,4,0,0,0,0,0,0,4,0,4,4,0,0,4,0,4,0,0,0,4,4,4,0,0,4,4,0,0,0,4,0,0,0,4,0,4,4,0,4,4,4,0,4,0,4,4,0,4,0,4,0,4,0,0,4,0,0,0,4,0,0,0,0,0,4,4,0,4,0,0,0,4,4,4,0,4,0,0,4,0,0,4,4,0,4,4,4,4,0,0,4,4,4,0,4,4,0,4,4,0,0,4,0,4,0,4,4,4,4,0,4,4,0,4,0,4,0,0,4,4,0,4,4,0,0,0,0,4,0,0,0,4,4,4,4,0,0,4,0,0,0,0,0,0,0,0,4,4,0,4,0,0,4,4,0,0,4,0,4,0,0,4,0,4,4,0,0,4,0,4,0,0,4,4,4,4,4,4,0,4,0,4,4,0,4,4,4,4,0,4,4,4,0,0,0,0,4,0,0,0,0,4,0,0,4,0,4,0,0,0,4,4,4,4,0,4,4,4,0,4,4,4,4,4,0,0,0,0,0,0,4,4,0,0,4,4,4,0,4,4,4,0,0,0,4,0,0,0,4,4,0,4,0,4,4,0,0,0,0,0,0,0,4,0,4,0,4,4,4,4,0,4,0,0,0,4,0,4,0,0,0,0,4,0,0,4,4,0,4,0,0,0,0,4,4,0,4,0,0,4,4,4,4,4,4,4,0,4,0,0,4,4,0,4,4,4,0,4,0,0,0,4,0,4,4,0,4,4,0,0,0,4,4,4,4,0,0,4,4,0,4,0,0,0,0,0,4,0,4,4,4,4,0,4,4,0,0,4,0,4,0,0,0,0,4,4,0,4,4,0,4,4,0,0,0,0,0,4,0,4,0,4,4,4,0,4,4,0,0,4,0,4,4,0,0,0,4,0,0,4,0,0,0,4,4,0,0,4,4,4,0,0,0,4,4,0,4,0,4,0,4,4,0,0,0,4,4,4,0,0,4,0,0,0,0,0,4,0,0,4,0,0,0,4,4,4,4,4,4,4,0,0,0,0,4,0,4,4,0,4,4,4,4,4,0,0,0,0,0,4,4,0,4,4,0,4,0,0,0,0,0,0,4,4,4,0,0,4,4,4,0,4,0,4,4,0,4,0,4,4,4,0,4,0,0,4,0,0,0,0,4,0,0,0,4,0,4,4,4,0,4,0,4,0,4,0,4,4,4,0,0,0,0,4,4,4,4,0,4,0,4,4,4,4,4,0,0,0,4,4,4,4,4,4,4,4,4,4,4,4,4,0,0,0,0,0,4,4,4,0,0,4,0,0,4,0,4,0,4,0,4,0,4,4,4,0,4,0,0,0,4,0,4,0,4,0,0,4,0,0,4,4,4,0,0,4,0,4,4,4,4,0,0,4,0,4,4,0,0,0,0,0,4,4,4,4,0,4,4,0,0,4,0,0,4,0,0,0,0,0,4,0,0,4,0,0,4,0,0,4,4,4,0,4,0,0,4,0,4,4,0,4,0,0,4,0,0,4,4,4,4,0,4,4,0,0,4,0,4,0,4,4,0,0,0,0,4,0,0,0,0,4,4,4,0,4,0,4,4,4,0,0,0,4,0,0,4,0,4,4,4,0,0,0,0,0,4,0,4,4,4,4,0,4,0,4,0,4,0,0,4,4,4,4,0,4,4,0,0,4,0,0,0,0,4,4,0,4,4,0,4,4,4,0,4,0,0,4,4,4,4,0,4,4,4,4,4,4,0,0,0,4,0,0,0,4,0,4,0,0,0,0,4,0,4,4,4,4,0,4,0,4,4,0,0,0,4,4,4,4,4,4,0,0,0,4,0,4,4,4,0,0,4,4,0,4,0,4,4,0,4,0,4,0,0,4,4,0,4,0,4,4,0,0,4,4,0,0,0,4,4,0,4,4,0,4,4,4,4,0,0,4,0,4,4,0,4,4,0,4,4,4,4,4,0,4,4,4,0,4,4,4,4,4,0,4,4,0,0,0,4,0,4,4,0,0,4,4,4,0,4,4,0,4,4,4,4,4,4,0,0,0,4,0,4,0,0,4,4,4,4,4,4,0,0,0,0,0,4,0,4,4,4,4,4,4,4,0,0,0,4,0,4,0,0,0,0,0,4,0,4,0,4,4,0,4,0,4,4,0,4,4,0
	};
	double outDdata[] = {
		2.0, 6.0, 0.0, 4.0, 2.0, 0.0, 6.0, 0.0, 0.0, 4.0, 2.0, 0.0, 6.0, 2.0, 0.0, 0.0, 6.0, 4.0, 2.0, 2.0, 6.0, 4.0, 0.0, 2.0, 2.0, 0.0, 4.0, 6.0, 4.0, 2.0, 2.0, 6.0, 0.0, 6.0, 2.0, 0.0, 0.0, 6.0, 6.0, 0.0
	};
	double outJdata[] = {
		4,0,4,4,0,4,0,4,4,4,4,0,0,4,0,0,4,4,4,4,4,0,0,0,4,4,0,0,4,0,4,0,0,4,0,0,0,0,4,0,0,4,0,4,4,0,0,4,4,4,0,0,4,4,4,4,0,0,0,0,0,0,4,0,4,4,0,0,0,4,4,4,4,0,0,4,0,4,0,0,4,0,0,0,4,4,0,0,4,0,0,0,4,0,0,4,0,4,4,4,0,0,4,4,4,0,0,0,4,0,4,0,0,0,0,0,0,4,4,0,4,4,0,0,0,4,4,0,4,4,0,0,0,4,4,0,0,4,0,0,4,4,4,0,0,0,4,4,0,4,4,0,0,4,4,0,4,0,4,0,0,4,4,0,4,4,0,0,4,0,4,4,4,0,4,4,4,0,0,0,0,4,0,0,0,4,4,4,4,0,0,4,0,0,0,4,0,4,0,4,4,0,4,4,4,0,4,0,0,0,0,0,0,0,4,0,0,4,4,4,4,0,4,0,0,0,0,0,0,4,4,0,0,4,4,4,4,0,0,4,0,0,0,4,0,4,4,4,4,0,4,0,4,4,4,4,0,4,0,4,0,0,4,0,0,4,4,4,0,0,4,0,0,0,4,4,4,4,0,0,4,4,0,0,0,0,4,0,0,4,4,4,4,4,0,0,4,4,0,0,0,0,4,0,0,0,0,0,0,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,4,0,0,4,4,0,4,0,0,0,0,0,4,4,4,4,0,0,4,0,0,0,0,0,0,4,0,4,4,0,4,0,0,4,4,4,0,4,0,0,0,4,4,0,0,4,0,4,0,0,0,4,4,4,0,0,0,4,0,0,0,0,4,4,0,4,0,0,0,4,0,4,4,4,4,0,0,0,4,0,4,4,4,0,4,4,0,0,4,0,0,0,0,0,4,0,0,0,4,0,4,4,4,0,0,0,4,0,4,4,0,4,4,4,0,0,0,0,4,0,0,4,0,4,4,0,4,0,4,0,0,4,4,0,4,4,4,0,4,4,4,4,4,4,0,0,0,0,0,4,0,0,0,4,0,4,4,0,4,0,4,4,4,0,0,4,4,0,4,4,0,4,4,0,0,0,0,4,0,4,0,4,0,0,0,4,0,0,4,4,4,4,0,4,4,4,0,4,0,0,4,4,0,4,0,0,0,4,4,4,0,0,4,0,0,4,0,4,4,0,0,4,0,4,4,0,4,4,0,4,0,4,4,4,0,4,0,4,4,4,4,0,0,0,4,4,4,4,0,0,4,4,4,4,4,0,0,0,4,4,4,4,4,4,4,4,0,0,4,4,4,4,4,0,0,4,4,0,4,0,4,0,0,0,0,0,4,4,0,0,0,0,0,0,0,0,4,4,0,4,4,4,4,4,4,4,0,0,0,0,4,4,4,0,4,0,0,0,4,0,0,4,0,0,0,0,0,0,4,0,4,4,0,0,4,0,4,0,0,0,4,4,4,0,0,4,4,0,0,0,4,0,0,0,4,0,4,4,0,4,4,4,0,4,0,4,4,0,4,0,4,0,4,0,0,4,0,0,0,4,0,0,0,0,0,4,4,0,4,0,0,0,4,4,4,0,4,0,0,4,0,0,4,4,0,4,4,4,4,0,0,4,4,4,0,4,4,0,4,4,0,0,4,0,4,0,4,4,4,4,0,4,4,0,4,0,4,0,0,4,4,0,4,4,0,0,0,0,4,0,0,0,4,4,4,4,0,0,4,0,0,0,0,0,0,0,0,4,4,0,4,0,0,4,4,0,0,4,0,4,0,0,4,0,4,4,0,0,4,0,4,0,0,4,4,4,4,4,4,0,4,0,4,4,0,4,4,4,4,0,4,4,4,0,0,0,0,4,4,0,0,0,4,0,0,4,0,4,0,0,0,4,4,4,4,0,4,4,4,0,4,4,4,4,4,0,0,0,0,0,0,4,4,0,0,4,4,4,0,4,4,4,0,0,0,4,0,0,0,4,4,0,4,0,4,4,0,0,0,0,0,0,0,4,0,4,0,4,4,4,4,0,4,0,0,0,4,0,0,0,0,0,0,4,0,0,4,4,0,4,0,0,0,0,4,4,0,4,0,0,4,4,4,4,4,4,4,0,4,0,0,4,4,0,4,4,4,0,0,0,0,0,4,0,4,4,0,4,4,0,0,0,4,4,4,4,0,0,4,4,0,4,0,0,0,0,0,4,0,4,4,4,4,0,4,4,0,0,4,0,4,0,0,0,0,4,4,0,4,4,0,4,4,0,0,0,0,0,4,0,4,0,4,4,4,0,4,4,0,0,4,0,4,4,0,0,0,4,4,0,4,0,0,0,4,4,0,0,4,4,4,0,0,0,4,4,0,4,0,4,0,4,4,0,0,0,4,4,4,0,0,4,0,0,0,0,0,4,4,0,4,0,0,0,4,4,4,4,4,4,4,0,0,0,0,4,0,4,4,0,4,4,4,4,4,0,0,0,0,0,4,4,0,4,4,0,4,0,4,0,0,0,0,4,4,4,0,0,4,4,4,0,4,0,4,4,0,4,0,4,4,4,0,4,0,0,4,0,0,0,0,4,0,0,0,4,0,4,0,4,0,4,0,4,0,4,0,4,4,4,0,0,0,0,4,4,4,4,0,4,0,4,4,4,4,4,0,0,0,4,4,4,4,4,4,4,4,4,0,4,4,4,0,0,0,0,0,4,4,4,0,0,4,0,0,4,0,4,0,4,0,4,0,4,4,4,0,4,0,0,0,4,0,4,0,4,0,0,4,0,0,4,4,4,0,0,4,0,4,4,4,4,0,0,4,0,4,4,0,0,0,0,0,4,4,4,4,0,4,4,0,0,4,0,0,4,0,0,0,0,0,4,0,0,4,0,0,4,0,0,4,4,4,0,4,0,0,4,0,4,4,0,4,0,0,4,0,0,4,4,4,4,0,4,4,0,0,4,0,4,0,4,4,0,0,0,0,4,0,0,0,0,4,4,4,0,4,0,4,4,4,0,0,0,4,0,0,4,0,4,4,4,0,0,0,0,0,4,0,4,4,4,4,0,4,0,4,0,4,0,0,4,4,4,4,0,4,4,0,0,4,0,0,0,0,4,4,0,4,4,0,4,4,4,0,4,0,0,4,4,4,4,0,4,4,4,4,4,4,0,0,0,4,0,0,0,4,0,4,0,0,0,0,4,0,4,4,4,4,0,4,0,4,4,0,0,0,4,0,4,4,4,4,0,0,0,4,0,4,4,4,0,0,4,4,0,4,0,4,4,0,4,0,4,0,0,4,4,0,4,0,4,4,0,0,4,4,0,4,0,4,4,0,4,4,0,4,4,4,4,0,0,4,0,4,4,0,4,4,0,4,4,4,4,4,0,4,4,4,0,4,4,4,4,4,0,4,4,0,0,0,4,0,4,4,0,0,4,4,4,0,4,4,0,4,4,4,4,4,4,0,0,0,4,0,4,0,0,4,4,4,4,4,4,0,0,0,0,4,4,0,4,4,4,4,4,4,4,0,0,0,4,0,4,0,0,0,0,0,4,0,4,0,4,4,0,4,0,4,4,0,4,4,0
	};
	
	gsl_matrix_view RmatrixView = gsl_matrix_view_array(Rdata, state.k, state.k);
	gsl_matrix *R;
	R = &RmatrixView.matrix;
	gsl_vector_view DvectorView = gsl_vector_view_array(Ddata, state.k);
	state.D = &DvectorView.vector;
	gsl_matrix_view JmatrixView = gsl_matrix_view_array(Jdata, state.k, state.k);
	state.J = &JmatrixView.matrix;
	gsl_vector_view outDVectorView = gsl_vector_view_array(outDdata, state.k);
	gsl_vector *outD;
	outD = &outDVectorView.vector;
	gsl_matrix_view outJMatrixView = gsl_matrix_view_array(outJdata, state.k, state.k);
	gsl_matrix *outJ;
	outJ = &outJMatrixView.matrix;
	
	updateDJ(&state, R);
	
	int isDWorking = 1;
	for(int i=0;i<state.k;i++){
		if(gsl_vector_get(state.D, i) != gsl_vector_get(outD, i)){
			isDWorking = 0;
			break;
		}
	}
	printf("D %s\n",isDWorking>0?"works":"fails");
	
	int isJWorking = 1;
	for(int i=0;i<state.k;i++){
		for(int j=0;j<state.k;j++){
			if(gsl_matrix_get(state.J, i, j) != gsl_matrix_get(outJ, i, j)){
				isJWorking = 0;
				break;
			}
		}
	}
	printf("J %s\n",isJWorking>0?"works":"fails");
	
	printf("----------------------");
	
	//free memory
	//gsl_vector_free(state.D);
	//gsl_vector_free(outD);
	//gsl_matrix_free(R);
	//gsl_matrix_free(state.J);
	//gsl_matrix_free(outJ);
}

void testUpdateQD(){
	printf("\nTest updateQD:\n");
	struct StabilizerStates state;
	
	state.n = 40;
	state.k = 40;
	state.Q = 3;
	double Ddata[] = {
		2.0, 6.0, 0.0, 4.0, 2.0, 6.0, 0.0, 6.0, 0.0, 4.0, 2.0, 0.0, 6.0, 2.0, 0.0, 0.0, 6.0, 4.0, 2.0, 2.0, 6.0, 4.0, 0.0, 2.0, 4.0, 6.0, 6.0, 0.0, 4.0, 2.0, 2.0, 6.0, 0.0, 6.0, 2.0, 0.0, 0.0, 6.0, 6.0, 6.0
	};
	double Jdata[] = {
		4,0,4,4,0,4,0,4,4,4,4,0,0,4,0,0,4,4,4,4,4,0,0,0,4,4,0,0,4,0,4,0,0,4,0,0,0,0,4,0,0,4,0,4,4,0,0,4,4,4,0,0,4,4,4,4,0,0,0,0,0,0,4,0,4,4,0,0,0,4,4,4,4,0,0,4,0,4,0,0,4,0,0,0,4,4,0,0,4,0,0,0,4,0,0,4,0,4,4,4,0,0,4,4,4,0,0,0,4,0,4,0,0,0,0,0,0,4,4,0,4,4,0,0,0,0,0,4,4,4,0,0,0,4,4,0,0,4,0,0,4,4,4,0,4,4,0,0,0,4,4,0,0,4,4,0,4,0,4,4,0,4,4,0,4,0,4,4,4,0,4,4,4,0,4,4,4,0,0,0,0,4,0,0,4,0,0,4,4,0,0,4,0,0,0,4,0,4,0,4,4,0,4,0,0,4,4,4,0,0,4,4,4,4,0,4,4,4,4,4,4,4,4,4,0,4,0,0,4,0,0,0,0,4,4,4,0,4,0,4,0,0,0,0,4,4,0,4,4,0,0,4,0,0,0,0,4,4,0,4,0,4,4,4,4,4,0,4,4,4,0,0,0,0,4,4,0,0,0,0,4,4,0,4,4,4,4,4,0,4,0,0,0,0,4,4,0,4,0,0,0,4,4,4,0,4,0,0,4,0,0,4,4,4,4,4,0,0,4,4,4,4,4,4,4,0,4,0,0,4,4,0,4,0,0,0,0,0,4,4,4,4,0,0,4,0,0,4,0,0,0,4,0,4,4,0,4,0,0,0,4,4,0,4,0,0,0,4,4,0,0,4,0,4,0,0,0,4,4,4,0,0,0,4,0,0,0,4,4,4,0,4,0,0,0,4,0,4,4,0,4,0,0,0,4,4,0,0,4,0,4,4,0,0,4,0,0,0,0,0,4,0,0,0,0,4,0,4,4,0,0,0,4,0,4,4,0,4,4,4,0,0,0,0,4,4,4,0,0,4,4,0,4,0,4,0,0,4,4,0,4,4,4,0,0,0,0,4,4,4,0,0,0,0,0,4,0,0,0,4,0,4,4,0,4,4,0,0,4,0,0,4,4,0,4,4,0,4,4,0,0,0,0,4,4,0,4,4,0,0,0,4,0,0,4,4,4,4,0,4,4,4,0,4,0,4,0,0,0,4,0,0,0,4,4,4,0,0,4,0,0,4,0,4,0,4,4,4,0,4,4,0,4,4,0,4,0,4,4,4,0,4,0,4,4,0,0,4,0,0,4,4,4,4,0,0,4,4,4,4,4,0,0,0,0,0,0,0,4,4,4,4,0,0,4,4,4,4,4,4,0,4,4,0,4,4,0,4,0,0,0,0,4,4,0,0,0,0,0,0,0,0,4,4,4,0,0,0,4,4,4,4,0,0,0,0,4,4,4,4,4,0,0,0,4,4,4,0,0,0,0,0,0,0,4,0,4,4,0,0,4,0,4,0,4,4,0,0,4,0,0,4,4,0,0,0,4,0,0,4,4,0,4,4,0,4,4,4,0,4,0,4,4,0,4,0,4,0,4,0,0,4,0,0,0,4,0,4,0,0,0,4,4,0,4,0,0,0,4,0,4,0,4,0,0,4,0,0,4,4,0,4,4,4,4,0,0,4,4,4,0,4,4,0,4,4,0,0,4,0,4,0,4,4,4,4,0,4,4,0,4,0,4,0,0,4,4,0,4,4,0,0,0,0,4,0,0,0,4,4,4,4,0,0,4,0,0,4,0,0,0,0,0,4,4,0,4,0,0,0,4,0,0,4,0,4,0,0,4,0,4,4,0,0,4,0,4,0,0,4,4,4,4,4,4,0,4,0,4,4,0,4,4,4,4,0,4,4,4,0,0,0,0,4,4,4,4,4,4,0,0,4,0,4,0,0,0,4,4,4,4,0,4,4,0,4,0,0,4,4,4,0,0,0,0,0,0,4,4,4,0,4,4,4,0,4,4,4,0,0,0,4,0,0,0,4,4,0,4,0,4,4,0,0,0,0,0,0,0,4,0,4,0,4,4,4,4,0,4,0,0,0,4,0,0,4,4,4,0,4,0,0,4,4,0,4,0,0,0,0,4,4,0,4,4,4,0,0,4,4,4,4,4,0,4,0,0,4,4,4,4,4,4,4,4,0,4,0,4,0,0,0,4,0,0,4,4,0,4,4,4,0,0,4,0,4,4,4,4,4,4,0,0,4,0,4,0,0,4,0,4,4,0,4,0,4,4,4,0,0,4,0,0,4,0,0,4,4,4,0,0,4,0,4,4,4,4,0,0,0,0,0,4,4,0,0,0,4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,4,4,0,0,0,0,0,0,4,0,0,0,4,4,4,4,0,4,4,0,4,4,4,0,4,0,0,0,0,0,0,0,4,0,4,0,4,4,4,4,4,4,0,0,0,4,0,4,0,0,0,0,4,0,4,0,0,4,4,0,4,0,4,4,0,4,4,4,4,0,4,0,4,4,4,4,0,4,4,4,0,0,4,4,4,0,4,0,4,4,0,4,4,0,0,0,0,4,0,0,4,0,0,0,0,4,0,4,0,4,0,4,0,0,4,0,0,4,0,4,0,4,4,4,0,0,0,0,4,4,4,4,4,0,4,4,4,4,4,4,0,0,0,4,4,4,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,4,4,4,0,0,4,0,0,4,0,4,4,0,4,4,0,4,4,4,0,4,0,0,0,4,0,4,0,4,0,0,4,0,0,4,4,4,0,0,4,0,4,4,4,4,0,0,4,0,4,4,0,0,0,0,0,4,4,4,4,0,4,4,0,0,4,0,0,4,0,0,0,0,0,4,0,0,4,0,0,4,0,0,4,4,4,0,4,0,0,4,0,4,4,4,4,0,0,4,0,0,4,4,4,4,0,0,4,0,0,4,0,4,0,4,4,0,0,0,0,4,0,0,0,0,4,4,4,0,4,0,4,4,4,0,0,0,4,0,0,4,0,4,4,4,0,0,0,0,0,4,0,4,4,4,4,0,4,0,4,0,4,0,0,4,4,4,4,0,4,4,0,0,4,4,0,0,0,4,4,0,4,4,0,4,4,0,0,4,0,0,4,4,4,4,0,4,4,4,4,4,4,0,0,0,4,0,0,0,4,0,4,0,0,4,0,4,0,4,4,4,4,0,4,0,4,0,0,0,0,4,0,0,0,0,4,0,0,0,4,0,4,4,4,0,0,4,4,0,4,0,0,0,4,0,0,4,0,0,4,4,0,4,0,4,4,4,0,4,4,0,4,4,0,0,0,4,4,0,4,4,4,4,0,0,4,0,4,4,0,4,0,4,0,4,4,4,4,0,4,4,4,0,4,4,4,4,4,0,4,4,0,0,0,4,0,4,4,0,0,4,4,4,0,4,4,0,4,4,4,4,4,4,0,4,0,4,0,4,0,0,4,4,4,4,4,0,0,0,0,4,4,4,0,4,0,0,4,4,4,4,4,4,4,0,0,0,0,4,0,4,0,4,0,4,4,4,4,0,0,0,0,0,4,4,0,4
	};
	double ydata[] = {
		0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
	};
	double outDdata[] = {
		2, 6, 0, 4, 2, 6, 0, 6, 0, 4, 2, 0, 6, 2, 0, 0, 6, 4, 2, 2, 6, 4, 0, 2, 4, 6, 6, 0, 4, 2, 2, 6, 0, 6, 2, 0, 0, 6, 6, 6
	};
	
	gsl_vector_view DvectorView = gsl_vector_view_array(Ddata, state.k);
	state.D = &DvectorView.vector;
	gsl_matrix_view JmatrixView = gsl_matrix_view_array(Jdata, state.k, state.k);
	state.J = &JmatrixView.matrix;
	gsl_vector_view yVectorView = gsl_vector_view_array(ydata, state.k);
	gsl_vector *y;
	y = &yVectorView.vector;
	gsl_vector_view outDVectorView = gsl_vector_view_array(outDdata, state.k);
	gsl_vector *outD;
	outD = &outDVectorView.vector;
	int outQ = 3;
	
	updateQD(&state, y);
	
	int isDWorking = 1;
	for(int i=0;i<state.k;i++){
		if(gsl_vector_get(state.D, i) != gsl_vector_get(outD, i)){
			isDWorking = 0;
			break;
		}
	}
	printf("D %s\n",isDWorking>0?"works":"fails");
	
	int isQWorking = state.Q == outQ ? 1 : 0;
	printf("Q %s\n",isQWorking>0?"works":"fails");
	
	printf("----------------------");
	
	//free memory
	//gsl_vector_free(state.D);
	//gsl_vector_free(outD);
	//gsl_vector_free(y);
	//gsl_matrix_free(state.J);
}

void testEvalW(){
	printf("\nTest evalW:\n");
	
	gsl_complex ans;
	int eps = 1;
	int p = 2;
	int m = 3;
	
	evalW(&ans, eps, p, m);
	
	int isWorking = 1;
	if(fabs(GSL_REAL(ans) - (-1.414214)) > 0.00001
		|| fabs(GSL_IMAG(ans) - (1.414214)) > 0.00001){
			isWorking = 0;
		}
	printf("evalW %s\n",isWorking>0?"works":"fails");
	
	printf("----------------------");
}

void testGamma(){
	printf("\nTest Gamma:\n");
	
	int eps, p, m, A = 4, B = 6;
	Gamma(&eps, &p, &m, A, B);
	
	int isEpsWorking = 1;
	if(eps != 1){
		isEpsWorking = 0;
	}
	int isPWorking = 1;
	if(p != 2){
		isPWorking = 0;
	}
	int isMWorking = 1;
	if(m != 6){
		isMWorking = 0;
	}
	
	printf("eps %s\n",isEpsWorking>0?"works":"fails");
	printf("p %s\n",isPWorking>0?"works":"fails");
	printf("m %s\n",isMWorking>0?"works":"fails");
	
	printf("----------------------");
}

void testPartialGamma(){
	printf("\nTest partialGamma:\n");
	
	int eps, p, m, A = 6;
	partialGamma(&eps, &p, &m, A);
	
	int isEpsWorking = 1;
	if(eps != 1){
		isEpsWorking = 0;
	}
	int isPWorking = 1;
	if(p != 1){
		isPWorking = 0;
	}
	int isMWorking = 1;
	if(m != 7){
		isMWorking = 0;
	}
	
	printf("eps %s\n",isEpsWorking>0?"works":"fails");
	printf("p %s\n",isPWorking>0?"works":"fails");
	printf("m %s\n",isMWorking>0?"works":"fails");
	
	printf("----------------------");
}

void testExponentialSum(){
	printf("\nTest exponentialSum:\n");
	struct StabilizerStates state;
	
	state.n = 18;
	state.k = 18;
	state.Q = 0;
	double Ddata[] = {
		0,4,6,4,2,0,0,4,4,4,0,6,0,0,2,2,0,6
	};
	double Jdata[] = {
		0,0,4,0,4,4,0,0,0,4,4,0,4,4,0,4,4,0,0,0,4,0,0,4,0,0,4,4,4,0,0,4,4,0,4,0,4,4,4,0,0,4,0,4,4,0,0,4,4,0,0,0,4,4,0,0,0,0,4,4,0,0,0,4,0,0,0,4,0,4,4,0,4,0,0,4,4,0,0,0,4,4,0,4,4,4,0,0,4,0,4,4,4,4,0,0,0,4,0,0,4,4,0,4,4,0,0,0,0,0,0,0,0,0,0,0,4,4,0,4,4,0,0,4,4,4,0,0,4,0,0,4,0,0,0,0,4,0,0,0,4,4,4,4,0,4,4,0,4,0,4,0,0,0,0,0,4,0,0,0,4,0,4,4,0,4,4,0,4,0,0,0,4,4,0,4,4,0,4,0,4,4,0,0,0,4,0,4,0,4,0,0,4,0,0,0,4,4,0,0,4,0,4,4,4,0,0,4,0,4,0,4,4,0,0,4,4,0,4,0,4,0,4,0,4,0,4,0,0,0,0,4,0,4,4,4,0,4,4,4,0,0,0,4,0,4,0,0,0,4,0,4,0,4,0,0,0,4,0,4,0,4,0,4,0,0,4,4,0,4,4,0,0,4,0,0,4,4,0,0,0,0,4,4,4,4,4,0,4,4,4,4,4,0,4,4,4,4,4,0,0,0,0,4,0,0,0,0,4,0,0,0,4,4,0,0,4,4,4,4,4,0,0,4
	};
	
	gsl_vector_view DvectorView = gsl_vector_view_array(Ddata, state.n);
	state.D = &DvectorView.vector;
	gsl_matrix_view JmatrixView = gsl_matrix_view_array(Jdata, state.n, state.n);
	state.J = &JmatrixView.matrix;
	int outEps = 1, outP = 18, outM = 4, eps, p, m;
	gsl_complex ans;
	
	exponentialSum(&state, &eps, &p, &m, &ans, 1);
	
	int isEpsWorking = 1;
	if(eps != outEps){
		isEpsWorking = 0;
	}
	int isPWorking = 1;
	if(p != outP){
		isPWorking = 0;
	}
	int isMWorking = 1;
	if(m != outM){
		isMWorking = 0;
	}
	
	printf("eps %s\n",isEpsWorking>0?"works":"fails");
	printf("p %s\n",isPWorking>0?"works":"fails");
	printf("m %s\n",isMWorking>0?"works":"fails");
	
	printf("----------------------");
}

void testShrink(){
	
	printf("\nTest shrink:\n");
	struct StabilizerStates state;

	state.n = 40;
	state.k = 40;
	state.Q = 3;
	double hdata[] = {
		0,0,0,1,0,0,1,0,0,1,1,1,1,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,1,0,1,1
	};
	double Gdata[] = {
		1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
	};
	double Gbardata[] = {
		1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1
	};
	double Ddata[] = {
		2,6,0,4,4,0,6,0,0,4,2,0,6,2,0,0,6,4,2,2,6,4,0,2,2,0,4,6,4,2,2,6,0,6,2,0,0,6,6,0
	};
	double Jdata[] = {
		4,0,4,4,0,4,0,4,4,4,4,0,0,4,0,0,4,4,4,4,4,0,0,0,4,4,0,0,4,0,4,0,0,4,0,0,0,0,4,0,0,4,0,4,4,0,0,4,4,4,0,0,4,4,4,4,0,0,0,0,0,0,4,0,4,4,0,0,0,4,4,4,4,0,0,4,0,4,0,0,4,0,0,0,4,4,0,0,4,0,0,0,4,0,0,4,0,4,4,4,0,0,4,4,4,0,0,0,4,0,4,0,0,0,0,0,0,4,4,0,4,4,0,0,4,4,4,0,4,4,0,0,0,4,4,0,0,4,0,0,4,4,4,0,0,0,4,4,0,4,4,0,0,4,4,0,4,0,4,0,0,4,4,4,0,4,4,0,4,0,0,0,0,4,0,0,0,0,0,0,0,0,0,4,4,4,0,0,0,4,4,4,0,0,0,4,4,0,0,0,4,0,4,4,4,0,4,0,0,0,0,0,0,0,4,0,0,4,4,4,4,0,4,0,0,0,0,0,0,4,4,0,0,4,4,4,4,0,0,4,0,0,0,4,4,4,4,4,4,0,4,0,4,4,4,4,0,4,0,4,0,0,4,0,0,4,4,4,0,0,4,0,0,0,4,4,4,4,0,0,4,4,0,0,0,0,4,0,0,4,4,4,4,4,0,0,4,4,0,0,0,0,4,0,0,0,0,0,0,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,0,4,0,0,4,4,0,4,0,0,0,0,0,4,4,4,4,0,0,4,0,0,0,0,0,0,4,0,4,4,0,4,0,0,4,4,4,0,4,0,0,0,4,4,0,0,4,0,4,0,0,0,4,4,4,0,0,0,4,0,0,0,0,4,4,0,4,0,0,0,4,0,4,4,4,4,0,0,0,0,0,4,4,4,0,4,4,0,0,4,0,0,0,0,0,4,0,0,0,4,0,4,4,4,0,0,0,4,0,4,4,0,4,4,4,0,0,0,0,0,0,0,4,0,4,4,0,4,0,4,0,0,4,4,0,4,4,4,0,4,4,4,4,4,4,0,0,0,0,0,4,0,0,0,4,0,4,4,0,0,0,4,4,4,0,0,4,4,0,4,4,0,4,4,0,0,0,0,4,0,4,0,4,0,0,0,4,0,0,4,4,4,4,0,4,4,4,0,4,4,0,4,4,0,4,0,0,0,4,4,4,0,0,4,0,0,4,0,4,4,0,0,4,0,4,4,0,4,4,0,4,0,4,4,4,0,4,0,4,0,4,4,0,0,0,4,4,4,4,0,0,4,4,4,4,4,0,0,0,4,4,4,4,4,4,4,4,0,0,4,4,4,4,4,0,0,4,4,0,0,0,4,0,0,0,0,0,4,4,0,0,0,0,0,0,0,0,4,4,0,4,4,4,4,4,4,4,0,0,0,0,4,4,4,0,4,0,0,0,0,0,0,4,0,0,0,0,0,0,4,0,4,4,0,0,4,0,4,0,0,0,4,4,4,0,0,4,4,0,0,0,4,0,0,0,4,0,4,4,0,4,4,4,0,4,0,4,4,0,4,0,4,0,4,0,0,4,0,0,0,4,0,0,0,0,0,4,4,0,4,0,0,0,4,4,4,0,4,0,0,4,0,0,4,4,0,4,4,4,4,0,0,4,4,4,0,4,4,0,4,4,0,0,4,0,4,0,4,4,4,4,0,4,4,0,4,0,4,0,0,4,4,0,4,4,0,0,0,0,4,0,0,0,4,4,4,4,0,0,4,0,0,0,0,0,0,0,0,4,4,0,4,0,0,4,4,0,0,4,0,4,0,0,4,0,4,4,0,0,4,0,4,0,0,4,4,4,4,4,4,0,4,0,4,4,0,4,4,4,4,0,4,4,4,0,0,0,0,4,0,0,0,0,4,0,0,4,0,4,0,0,0,4,4,4,4,0,4,4,4,0,4,4,4,4,4,0,0,0,0,0,0,4,4,0,0,4,4,4,0,4,4,4,0,0,0,4,0,0,0,4,4,0,4,0,4,4,0,0,0,0,0,0,0,4,0,4,0,4,4,4,4,0,4,0,0,0,4,0,4,0,0,0,0,4,0,0,4,4,0,4,0,0,0,0,4,4,0,4,0,0,4,4,4,4,4,4,4,0,4,0,0,4,4,0,4,4,4,0,4,0,0,0,4,0,4,4,0,4,4,0,0,0,4,4,4,4,0,0,4,4,0,4,0,0,0,0,0,4,0,4,4,4,4,0,4,4,0,0,4,0,4,0,0,0,0,4,4,0,4,4,0,4,4,0,0,0,0,0,4,0,4,0,4,4,4,0,4,4,0,0,4,0,4,4,0,0,0,4,0,0,4,0,0,0,4,4,0,0,4,4,4,0,0,0,4,4,0,4,0,4,0,4,4,0,0,0,4,4,4,0,0,4,0,0,0,0,0,4,0,0,4,0,0,0,4,4,4,4,4,4,4,0,0,0,0,4,0,4,4,0,4,4,4,4,4,0,0,0,0,0,4,4,0,4,4,0,4,0,0,0,0,0,0,4,4,4,0,0,4,4,4,0,4,0,4,4,0,4,0,4,4,4,0,4,0,0,4,0,0,0,0,4,0,0,0,4,0,4,4,4,0,4,0,4,0,4,0,4,4,4,0,0,0,0,4,4,4,4,0,4,0,4,4,4,4,4,0,0,0,4,4,4,4,4,4,4,4,4,4,4,4,4,0,0,0,0,0,4,4,4,0,0,4,0,0,4,0,4,0,4,0,4,0,4,4,4,0,4,0,0,0,4,0,4,0,4,0,0,4,0,0,4,4,4,0,0,4,0,4,4,4,4,0,0,4,0,4,4,0,0,0,0,0,4,4,4,4,0,4,4,0,0,4,0,0,4,0,0,0,0,0,4,0,0,4,0,0,4,0,0,4,4,4,0,4,0,0,4,0,4,4,0,4,0,0,4,0,0,4,4,4,4,0,4,4,0,0,4,0,4,0,4,4,0,0,0,0,4,0,0,0,0,4,4,4,0,4,0,4,4,4,0,0,0,4,0,0,4,0,4,4,4,0,0,0,0,0,4,0,4,4,4,4,0,4,0,4,0,4,0,0,4,4,4,4,0,4,4,0,0,4,0,0,0,0,4,4,0,4,4,0,4,4,4,0,4,0,0,4,4,4,4,0,4,4,4,4,4,4,0,0,0,4,0,0,0,4,0,4,0,0,0,0,4,0,4,4,4,4,0,4,0,4,4,0,0,0,4,4,4,4,4,4,0,0,0,4,0,4,4,4,0,0,4,4,0,4,0,4,4,0,4,0,4,0,0,4,4,0,4,0,4,4,0,0,4,4,0,0,0,4,4,0,4,4,0,4,4,4,4,0,0,4,0,4,4,0,4,4,0,4,4,4,4,4,0,4,4,4,0,4,4,4,4,4,0,4,4,0,0,0,4,0,4,4,0,0,4,4,4,0,4,4,0,4,4,4,4,4,4,0,0,0,4,0,4,0,0,4,4,4,4,4,4,0,0,0,0,0,4,0,4,4,4,4,4,4,4,0,0,0,4,0,4,0,0,0,0,0,4,0,4,0,4,4,0,4,0,4,4,0,4,4,0
	};
	double xidata[] = {
		0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0
	};
	int alpha = 1;
	int lazy = 0;
	
	int outk = 39;
	int outQ = 3;
	int outStatus = 2;
	double outhdata[] = {
		0,0,0,1,0,0,1,0,0,1,1,1,1,0,0,1,1,0,0,1,0,1,1,1,1,1,1,1,1,1,0,1,0,0,0,0,1,0,1,1
	};
	double outGdata[] = {
		1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0
	}; 
	double outGbardata[] = {
		1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0
	};
	double outDdata[] = {
		2,6,0,4,2,6,0,6,0,4,2,0,6,2,0,0,6,4,2,2,6,4,0,2,4,6,6,0,4,2,2,6,0,6,2,0,0,6,6,6
	};
	double outJdata[] = {
		4,0,4,4,0,4,0,4,4,4,4,0,0,4,0,0,4,4,4,4,4,0,0,0,4,4,0,0,4,0,4,0,0,4,0,0,0,0,4,0,0,4,0,4,4,0,0,4,4,4,0,0,4,4,4,4,0,0,0,0,0,0,4,0,4,4,0,0,0,4,4,4,4,0,0,4,0,4,0,0,4,0,0,0,4,4,0,0,4,0,0,0,4,0,0,4,0,4,4,4,0,0,4,4,4,0,0,0,4,0,4,0,0,0,0,0,0,4,4,0,4,4,0,0,0,0,0,4,4,4,0,0,0,4,4,0,0,4,0,0,4,4,4,0,4,4,0,0,0,4,4,0,0,4,4,0,4,0,4,4,0,4,4,0,4,0,4,4,4,0,4,4,4,0,4,4,4,0,0,0,0,4,0,0,4,0,0,4,4,0,0,4,0,0,0,4,0,4,0,4,4,0,4,0,0,4,4,4,0,0,4,4,4,4,0,4,4,4,4,4,4,4,4,4,0,4,0,0,4,0,0,0,0,4,4,4,0,4,0,4,0,0,0,0,4,4,0,4,4,0,0,4,0,0,0,0,4,4,0,4,0,4,4,4,4,4,0,4,4,4,0,0,0,0,4,4,0,0,0,0,4,4,0,4,4,4,4,4,0,4,0,0,0,0,4,4,0,4,0,0,0,4,4,4,0,4,0,0,4,0,0,4,4,4,4,4,0,0,4,4,4,4,4,4,4,0,4,0,0,4,4,0,4,0,0,0,0,0,4,4,4,4,0,0,4,0,0,4,0,0,0,4,0,4,4,0,4,0,0,0,4,4,0,4,0,0,0,4,4,0,0,4,0,4,0,0,0,4,4,4,0,0,0,4,0,0,0,4,4,4,0,4,0,0,0,4,0,4,4,0,4,0,0,0,4,4,0,0,4,0,4,4,0,0,4,0,0,0,0,0,4,0,0,0,0,4,0,4,4,0,0,0,4,0,4,4,0,4,4,4,0,0,0,0,4,4,4,0,0,4,4,0,4,0,4,0,0,4,4,0,4,4,4,0,0,0,0,4,4,4,0,0,0,0,0,4,0,0,0,4,0,4,4,0,4,4,0,0,4,0,0,4,4,0,4,4,0,4,4,0,0,0,0,4,4,0,4,4,0,0,0,4,0,0,4,4,4,4,0,4,4,4,0,4,0,4,0,0,0,4,0,0,0,4,4,4,0,0,4,0,0,4,0,4,0,4,4,4,0,4,4,0,4,4,0,4,0,4,4,4,0,4,0,4,4,0,0,4,0,0,4,4,4,4,0,0,4,4,4,4,4,0,0,0,0,0,0,0,4,4,4,4,0,0,4,4,4,4,4,4,0,4,4,0,4,4,0,4,0,0,0,0,4,4,0,0,0,0,0,0,0,0,4,4,4,0,0,0,4,4,4,4,0,0,0,0,4,4,4,4,4,0,0,0,4,4,4,0,0,0,0,0,0,0,4,0,4,4,0,0,4,0,4,0,4,4,0,0,4,0,0,4,4,0,0,0,4,0,0,4,4,0,4,4,0,4,4,4,0,4,0,4,4,0,4,0,4,0,4,0,0,4,0,0,0,4,0,4,0,0,0,4,4,0,4,0,0,0,4,0,4,0,4,0,0,4,0,0,4,4,0,4,4,4,4,0,0,4,4,4,0,4,4,0,4,4,0,0,4,0,4,0,4,4,4,4,0,4,4,0,4,0,4,0,0,4,4,0,4,4,0,0,0,0,4,0,0,0,4,4,4,4,0,0,4,0,0,4,0,0,0,0,0,4,4,0,4,0,0,0,4,0,0,4,0,4,0,0,4,0,4,4,0,0,4,0,4,0,0,4,4,4,4,4,4,0,4,0,4,4,0,4,4,4,4,0,4,4,4,0,0,0,0,4,4,4,4,4,4,0,0,4,0,4,0,0,0,4,4,4,4,0,4,4,0,4,0,0,4,4,4,0,0,0,0,0,0,4,4,4,0,4,4,4,0,4,4,4,0,0,0,4,0,0,0,4,4,0,4,0,4,4,0,0,0,0,0,0,0,4,0,4,0,4,4,4,4,0,4,0,0,0,4,0,0,4,4,4,0,4,0,0,4,4,0,4,0,0,0,0,4,4,0,4,4,4,0,0,4,4,4,4,4,0,4,0,0,4,4,4,4,4,4,4,4,0,4,0,4,0,0,0,4,0,0,4,4,0,4,4,4,0,0,4,0,4,4,4,4,4,4,0,0,4,0,4,0,0,4,0,4,4,0,4,0,4,4,4,0,0,4,0,0,4,0,0,4,4,4,0,0,4,0,4,4,4,4,0,0,0,0,0,4,4,0,0,0,4,4,4,0,0,0,0,0,0,0,0,0,0,0,0,4,4,0,0,0,0,0,0,4,0,0,0,4,4,4,4,0,4,4,0,4,4,4,0,4,0,0,0,0,0,0,0,4,0,4,0,4,4,4,4,4,4,0,0,0,4,0,4,0,0,0,0,4,0,4,0,0,4,4,0,4,0,4,4,0,4,4,4,4,0,4,0,4,4,4,4,0,4,4,4,0,0,4,4,4,0,4,0,4,4,0,4,4,0,0,0,0,4,0,0,4,0,0,0,0,4,0,4,0,4,0,4,0,0,4,0,0,4,0,4,0,4,4,4,0,0,0,0,4,4,4,4,4,0,4,4,4,4,4,4,0,0,0,4,4,4,4,4,4,4,4,4,0,0,0,0,0,0,0,0,0,4,4,4,0,0,4,0,0,4,0,4,4,0,4,4,0,4,4,4,0,4,0,0,0,4,0,4,0,4,0,0,4,0,0,4,4,4,0,0,4,0,4,4,4,4,0,0,4,0,4,4,0,0,0,0,0,4,4,4,4,0,4,4,0,0,4,0,0,4,0,0,0,0,0,4,0,0,4,0,0,4,0,0,4,4,4,0,4,0,0,4,0,4,4,4,4,0,0,4,0,0,4,4,4,4,0,0,4,0,0,4,0,4,0,4,4,0,0,0,0,4,0,0,0,0,4,4,4,0,4,0,4,4,4,0,0,0,4,0,0,4,0,4,4,4,0,0,0,0,0,4,0,4,4,4,4,0,4,0,4,0,4,0,0,4,4,4,4,0,4,4,0,0,4,4,0,0,0,4,4,0,4,4,0,4,4,0,0,4,0,0,4,4,4,4,0,4,4,4,4,4,4,0,0,0,4,0,0,0,4,0,4,0,0,4,0,4,0,4,4,4,4,0,4,0,4,0,0,0,0,4,0,0,0,0,4,0,0,0,4,0,4,4,4,0,0,4,4,0,4,0,0,0,4,0,0,4,0,0,4,4,0,4,0,4,4,4,0,4,4,0,4,4,0,0,0,4,4,0,4,4,4,4,0,0,4,0,4,4,0,4,0,4,0,4,4,4,4,0,4,4,4,0,4,4,4,4,4,0,4,4,0,0,0,4,0,4,4,0,0,4,4,4,0,4,4,0,4,4,4,4,4,4,0,4,0,4,0,4,0,0,4,4,4,4,4,0,0,0,0,4,4,4,0,4,0,0,4,4,4,4,4,4,4,0,0,0,0,4,0,4,0,4,0,4,4,4,4,0,0,0,0,0,4,4,0,4
	};
	
	gsl_vector_view hvectorView = gsl_vector_view_array(hdata, state.n);
	state.h = &hvectorView.vector;
	gsl_vector_view DvectorView = gsl_vector_view_array(Ddata, state.n);
	state.D = &DvectorView.vector;
	gsl_vector_view xiVectorView = gsl_vector_view_array(xidata, state.n);
	gsl_vector *xi;
	xi = &xiVectorView.vector;
	
	gsl_matrix_view GmatrixView = gsl_matrix_view_array(Gdata, state.n, state.n);
	state.G = &GmatrixView.matrix;
	gsl_matrix_view GbarmatrixView = gsl_matrix_view_array(Gbardata, state.n, state.n);
	state.Gbar = &GbarmatrixView.matrix;
	gsl_matrix_view JmatrixView = gsl_matrix_view_array(Jdata, state.n, state.n);
	state.J = &JmatrixView.matrix;
	
	gsl_vector_view outhVectorView = gsl_vector_view_array(outhdata, state.n);
	gsl_vector *outh;
	outh = &outhVectorView.vector;
	gsl_vector_view outDVectorView = gsl_vector_view_array(outDdata, state.n);
	gsl_vector *outD;
	outD = &outDVectorView.vector;
	
	gsl_matrix_view outGMatrixView = gsl_matrix_view_array(outGdata, state.n, state.n);
	gsl_matrix *outG;
	outG = &outGMatrixView.matrix;
	gsl_matrix_view outGbarMatrixView = gsl_matrix_view_array(outGbardata, state.n, state.n);
	gsl_matrix *outGbar;
	outGbar = &outGbarMatrixView.matrix;
	gsl_matrix_view outJMatrixView = gsl_matrix_view_array(outJdata, state.n, state.n);
	gsl_matrix *outJ;
	outJ = &outJMatrixView.matrix;
	
	int status = shrink(&state, xi, alpha, lazy);
	
	printf("out status %s\n",status==outStatus?"works":"fails");
	printf("k %s\n",state.k==outk?"works":"fails");
	printf("Q %s\n",state.Q==outQ?"works":"fails");
	
	if(outStatus != 2){
		return;
	}
	
	int ishWorking = 1;
	for(int i=0;i<state.k+1;i++){
		if(gsl_vector_get(state.h, i) != gsl_vector_get(outh, i)){
			ishWorking = 0;
			break;
		}
	}
	printf("h %s\n",ishWorking>0?"works":"fails");
	
	int isDWorking = 1;
	for(int i=0;i<state.k;i++){
		if(gsl_vector_get(state.D, i) != gsl_vector_get(outD, i)){
			isDWorking = 0;
			break;
		}
	}
	printf("D %s\n",isDWorking>0?"works":"fails");
	
	int isGWorking = 1;
	for(int i=0;i<state.k+1;i++){
		for(int j=0;j<state.k+1;j++){
			if(gsl_matrix_get(state.G, i, j) != gsl_matrix_get(outG, i, j)){
				isGWorking = 0;
				break;
			}
		}
	}
	printf("G %s\n",isGWorking>0?"works":"fails");
	
	int isGbarWorking = 1;
	for(int i=0;i<state.k+1;i++){
		for(int j=0;j<state.k+1;j++){
			if(gsl_matrix_get(state.Gbar, i, j) != gsl_matrix_get(outGbar, i, j)){
				isGbarWorking = 0;
				break;
			}
		}
	}
	printf("Gbar %s\n",isGbarWorking>0?"works":"fails");
	
	int isJWorking = 1;
	for(int i=0;i<state.k;i++){
		for(int j=0;j<state.k;j++){
			if(gsl_matrix_get(state.J, i, j) != gsl_matrix_get(outJ, i, j)){
				isJWorking = 0;
				break;
			}
		}
	}
	printf("J %s\n",isJWorking>0?"works":"fails");
	
	printf("----------------------");
}

//TODO: testInnerProduct

//TODO: test randomStabilizerState

int main(){
	
	/*struct StabilizerStates state;
	state.n = 5;
	state.k = 4;
	state.Q = 3;
	double hdata[] = {
		1,0,1,0,0
	};
	double Ddata[] = {
		2,0,6,6,0
	};
	double xidata[] = {
		0,1,1,0,0
	};
	double Gdata[] = {
		1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1
	};
	double Gbardata[] = {
		1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,1
	};
	double Jdata[] = {
		4,0,0,0,0,4,0,0,0,0,4,0,4,0,0,4,0,0,4,0,0,0,0,0,0
	};
	gsl_vector_view hvectorView = gsl_vector_view_array(hdata, state.n);
	state.h = &hvectorView.vector;
	gsl_vector_view DvectorView = gsl_vector_view_array(Ddata, state.n);
	state.D = &DvectorView.vector;
	gsl_vector_view xivectorView = gsl_vector_view_array(xidata, state.n);
	gsl_vector *xi;
	xi = &xivectorView.vector;
	gsl_matrix_view GMatrixView = gsl_matrix_view_array(Gdata, state.n, state.n);
	state.G = &GMatrixView.matrix;
	gsl_matrix_view GbarMatrixView = gsl_matrix_view_array(Gbardata, state.n, state.n);
	state.Gbar = &GbarMatrixView.matrix;
	gsl_matrix_view JMatrixView = gsl_matrix_view_array(Jdata, state.n, state.n);
	state.J = &JMatrixView.matrix;
	
	shrink(&state, xi, 1, 0);
	
	printf("\n----------answer-----------\n");
	printf("\nD: ");
	for(int i=0;i<state.n;i++){
		printf("%f ", gsl_vector_get(state.D, i));
	}
	printf("\nh: ");
	for(int i=0;i<state.n;i++){
		printf("%f ", gsl_vector_get(state.h, i));
	}
	printf("\nQ: %d", state.Q);*/
	
	//return 0;
	
	testUpdateDJ();
	testUpdateQD();
	testEvalW();
	testGamma();
	testPartialGamma();
	testExponentialSum();
	testShrink();
	return 0;
}