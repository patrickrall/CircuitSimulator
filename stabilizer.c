#include <stdio.h>
#include <string.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>

//define K from RandomStabilizerState algorithm (page 16)
struct StabilizerStates {
   int k;
   gsl_vector *h;		//in \mathbb{F}^n_2
   gsl_matrix *G;		//in \mathbb{F}^{n\times n}_2
   gsl_matrix *Gbar;	//= (G^-1)^T
   
   //define q to be zero for all x
   int Q;				//in \mathbb{Z}_8
   gsl_vector *D;		//in {0,2,4,6}^k
   gsl_matrix *J;		//in {0,4}^{k\times k}, symmetric
};

//helper to update D, J using equations 48, 49 on page 10		
void updateDJ(struct StabilizerStates *state, gsl_matrix *R){
	
	//temporary variables for storing intermediary results
	gsl_vector *tempVector, *tempVector1;
	gsl_matrix *tempMatrix;
	tempVector = gsl_vector_alloc(state->k);
	tempVector1 = gsl_vector_alloc(state->k);
	tempMatrix = gsl_matrix_alloc(state->k, state->k);
	
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
		gsl_vector_set(state->D, i, (int)gsl_vector_get(state->D, i) % 8);
	}
	
	//equation 49
	//tempMatrix <- R * J
	gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, R, state->J, 0, tempMatrix);
	//J <- tempMatrix * R'
	gsl_blas_dgemm(CblasNoTrans, CblasTrans, 1, tempMatrix, R, 0, state->J);
	//J = J % 8
	for(int i=0;i<state->k;i++){
		for(int j=0;j<state->k;j++){
			gsl_matrix_set(state->J, i, j, (int)gsl_matrix_get(state->J, i, j) % 8);
		}
	}
	
	//free memory
	gsl_vector_free(tempVector);
	gsl_vector_free(tempVector1);
	gsl_matrix_free(tempMatrix);
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
	state->Q = state->Q % 8;
	
	//equation 52
	//D_a += J[a,:] dot y
	for(int a=0;a<state->k;a++){
		gsl_matrix_get_row(tempVector, state->J, a);
		gsl_blas_ddot(tempVector, y, &tempInt);
		gsl_vector_set(state->D, a, (int)(gsl_vector_get(state->D, a) + tempInt) % 8);
	}
	
	//free memory
	gsl_vector_free(tempVector);
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
	if(A%2==1 || B%2==1){
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
	gsl_complex gamma = gsl_complex_add(gsl_vector_complex_get(lookup, A%8), gsl_vector_complex_get(lookup, B%8));
	gamma = gsl_complex_sub(gamma, gsl_vector_complex_get(lookup, (A+B)%8));
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
	if(A%2==1){
		printf("partialGamma: A must be even!");
		return;
	}
	
	//lookup = {0: (1, 2, 0), 2: (1, 1, 1), 4: (0, 0, 0), 6: (1, 1, 7)}
	//return lookup[A % 8]
	switch(A%8){
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
		tempM = (tempM + *m) % 8;
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
		tempM = (tempM + *m) % 8;
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
	R = gsl_matrix_alloc(state->k, state->k);
	
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
			gsl_matrix_set(R, *(S+i), a, ((int)gsl_matrix_get(R, *(S+i), a) + 1) % 2);
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
	for(int k=0;k<state->k-1;k++){
		*(E + Elength++) = k;
	}
	if(Slength == 1){
		*(E + Elength++) = *(S);
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
						gsl_matrix_set(R, *(E + i), b, ((int)gsl_matrix_get(R, *(E + i), b) + 1) % 2);
					}
					if((int)gsl_matrix_get(state->J, b, *(E + i)) == 4){
						gsl_matrix_set(R, *(E + i), a, ((int)gsl_matrix_get(R, *(E + i), a) + 1) % 2);
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
			if((m1-m0)%2 == 1){
				printf("ExponentialSum: m1-m0 must be even!");
				return;
			}
			
			//Rearrange 2^{p0/2} e^{i pi m0/4} + 2^{p1/2} e^{i pi m1/4}
			//To 2^(p0/2) ( 1 + e^(i pi (m1-m0)/4)) and use partialGamma
			
			partialGamma(eps, p, m, m1-m0);
			if(eps == 0){
				*p = 0;
				*m = 0;
			}
			else{
				*p += p0;
				*m = (*m + m0) % 8;
			}
		}
	}
}

void testUpdateDJ(){
	printf("\nTest updateDJ:\n");
	struct StabilizerStates state;
	
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
	
	state.k = 18;
	state.Q = 0;
	double Ddata[] = {
		0,4,6,4,2,0,0,4,4,4,0,6,0,0,2,2,0,6
	};
	double Jdata[] = {
		0,0,4,0,4,4,0,0,0,4,4,0,4,4,0,4,4,0,0,0,4,0,0,4,0,0,4,4,4,0,0,4,4,0,4,0,4,4,4,0,0,4,0,4,4,0,0,4,4,0,0,0,4,4,0,0,0,0,4,4,0,0,0,4,0,0,0,4,0,4,4,0,4,0,0,4,4,0,0,0,4,4,0,4,4,4,0,0,4,0,4,4,4,4,0,0,0,4,0,0,4,4,0,4,4,0,0,0,0,0,0,0,0,0,0,0,4,4,0,4,4,0,0,4,4,4,0,0,4,0,0,4,0,0,0,0,4,0,0,0,4,4,4,4,0,4,4,0,4,0,4,0,0,0,0,0,4,0,0,0,4,0,4,4,0,4,4,0,4,0,0,0,4,4,0,4,4,0,4,0,4,4,0,0,0,4,0,4,0,4,0,0,4,0,0,0,4,4,0,0,4,0,4,4,4,0,0,4,0,4,0,4,4,0,0,4,4,0,4,0,4,0,4,0,4,0,4,0,0,0,0,4,0,4,4,4,0,4,4,4,0,0,0,4,0,4,0,0,0,4,0,4,0,4,0,0,0,4,0,4,0,4,0,4,0,0,4,4,0,4,4,0,0,4,0,0,4,4,0,0,0,0,4,4,4,4,4,0,4,4,4,4,4,0,4,4,4,4,4,0,0,0,0,4,0,0,0,0,4,0,0,0,4,4,0,0,4,4,4,4,4,0,0,4
	};
	
	gsl_vector_view DvectorView = gsl_vector_view_array(Ddata, state.k);
	state.D = &DvectorView.vector;
	gsl_matrix_view JmatrixView = gsl_matrix_view_array(Jdata, state.k, state.k);
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

int main(){
	testUpdateDJ();
	testUpdateQD();
	testEvalW();
	testGamma();
	testPartialGamma();
	testExponentialSum();
	return 0;
}