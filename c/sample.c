#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_linalg.h>
#include "stabilizer.h"


// Main: read inputs from stdin, and sample an inner product
// Arguments:
// <parallel:1 or 0>
// <NSamples> 
// <Norm>
// p
// zzz
// xxx
// p
// zzz
// xxx
// p
// zzz
// xxx
// L L L
// L L L
// L L L

void decompose(int *isProbability, gsl_matrix *L, double *norm, const int t, const double delta, const short exact){
	
	//trivial case
	if(t == 0){
		L = gsl_matrix_alloc(0, 0);
		*norm = 1;
		return;
	}
	
	double v = cos(M_PI/8);

    int k = ceil(1 - 2*t*log2(v) - log2(delta));
	
	//can achieve k = t/2 by pairs of stabilizer states
	if(k > t/2 || exact){
		
		*isProbability = 1;
		
		*norm = pow(2, t/2);
		if(t%2){
			*norm *= 2*v;
		}
		
		//printf("\nk > t/2 case");
		
		return;
	}
	
	//printf("\nk: %d", k);
	
	*isProbability = 0;
	
	double innerProduct=0;
	int count = 0;
	
	srand(time(NULL));
	
	gsl_matrix *U, *V;
	gsl_vector *S, *work;
	L = gsl_matrix_alloc(k, t);
	U = gsl_matrix_alloc(t, k);
	V = gsl_matrix_alloc(k, k);  
	S = gsl_vector_alloc(k); 
	work = gsl_vector_alloc(k);
	
	double Z_L;
	
	while(innerProduct < 1-delta){
		
		count++;
		//printf("\nrank attempt: %d", count);
		
		for(int i=0;i<k;i++){
			for(int j=0;j<t;j++){
				gsl_matrix_set(L, i, j, rand() & 1);	//set to either 0 or 1
			}
		}
		
		int rank = 0;
		//rank of a matrix is the number of non-zero values in its singular value decomposition
		gsl_matrix_transpose_memcpy(U, L);
		gsl_linalg_SV_decomp(U, V, S, work);
		for(int i=0;i<k;i++){
			if(fabs(gsl_vector_get(S, i)) > 0.00001){
				rank++;
			}
		}
		
		//check rank
		if(rank < k){
			continue;
		}
		
		//compute Z(L) = sum_x 2^{-|x|/2}
        Z_L = 0;
		
		gsl_vector *z, *x;
		z = gsl_vector_alloc(k);
		x = gsl_vector_alloc(t);
		int *zArray = (int *)calloc(k, sizeof(int));
		int currPos, currTransfer;
		for(int i=0;i<pow(2,k);i++){
			//starting from k 0s, add (binary) +1 2^k times, effectively generating all possible vectors z of length k
			//least important figure is on the very left
			currPos = 0;
			currTransfer = 1;
			while(currTransfer && currPos<k){
				*(zArray+currPos) += currTransfer;
				if(*(zArray+currPos) > 1){
					//current position overflowed -> transfer to next
					*(zArray+currPos) = 0;
					currTransfer = 1;
					currPos++;
				}
				else{
					currTransfer = 0;
				}
			}
			
			for(int i=0;i<k;i++){
				gsl_vector_set(z, i, (double)(*(zArray+currPos)));
			}
			
			gsl_blas_dgemv(CblasTrans, 1., L, z, 0., x);
			
			double temp = 0;
			for(int i=0;i<t;i++){
				temp += mod((int)gsl_vector_get(x, i), 2);
			}
			
			Z_L += pow(2, -temp/2);
		}
		
		innerProduct = pow(2,k) * pow(v, 2*t) / Z_L;
		
	}
	
	*norm = sqrt(pow(2,k) * Z_L);
	
	//printf("\ninner product: %.3lf", innerProduct);
}

static char *binrep (unsigned int val, char *buff, int sz) {
    char *pbuff = buff;

    /* Must be able to store one character at least. */
    if (sz < 1) return NULL;

    /* Special case for zero to ensure some output. */
    if (val == 0) {
        *pbuff++ = '0';
        *pbuff = '\0';
        return buff;
    }

    /* Work from the end of the buffer back. */
    pbuff += sz;
    *pbuff-- = '\0';

    /* For each bit (going backwards) store character. */
    while (val != 0) {
        if (sz-- == 0) return NULL;
        *pbuff-- = ((val & 1) == 1) ? '1' : '0';

        /* Get next bit. */
        val >>= 1;
    }
    return pbuff+1;
}

void evalLcomponent(gsl_complex *innerProduct, unsigned int i, gsl_matrix *L, struct StabilizerStates *theta, int k, int t){
	
	//compute bitstring by adding rows of l
    char buff[k+1];
	char *Lbits = binrep(i,buff,k);
	
	gsl_vector *bitstring, *tempVector, *zeroVector;
	bitstring = gsl_vector_alloc(t);
	tempVector = gsl_vector_alloc(t);
	zeroVector = gsl_vector_alloc(t);
	gsl_vector_set_zero(bitstring);
	gsl_vector_set_zero(zeroVector);
	
	int j=0;
	while(Lbits[j] != '\0'){
		if(Lbits[j] == '1'){
			gsl_matrix_get_row(tempVector, L, j);
			gsl_vector_add(bitstring, tempVector);
		}
		j++;
	}
	for(j=0;j<t;j++){
		gsl_vector_set(bitstring, j, mod(gsl_vector_get(bitstring, j), 2));
	}
	
	struct StabilizerStates *phi = (struct StabilizerStates *)malloc(sizeof(struct StabilizerStates));
	phi->n = t;
	phi->k = t;
	phi->Q = 0;
	phi->h = gsl_vector_alloc(phi->n);
	gsl_vector_set_zero(phi->h);
	phi->D = gsl_vector_alloc(phi->n);
	gsl_vector_set_zero(phi->D);
	phi->G = gsl_matrix_alloc(phi->n, phi->n);
	gsl_matrix_set_identity(phi->G);
	phi->Gbar = gsl_matrix_alloc(phi->n, phi->n);
	gsl_matrix_set_identity(phi->Gbar);
	phi->J = gsl_matrix_alloc(phi->n, phi->n);
	gsl_matrix_set_zero(phi->J);
	
	//construct state by measuring paulis
	for(int xtildeidx=0;xtildeidx<t;xtildeidx++){
		gsl_vector_set_zero(tempVector);
		tempVector[xtildeidx] = 1;
		if((int)gsl_vector_get(bitstring, xtildeidx) == 1){
			//|+> at index, so measure X
			measurePauli(phi, 0, zeroVector, tempVector);
		}
		else{
			//|0> at index, so measure Z
			measurePauli(phi, 0, tempVector, zeroVector);
		}
	}
	
	int *eps, *p, *m;
	innerProduct(theta, phi, eps, p, m, innerProduct, 0);
}

int main(int argc, char* argv[]){
	
	// int isProbability;
	// gsl_matrix *L;
	// double norm;
	// decompose(&isProbability, L, &norm, atoi(argv[1]), 0.01, 0); 
	//printf("\nnorm: %.3lf", norm);
	//decompose(&isProbability, L, &norm, atoi(argv[1]), 0.0001, 1);
	//printf("\nnorm: %.3lf", norm);
	
	return 0;
	
    // // parallel
    // int parallel;
    // scanf("%d\n", &parallel);

    // // samples
    // int samples;
    // scanf("%d\n", &samples);

    // // norm
    // double norm;
    // scanf("%lg\n", &norm);



    // if (parallel == 1) printf("Parallel\n");
    // else printf("Not parallel\n");

    // printf("Samples: %d\n", samples);
    
    // printf("Norm: %lg\n", norm);

    // return 0;
}
