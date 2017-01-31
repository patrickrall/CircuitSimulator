#include <stdio.h>
#include <stdlib.h>
#include <math.h>
// #include "stabilizer/stabilizer.h"
#include "mpi.h"
#include "utils/utils.h"

struct StabilizerState* prepH(int i, int t) {
	int size = ceil((double)t/2);
	
    char buff[size+1];
	char *bits = binrep(i,buff,size);
	
	struct StabilizerState *phi = allocStabilizerState(t, t);

    // set J matrix
	for(int j=0;j<size;j++){
		if(bits[j] == '0' && !(t%2 && j==size-1)){
			gsl_matrix_set(phi->J, j*2+1, j*2, 4);
			gsl_matrix_set(phi->J, j*2, j*2+1, 4);
		}
	}

	gsl_vector *tempVector;
	tempVector = gsl_vector_alloc(t);
	
	for(int j=0;j<size;j++){
		gsl_vector_set_zero(tempVector);
		
		if(t%2 && j==size-1){
			gsl_vector_set(tempVector, t-1, 1);
		
            // bit = 0 is |+>
            // bit = 1 is |0>
			if(bits[j] == '1'){
			    shrink(phi, tempVector, 0, 0);	//|0>
			}
			
			continue;
		}
        
	    // bit = 1 corresponds to |00> + |11> state
        // bit = 0 corresponds to |00> + |01> + |10> - |11>
		if(bits[j] == '1'){
            gsl_vector_set(tempVector, j*2+1, 1);
            gsl_vector_set(tempVector, j*2, 1);
        
            shrink(phi, tempVector, 0, 0); // only 00 and 11 have inner prod 0 with 11
		}
	}

    return phi;
}

double exactProjector(struct Projector *P, double Lnorm) {
    printf("Lnorm: %f\n", Lnorm);
    printf("Projector\n");
    printProjector(P);
    // empty projector
    if (P->Nstabs == 0) return Lnorm;

    int t = P->Nqubits;

    // clifford circuit
    if (t == 0) {
        double sum = 1; // include identity
        for (int i = 0; i < P->Nstabs; i++) {
            double ph = gsl_vector_get(P->phases, i);
            if (ph == 0) sum += 1;
            if (ph == 2) sum -= 1;
        } 

        return sum/(1 + (double)P->Nstabs);
    }

    // exact inner product
    int size = pow(2, ceil((double)t / 2));
    gsl_complex total = gsl_complex_rect(0,0);
    for (int i = 0; i < size; i++) {
        struct StabilizerState* theta = prepH(i, t);

        // Project theta
        double projfactor = 1;
        gsl_vector *zeta = gsl_vector_alloc(P->Nqubits);
        gsl_vector *xi = gsl_vector_alloc(P->Nqubits);

        for (int j = 0; j < P->Nstabs; j++) {
            int m = gsl_vector_get(P->phases, i);
            gsl_matrix_get_row(zeta, P->zs, i);
            gsl_matrix_get_row(xi, P->xs, i);

            double res = measurePauli(theta, m, zeta, xi);
            projfactor *= res;

            if (res == 0) break;
        } 
        
        gsl_vector_free(zeta);
        gsl_vector_free(xi);

        if (projfactor == 0) continue;
        
        // Evaluate other components
        for (int j = 0; j < size; j++) {
            struct StabilizerState* phi = prepH(j, t);
            gsl_complex innerProd;
            
            int eps, p, m;
        	innerProduct(theta, phi, &eps, &p, &m, &innerProd, 0);
            freeStabilizerState(phi);
            total = gsl_complex_add(total, innerProd);  
        }

        total = gsl_complex_mul_real(total, projfactor);
    }

    return gsl_complex_abs(total);
}
