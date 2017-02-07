#include <math.h>
#include "stabilizer/stabilizer.h"

/**************************** binrep *******************************/
char *binrep (unsigned int val, char *buff, int sz) {
    char *pbuff = buff;

    /* Must be able to store one character at least. */
    if (sz < 1) return NULL;

    /* Special case for zero to ensure some output. */
    if (val == 0) {
        for (int i=0; i<sz; i++) *pbuff++ = '0';
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
    while (sz-- != 0) *pbuff-- = '0';
    return pbuff+1;
}


/**************************** prepH *******************************/
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

    gsl_vector_free(tempVector);
    return phi;
}


/**************************** prepL *******************************/
struct StabilizerState* prepL(int i, int t, gsl_matrix* L) {
    //compute bitstring by adding rows of l
    int k = L->size1;
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

	struct StabilizerState *phi = allocStabilizerState(t, t);
	
	//construct state using shrink
	for(int xtildeidx=0;xtildeidx<t;xtildeidx++){
		gsl_vector_set_zero(tempVector);
		gsl_vector_set(tempVector, xtildeidx, 1);
		
        if((int)gsl_vector_get(bitstring, xtildeidx) == 0){
			shrink(phi, tempVector, 0, 0); // |0> at index, inner prod with 1 is 0
		}
        // |+> at index -> do nothing
	}
	
    gsl_vector_free(bitstring);
    gsl_vector_free(tempVector);
    gsl_vector_free(zeroVector);
    return phi;
}
