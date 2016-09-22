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
#include <pthread.h>

#include <gsl/gsl_errno.h>

/************* Projector struct *************/
struct Projector {
    int Nstabs;
    int Nqubits;
    gsl_vector* phases;
    gsl_matrix* xs;
    gsl_matrix* zs; 
};

/* some prototypes */
struct Projector* readProjector(void);
void decompose(gsl_matrix *L, double *norm, const int t, int *k, 
		const double fidbound, short *exact, const short rank, const short fidelity);
double sampleProjector(struct Projector *P, gsl_matrix *L, int k, const short exact, const short parallel);

/* for debugging, remove later */
void printProjector(struct Projector *P) {
    for (int i = 0; i < P->Nstabs; i++) {
        int tmpphase = (int)gsl_vector_get(P->phases, i);

        gsl_vector *xs = gsl_vector_alloc(P->Nqubits);
        gsl_vector *zs = gsl_vector_alloc(P->Nqubits);
        gsl_matrix_get_row(xs, P->xs, i);
        gsl_matrix_get_row(zs, P->zs, i);

        char stab[20];
        stab[0] ='\0';

        for (int j = 0; j < P->Nqubits; j++) {
            double x = gsl_vector_get(xs, j);
            double z = gsl_vector_get(zs, j);

            if (x == 1 && z == 1) {
                tmpphase -= 1;
                strcat(stab, "Y");
                continue;
            }

            if (x == 1) {
                strcat(stab, "X");
                continue;
            }

            if (z == 1) {
                strcat(stab, "Z");
                continue;
            }
            strcat(stab, "_");
        }
        while (tmpphase < 0) tmpphase += 4;

        char* lookup[4] = {" +", " i", " -", "-i"};
        printf("%s%s\n", lookup[tmpphase], stab);

        gsl_vector_free(xs);
        gsl_vector_free(zs);
    }
}

/************* Main: parse args, get L, and eval projectors *************/
int main(int argc, char* argv[]){
    gsl_set_error_handler_off();

    int debug = 0; 
    int print = 0;

    /************* Parse args for decompose *************/
    int t;
    if (!debug) scanf("%d", &t);
    // else t = 3;
     else t = 1;
    if (print) printf("t: %d\n", t);

    int k;
    if (!debug) scanf("%d", &k);  
    else k = 0;
    if (print) printf("k: %d\n", k);

    double fidbound;
    if (!debug) scanf("%lf", &fidbound);
    else fidbound = 0.001;
    if (print) printf("fidbound: %f\n", fidbound);

    short exact;
    if (!debug) scanf("%d", &exact); 
    else exact = 1;
    if (print) printf("exact: %d\n", exact);

    short rank;
    if (!debug) scanf("%d", &rank);
    else rank = 1;
    if (print) printf("rank: %d\n", rank);

    short fidelity;
    if (!debug) scanf("%d", &fidelity); 
    else fidelity = 1;
    if (print) printf("fidelity: %d\n", fidelity);

    /************* Parse args for main proc *************/

    struct Projector *G;
    struct Projector *H;
    if (!debug) {
        G = readProjector();
        H = readProjector();
    } else {
        G = (struct Projector *)malloc(sizeof(struct Projector));
        H = (struct Projector *)malloc(sizeof(struct Projector));
       
        
        G->Nstabs = 3;
        H->Nstabs = 2;
        G->Nqubits = 3;
        H->Nqubits = 3;
        

        /*
        G->Nstabs = 1;
        H->Nstabs = 0;
        G->Nqubits = 1;
        H->Nqubits = 1;
        */
            
        G->phases = gsl_vector_alloc(G->Nstabs);
        G->xs = gsl_matrix_alloc(G->Nstabs, G->Nqubits);
        G->zs = gsl_matrix_alloc(G->Nstabs, G->Nqubits);

        H->phases = gsl_vector_alloc(H->Nstabs);
        H->xs = gsl_matrix_alloc(H->Nstabs, H->Nqubits);
        H->zs = gsl_matrix_alloc(H->Nstabs, H->Nqubits);
       
        int v;
        int idx = 0;

        int Gvals[] = {1,1,1,0,1,0,1,2,0,1,1,0,0,1,1,0,0,1,1,1,0};
        // int Gvals[] = {0, 1};
        for (int i = 0; i < G->Nstabs; i++) {
            v = Gvals[idx++];
            gsl_vector_set(G->phases, i, (double)v);
            for (int j = 0; j < G->Nqubits; j++) {
                v = Gvals[idx++];
                gsl_matrix_set(G->xs, i, j, (double)v);
                v = Gvals[idx++];
                gsl_matrix_set(G->zs, i, j, (double)v);
            }
        }

        int Hvals[] = {1,1,1,0,1,0,1,1,0,0,1,1,1,0};
        //int Hvals[] = {};
        idx = 0;
        for (int i = 0; i < H->Nstabs; i++) {
            v = Hvals[idx++];
            gsl_vector_set(H->phases, i, (double)v);
            for (int j = 0; j < H->Nqubits; j++) {
                v = Hvals[idx++];
                gsl_matrix_set(H->xs, i, j, (double)v);
                v = Hvals[idx++];
                gsl_matrix_set(H->zs, i, j, (double)v);
            }
        }
    }
    if (print) printf("Proj: G\n");
    if (print) printProjector(G);
    if (print) printf("Proj: H\n");
    if (print) printProjector(H);

    int Nsamples;
    if (!debug) scanf("%d", &Nsamples);
    else Nsamples = 5000;
    if (print) printf("Nsamples: %d\n", Nsamples);

    int parallel;
    if (!debug) scanf("%d", &parallel); 
    else parallel = 0; // parallelism not implemented yet
    if (print) printf("parallel: %d\n", parallel);
    
	if (print) printf("Finished reading input.\n");

    /************* Get L, using decompose *************/

    srand(time(NULL)); // set random seed
    //srand(0);

    gsl_matrix *L;
    double Lnorm;
    decompose(L, &Lnorm, t, &k, fidbound, &exact, rank, fidelity);

    if (exact == 1) {
        printf("Using exact decomposition of |H^t>: 2^%d\n", t);
    }
    if (exact == 0) {
        printf("Stabilizer rank of |L>: 2^%d\n", k);
    }

    /************* Calculate result *************/

    double numerator = 0;
    double denominator = 0;

    // calculate || Gprime |L> ||^2 
    if (Lnorm > 0 && (G->Nqubits == 0 || G->Nstabs == 0)) {
        numerator = Nsamples * sampleProjector(G, L, k, exact, 0) * Lnorm*Lnorm;
    } else {
        for (int i = 0; i < Nsamples; i++) {
            numerator += sampleProjector(G, L, k, exact, 0);
        }
    }
    if (print) printf("Calculated G\n");

    // calcalate || Hprime |L> ||^2
    if (Lnorm > 0 && (H->Nqubits == 0 || H->Nstabs == 0)) {
        denominator = Nsamples * sampleProjector(H, L, k, exact, 0) * Lnorm*Lnorm;
    } else {
        for (int i = 0; i < Nsamples; i++) {
            denominator += sampleProjector(H, L, k, exact, 0);
        }
    }
    if (print) printf("Calculated H\n");

    if (debug == 1) {
        printf("|| Gprime |H^t> ||^2 ~= %f\n", numerator/Nsamples);
        printf("|| Hprime |H^t> ||^2 ~= %f\n", denominator/Nsamples);

        if (denominator > 0) printf("Output: %f\n", numerator/denominator);
    } else {
        printf("%f\n", numerator/Nsamples);
        printf("%f\n", denominator/Nsamples);
    }

    free(G);
    free(H);
    return 0;
}


/************* Helper for reading projectors *************/
struct Projector* readProjector(void) {
    struct Projector *P = (struct Projector *)malloc(sizeof(struct Projector));
    scanf("%d", &(P->Nstabs));
    scanf("%d", &(P->Nqubits));
    P->phases = gsl_vector_alloc(P->Nstabs);
    P->xs = gsl_matrix_alloc(P->Nstabs, P->Nqubits);
    P->zs = gsl_matrix_alloc(P->Nstabs, P->Nqubits);
    
    int v;
    for (int i = 0; i < P->Nstabs; i++) {
        scanf("%d", &v);
        gsl_vector_set(P->phases, i, (double)v);

        for (int j = 0; j < P->Nqubits; j++) {
            scanf("%d", &v);
            gsl_matrix_set(P->xs, i, j, (double)v);
            
            scanf("%d", &v);
            gsl_matrix_set(P->zs, i, j, (double)v);
        }
    }
    return P;
}


/************* Decompose: calculate L, or decide on exact decomp *************/
//if k <= 0, the function finds k on its own
void decompose(gsl_matrix *L, double *norm, const int t, int *k, 
		const double fidbound, short *exact, const short rank, const short fidelity){
	
	//trivial case
	if(t == 0){
		L = gsl_matrix_alloc(0, 0);
        *exact = 0;
		*norm = 1;
		return;
	}
	
	double v = cos(M_PI/8);

	short forceK = 1;
	if (*k <= 0){
		//pick unique k such that 1/(2^(k-2)) \geq v^(2t) \delta \geq 1/(2^(k-1))
		forceK  = 0;
		*k = ceil(1 - 2*t*log2(v) - log2(fidbound));
	}
	
	//can achieve k = t/2 by pairs of stabilizer states
	if(*exact || (*k > t/2 && !forceK)){
        *exact = 1;
		*norm = pow(2, floor((float)t/2)/2);
		if (t % 2) *norm *= 2*v;
		return;
	}
	
	
	//prevents infinite loops
    if (*k > t){
        if (forceK){
			printf("\nCan't have k > t. Setting k to %d.", t);
		}
		*k = t;
	}
	
	double innerProduct = 0;
	
	gsl_matrix *U, *V;
	gsl_vector *S, *work;
	L = gsl_matrix_alloc(*k, t);
	U = gsl_matrix_alloc(t, *k);
	V = gsl_matrix_alloc(*k, *k);  
	S = gsl_vector_alloc(*k); 
	work = gsl_vector_alloc(*k);
	
	double Z_L;
	
	while(innerProduct < 1-fidbound){
	
        // sample random matrix
		for(int i=0;i<*k;i++){
			for(int j=0;j<t;j++){
				gsl_matrix_set(L, i, j, rand() & 1);	//set to either 0 or 1
			}
		}
		
        // optionally verify rank
		if(rank){
			int currRank = 0;
			//rank of a matrix is the number of non-zero values in its singular value decomposition
			gsl_matrix_transpose_memcpy(U, L);
			gsl_linalg_SV_decomp(U, V, S, work);
			for(int i=0;i<*k;i++){
				if(fabs(gsl_vector_get(S, i)) > 0.00001){
					currRank++;
				}
			}
			
			//check rank
			if(currRank < *k){
				printf("\nL has insufficient rank. Sampling again...");
				continue;
			}
		}
		
		if(fidelity){
			//compute Z(L) = sum_x 2^{-|x|/2}
			Z_L = 0;
			
			gsl_vector *z, *x;
			z = gsl_vector_alloc(*k);
			x = gsl_vector_alloc(t);
			int *zArray = (int *)calloc(*k, sizeof(int));
			int currPos, currTransfer;
			for(int i=0;i<pow(2,*k);i++){
				//starting from k 0s, add (binary) +1 2^k times,
                //effectively generating all possible vectors z of length k
				//least important figure is on the very left
				currPos = 0;
				currTransfer = 1;
				while(currTransfer && currPos<*k){
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
				
				for(int i=0;i<*k;i++){
					gsl_vector_set(z, i, (double)(*(zArray+currPos)));
				}
				
				gsl_blas_dgemv(CblasTrans, 1., L, z, 0., x);
				
				double temp = 0;
				for(int i=0;i<t;i++){
					temp += mod((int)gsl_vector_get(x, i), 2);
				}
				
				Z_L += pow(2, -temp/2);
			}
			
			innerProduct = pow(2,*k) * pow(v, 2*t) / Z_L;
			
			if(forceK){
				printf("\nInner product <H^t|L>: %lf\n", innerProduct);
				break;
			}
			else if(innerProduct < 1-fidbound){
				printf("\nInner product <H^t|L>: %lf - Not good enough!\n", innerProduct);
			}
			else{
				printf("\nInner product <H^t|L>: %lf\n", innerProduct);
			}
		}
		else{
			break;
		}
	}
	
	if(fidelity){
		*norm = sqrt(pow(2,*k) * Z_L);
	}
	
}

static char *binrep (unsigned int val, char *buff, int sz) {
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

//Inner product for some state in |L> ~= |H^t>
void evalLcomponent(gsl_complex *innerProd, unsigned int i, gsl_matrix *L, struct StabilizerState *theta, int k, int t){
	
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
	
	struct StabilizerState *phi = (struct StabilizerState *)malloc(sizeof(struct StabilizerState));
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
		gsl_vector_set(tempVector, xtildeidx, 1);
		if((int)gsl_vector_get(bitstring, xtildeidx) == 1){
			//|+> at index, so measure X
			measurePauli(phi, 0, zeroVector, tempVector);
		}
		else{
			//|0> at index, so measure Z
			measurePauli(phi, 0, tempVector, zeroVector);
		}
	}
	
	int eps, p, m;
	innerProduct(theta, phi, &eps, &p, &m, innerProd, 0);
    freeStabilizerState(phi);

    gsl_vector_free(bitstring);
    gsl_vector_free(tempVector);
    gsl_vector_free(zeroVector);
}

//Inner product for some state in |H^t> using pairwise decomposition
void evalHcomponent(gsl_complex *innerProd, unsigned int i, struct StabilizerState *theta, int t){
	
	int size = ceil((double)t/2);
	
    char buff[size+1];
	char *bits = binrep(i,buff,size);
	
	struct StabilizerState *phi = (struct StabilizerState *)malloc(sizeof(struct StabilizerState));
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


	for(int j=0;j<size;j++){
		if(bits[j] == '0' && !(size%2 && j==size-1)){
			gsl_matrix_set(phi->J, j*2+1, j*2, 4);
			gsl_matrix_set(phi->J, j*2, j*2+1, 4);
		}
	}


	gsl_vector *tempVector, *zeroVector;
	tempVector = gsl_vector_alloc(t);
	zeroVector = gsl_vector_alloc(t);
	gsl_vector_set_zero(zeroVector);
	
	for(int j=0;j<size;j++){
		gsl_vector_set_zero(tempVector);
		
		if(size%2 && j==size-1){
			
			gsl_vector_set(tempVector, t-1, 1);
			
			//last qubit: |H> = (1/2v)(|0> + |+>)
            //printf("In: %d, [%d], [[%d]], [[%d]], %d, [%d], [%d]\n", phi->k, (int)gsl_vector_get(phi->h, 0) ,(int)gsl_matrix_get(phi->G, 0, 0), (int)gsl_matrix_get(phi->Gbar, 0, 0), phi->Q, (int)gsl_vector_get(phi->D, 0), (int)gsl_matrix_get(phi->J, 0, 0));
			if(bits[j] == '0'){
				measurePauli(phi, 0, tempVector, zeroVector);	//|0>, measure Z
			}
			else{
				measurePauli(phi, 0, zeroVector, tempVector);	//|+>, measure X
			}
			
			continue;
		}
		
		if(bits[j] == '0'){
			continue;
		}
		
		gsl_vector_set(tempVector, j*2+1, 1);
		gsl_vector_set(tempVector, j*2, 1);
	
		measurePauli(phi, 0, zeroVector, tempVector);	//measure XX
		measurePauli(phi, 0, tempVector, zeroVector);	//measure ZZ
	}
	
    //printf("Phi: %d, [%d], [[%d]], [[%d]], %d, [%d], [%d]\n", phi->k, (int)gsl_vector_get(phi->h, 0) ,(int)gsl_matrix_get(phi->G, 0, 0), (int)gsl_matrix_get(phi->Gbar, 0, 0), phi->Q, (int)gsl_vector_get(phi->D, 0), (int)gsl_matrix_get(phi->J, 0, 0));
    //printf("Theta: %d, [%d], [[%d]], [[%d]], %d, [%d], [%d]\n", theta->k, (int)gsl_vector_get(theta->h, 0) ,(int)gsl_matrix_get(theta->G, 0, 0), (int)gsl_matrix_get(theta->Gbar, 0, 0), theta->Q, (int)gsl_vector_get(theta->D, 0), (int)gsl_matrix_get(theta->J, 0, 0));

	// int *eps, *p, *m;
	int eps;
	int p;
	int m;
	innerProduct(theta, phi, &eps, &p, &m, innerProd, 0);
    freeStabilizerState(phi);
    gsl_vector_free(tempVector);
    gsl_vector_free(zeroVector);
}

typedef struct ThreadData {
      gsl_complex *total;
      int i;
      gsl_matrix* L;
      int exact;
      struct StabilizerState* theta;
      int k;
      int t;
      pthread_mutex_t *mutex;
} ThreadData;

void *samplethread(void *args) {
    struct ThreadData *data = (struct ThreadData*)args;

    gsl_complex innerProd;

    if (data->exact) {
        evalHcomponent(&innerProd, data->i, data->theta, data->t);
    } else {
        evalLcomponent(&innerProd, data->i, data->L, data->theta, data->k, data->t);
    }

    pthread_mutex_lock(data->mutex);
    *data->total = gsl_complex_add(*data->total, innerProd);
    pthread_mutex_unlock(data->mutex);

    free(args);
    return NULL;
}



double sampleProjector(struct Projector *P, gsl_matrix *L, int k, const short exact, const short parallel){
    // empty projector
    if (P->Nstabs == 0) return 1;

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

    

    // Sample random stabilizer state
	struct StabilizerState *theta = (struct StabilizerState *)malloc(sizeof(struct StabilizerState));
    randomStabilizerState(theta, t);

    // project state onto P
    double projfactor = 1;
    gsl_vector *zeta = gsl_vector_alloc(P->Nqubits);
    gsl_vector *xi = gsl_vector_alloc(P->Nqubits);

    for (int i = 0; i < P->Nstabs; i++) {
        int m = gsl_vector_get(P->phases, i);
        gsl_matrix_get_row(zeta, P->zs, i);
        gsl_matrix_get_row(xi, P->xs, i);

        double res = measurePauli(theta, m, zeta, xi);
        projfactor *= res;

        if (res == 0) {
            freeStabilizerState(theta);
            gsl_vector_free(zeta);
            gsl_vector_free(xi);
            return 0;
        }
    } 

    gsl_complex total = gsl_complex_rect(0,0);
    gsl_complex innerProd;

    struct ThreadData *data;
    unsigned int size;

    if (exact) {
        size = pow(2, ceil((double)t / 2));
    } else {
        size = pow(2, k);
    }
    
    pthread_mutex_t total_mutex = PTHREAD_MUTEX_INITIALIZER;

    int maxthreads = 10; // maximum number of parallel threads
    pthread_t *tids = (pthread_t*)malloc(sizeof(pthread_t)*maxthreads);
    int j = 0;

    for (unsigned int i = 0; i < size; i++) {
        data = (struct ThreadData *)malloc(sizeof(struct ThreadData));
        data->total = &total;
        data->i = i;
        data->k = k;
        data->t = t;
        data->exact = exact;
        data->theta = theta;
        data->L = L;
        data->mutex = &total_mutex;

        pthread_t tid;
        pthread_create(&tid, NULL, samplethread, data); 
        tids[j] = tid;

        j++;
        if (j == maxthreads) {
            for (int k = 0; k < maxthreads; k++) {
                pthread_join(tids[k], NULL);
            }
            j = 0;
        }
    }
    for (int k = 0; k < j; k++) {
        pthread_join(tids[k], NULL);
    }

    freeStabilizerState(theta);
    gsl_vector_free(zeta);
    gsl_vector_free(xi);
    return pow(2, t) * gsl_complex_abs2(gsl_complex_mul_real(total, projfactor));
}
