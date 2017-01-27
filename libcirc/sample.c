#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_linalg.h>
#include "stabilizer/stabilizer.h"
#include <gsl/gsl_errno.h>
#include "mpi.h"
#include <limits.h>


/***************************** Projector struct ************************/
struct Projector {
    int Nstabs;
    int Nqubits;
    gsl_vector* phases;
    gsl_matrix* xs;
    gsl_matrix* zs; 
};

/***************************** Some prototypes *************************/


void decompose(const int t, gsl_matrix **L, double *norm, int *exact, int *k, 
		const double fidbound,  const int rank, const int fidelity, const int forceL,
        const int verbose, const int quiet);

void evalLcomponent(gsl_complex *innerProd, unsigned int i, gsl_matrix *L, struct StabilizerState *theta, int k, int t);

void evalHcomponent(gsl_complex *innerProd, unsigned int i, struct StabilizerState *theta, int t);

double sampleProjector(struct Projector *P, gsl_matrix *L, int k,
        const int exact, const int stateParallel);


/************* Helpers for reading and printing projectors *************/
struct Projector* readProjector(FILE* stream) {
    struct Projector *P = (struct Projector *)malloc(sizeof(struct Projector));
    fscanf(stream,"%d", &(P->Nstabs));
    fscanf(stream,"%d", &(P->Nqubits));

    if (P->Nstabs == 0) return P;

    P->phases = gsl_vector_alloc(P->Nstabs);
    P->xs = gsl_matrix_alloc(P->Nstabs, P->Nqubits);
    P->zs = gsl_matrix_alloc(P->Nstabs, P->Nqubits);
    
    int v;
    for (int i = 0; i < P->Nstabs; i++) {
        fscanf(stream,"%d", &v);
        gsl_vector_set(P->phases, i, (double)v);

        for (int j = 0; j < P->Nqubits; j++) {
            fscanf(stream,"%d", &v);
            gsl_matrix_set(P->xs, i, j, (double)v);
            
            fscanf(stream,"%d", &v);
            gsl_matrix_set(P->zs, i, j, (double)v);
        }
    }
    return P;
}

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

/************* MPI message passing *************/


//----------- int macro
void send_int(int i, int dest) {
    MPI_Send(&i, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
}
int recv_int(int src) {
    int buff;
    MPI_Recv(&buff, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return buff;
}

//----------- double macro
void send_double(double i, int dest) {
    MPI_Send(&i, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
}
double recv_double(int src) {
    double buff;
    MPI_Recv(&buff, 1, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return buff;
}


//----------- gsl_vector
void send_gsl_vector(gsl_vector* vec, int dest) {
    int size = (int)vec->size;
    send_int(size, dest); 
    for (int i = 0; i < size; i++) {
        send_double(gsl_vector_get(vec, i), dest);
    }
}
gsl_vector* recv_gsl_vector(int src) {
    int size = recv_int(src);
    gsl_vector* vec = gsl_vector_alloc(size);
    for (int i = 0; i < size; i++) {
        gsl_vector_set(vec, i, recv_double(src));
    }
    return vec;
}

//----------- gsl_matrix
void send_gsl_matrix(gsl_matrix* mat, int dest){
    int size1 = (int)mat->size1;
    int size2 = (int)mat->size2;

    send_int(size1, dest); 
    send_int(size2, dest); 

    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            send_double(gsl_matrix_get(mat, i, j), dest);
        }
    }
} 
gsl_matrix* recv_gsl_matrix(int src) {
    int size1 = recv_int(src);
    int size2 = recv_int(src);

    gsl_matrix* mat = gsl_matrix_alloc(size1, size2);

    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            gsl_matrix_set(mat, i, j, recv_double(src));
        }
    }
    return mat;
}


//---------- gsl_complex
void send_gsl_complex(gsl_complex z, int dest) {
    send_double(GSL_REAL(z), dest);
    send_double(GSL_IMAG(z), dest);
}
gsl_complex recv_gsl_complex(int src) {
    double real = recv_double(src);
    double imag = recv_double(src);
    return gsl_complex_rect(real, imag);
}

//---------- projectors
void send_projector(struct Projector* P, int dest) {
    send_int(P->Nstabs, dest);
    send_int(P->Nqubits, dest);
    if (P->Nstabs == 0) return;

    send_gsl_vector(P->phases, dest);
    send_gsl_matrix(P->xs, dest);
    send_gsl_matrix(P->zs, dest);
}
struct Projector* recv_projector(int src) {
    struct Projector *P = (struct Projector *)malloc(sizeof(struct Projector));
    P->Nstabs = recv_int(src);
    P->Nqubits = recv_int(src);
    if (P->Nstabs == 0) return P;

    P->phases = recv_gsl_vector(src);
    P->xs = recv_gsl_matrix(src);
    P->zs = recv_gsl_matrix(src);
    return P;
}

//---------- stabilizer states
void send_stabilizer_state(struct StabilizerState* state, int dest) {
    send_int(state->n, dest);
    send_int(state->k, dest);

    send_gsl_vector(state->h, dest);
    send_gsl_matrix(state->G, dest);
    send_gsl_matrix(state->Gbar, dest);

    send_int(state->Q, dest);
    send_gsl_vector(state->D, dest);
    send_gsl_matrix(state->J, dest);
}
struct StabilizerState *recv_stabilizer_state(int src) {
    int n = recv_int(src);
    int k = recv_int(src);

    struct StabilizerState *state = allocStabilizerState(n, k);
    state->h = recv_gsl_vector(src);
    state->G = recv_gsl_matrix(src);
    state->Gbar = recv_gsl_matrix(src);

    state->Q = recv_int(src);
    state->D = recv_gsl_vector(src);
    state->J = recv_gsl_matrix(src);
    return state;
}


/************* Main: parse args, get L, and eval projectors *************/
int main(int argc, char* argv[]) {
    /***************** Initialize the environment ***************/
    MPI_Init(NULL, NULL);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    gsl_set_error_handler_off();
        
    
    // declare variables that every process will need for sampling parallelization
    int samples;
    int k; 
    int t;
    int exact; 
    gsl_matrix *L; 
    struct Projector *G;
    struct Projector *H;
    double Lnorm;

    int stateParallel; // always needed

    // print mode, used for IO debugging. 
    // For algorithm related output use verbose
    int print = 0;
    if (print && world_rank == 0) printf("C backend print mode is on.\n");

    if (world_rank == 0) {
        /************* Determine data source: file or stdin *************/

        char file[256] = "";

        if (argc == 1) scanf("%s", file);
        else strcpy(file, argv[1]);
        
        FILE* stream;
        if (strlen(file) == 0 || strcmp(file, "stdin") == 0) {
            if (print) printf("Reading arguments from stdin\n");
            stream = stdin;
        } else {
            if (print) printf("Reading arguments from file: %s\n", file);
            stream = fopen(file, "r");
            if (stream == NULL) {
                printf("Error reading file.\n");
                return -1;
            }
        }

        /************* Parse args for decompose *************/
        fscanf(stream,"%d", &t);
        if (print) printf("t: %d\n", t);

        fscanf(stream,"%d", &k);  
        if (print) printf("k: %d\n", k);

        int quiet;
        fscanf(stream,"%d", &quiet);
        if (print) printf("quiet: %d\n", quiet);

        int verbose;
        fscanf(stream,"%d", &verbose);
        if (print) printf("verbose: %d\n", quiet);

        double fidbound;
        fscanf(stream,"%lf", &fidbound);
        if (print) printf("fidbound: %f\n", fidbound);

        fscanf(stream,"%d", &exact); 
        if (print) printf("exact: %d\n", exact);

        int rank;
        fscanf(stream,"%d", &rank);
        if (print) printf("rank: %d\n", rank);

        int forceL;
        fscanf(stream,"%d", &forceL);
        if (print) printf("forceL: %d\n", forceL);

        int fidelity;
        fscanf(stream,"%d", &fidelity); 
        if (print) printf("fidelity: %d\n", fidelity);
        
        /************* Parse args for main proc *************/

        // stateParallel variable: 0=Sample, 1=State
        fscanf(stream,"%d", &stateParallel); 
        if (print) printf("stateParallel: %d\n", stateParallel);
        // number of cores is determined by mpirun call

        G = readProjector(stream);
        H = readProjector(stream);
       
        if (print) printf("Proj: G\n");
        if (print) printProjector(G);
        if (print) printf("Proj: H\n");
        if (print) printProjector(H);

        fscanf(stream,"%d", &samples);
        if (print) printf("samples: %d\n", samples);
        
        if (print) printf("Finished reading input.\n");

        /************* Get L, using decompose *************/

        L = NULL;
        decompose(t, &L, &Lnorm, &exact, &k, fidbound, rank, fidelity, forceL, verbose, quiet);

        if (verbose) {
            if (exact == 1) printf("Using exact decomposition of |H^t>: 2^%d\n", t);
            else printf("Stabilizer rank of |L>: 2^%d\n", k);
        }

        /********** Report parallelization mode ***********/

        if (verbose) {
            if (stateParallel == 0) printf("Parallelizing over %d samples\n", samples);
            else {
                 if (exact == 1) printf("Parallelizing over %d stabilizers in |H^t>\n", (int)pow(2,ceil((double)t/2)));
                 else printf("Parallelizing over %d stabilizers in |L>\n", (int)pow(2,k));
            }
        }

        /************* Send data to workers *************/
    
        int seed = (int)time(NULL);
        srand(seed); // set random seed
        
        // send stateParallel, random seed to workers
        for (int dest = 1; dest < world_size; dest++) {
            send_int(rand(), dest);
            send_int(stateParallel, dest);
        }

        // sample mode data
        if (stateParallel == 0) {
            for (int dest = 1; dest < world_size; dest++) {
                send_int(samples, dest);
                send_int(k, dest); 
                send_int(exact, dest); 
                if (exact == 0) send_gsl_matrix(L, dest); 
                send_double(Lnorm, dest); 
                send_projector(G, dest);
                send_projector(H, dest);
            }
        }




    } else {
        /************* Worker process *************/
        int seed = recv_int(0);
        srand(seed);
        stateParallel = recv_int(0);

        if (stateParallel == 0) {  
            samples = recv_int(0);
            k = recv_int(0); 
            exact = recv_int(0); 
            if (exact == 0) L = recv_gsl_matrix(0); 
            Lnorm = recv_double(0);
            G = recv_projector(0);
            H = recv_projector(0);
        }
    }

    int inc;
    if (stateParallel == 0) inc = world_size;
    else inc = 1;

    /************* Calculate numerator  *************/
    double numerator = 0;

    // calculate || Gprime |L> ||^2 
    if (Lnorm > 0 && (G->Nqubits == 0 || G->Nstabs == 0)) {
        if (world_rank == 0) numerator = samples * sampleProjector(G, L, k, exact, stateParallel) * Lnorm*Lnorm;
    } else {
        if (stateParallel == 0 || world_rank == 0) {
            for (int i = world_rank; i < samples; i+=inc) {
                numerator += sampleProjector(G, L, k, exact, stateParallel);

                if (world_rank == 0 && stateParallel == 1) { // show progress if stateparallel
                    printf("Numerator: %d/%d samples\n", i+1, samples);
                    fflush(stdout);
                }
            }
        }
    }

    // Sync numerator
    if (stateParallel == 0) {
        if (world_rank == 0) { 
            for (int src = 1; src < world_size; src++) {
                numerator += recv_double(src);
            }
        } else send_double(numerator, 0);
    }

    /************* Calculate denominator  *************/
    double denominator = 0;

    // calcalate || Hprime |L> ||^2
    if (Lnorm > 0 && (H->Nqubits == 0 || H->Nstabs == 0)) {
        if (world_rank == 0) denominator = samples * sampleProjector(H, L, k, exact, stateParallel) * Lnorm*Lnorm;
    } else {
        if (stateParallel == 0 || world_rank == 0) {
            for (int i = world_rank; i < samples; i+=inc) {
                denominator += sampleProjector(H, L, k, exact, stateParallel);
                
                if (world_rank == 0 && stateParallel == 1) { // show progress if stateparallel
                    printf("Denominator: %d/%d samples\n", i+1, samples);
                    fflush(stdout);
                }
            }
        }
    }

    // Sync denominator
    if (stateParallel == 0) {
        if (world_rank == 0) { 
            for (int src = 1; src < world_size; src++) {
                denominator += recv_double(src);
            }
        } else send_double(denominator, 0);
    }

        
    /************* Print output *************/

    if (world_rank == 0) { 
        if (print == 1) {
            printf("|| Gprime |H^t> ||^2 ~= %f\n", numerator/samples);
            printf("|| Hprime |H^t> ||^2 ~= %f\n", denominator/samples);
            if (denominator > 0) printf("Output: %f\n", numerator/denominator);
        } 
        
        printf("%f\n", numerator);
        printf("%f\n", denominator);
    }

    if (stateParallel == 0 || world_rank == 0) {
        if (exact == 0) gsl_matrix_free(L);
        free(G);
        free(H);
    }

    /************* stateParallel worker thread  *************/

    while (stateParallel && world_rank != 0) {
        if (recv_int(0) == 1) break;
        
        gsl_complex total = gsl_complex_rect(0,0);
        gsl_complex innerProd;
        int size;

        // get data
        exact = recv_int(0);
        t = recv_int(0);
        k = recv_int(0);
        if (!exact) L = recv_gsl_matrix(0);
        struct StabilizerState *theta = recv_stabilizer_state(0);

        if (exact) {
            size = pow(2, ceil((double)t / 2));
            for (int i = world_rank; i < size; i++) {
                evalHcomponent(&innerProd, i, theta, t);
                total = gsl_complex_add(total, innerProd);  
            }
        } else {
            size = pow(2, k);
            for (int i = world_rank; i < size; i++) {
                evalLcomponent(&innerProd, i, L, theta, k, t);
                total = gsl_complex_add(total, innerProd);
            }
        }

        // sync data
        send_gsl_complex(total, 0);
        
        if (!exact) gsl_matrix_free(L);
        freeStabilizerState(theta);

    }

    /************* done *************/

    // terminate other processes
    if (stateParallel && world_rank == 0) {
        for (int dest = 1; dest < world_size; dest++) send_int(1, dest);
    }
    
    MPI_Finalize();
    return 0;
}



/************* Decompose: calculate L, or decide on exact decomp *************/
//if k <= 0, the function finds k on its own
void decompose(const int t, gsl_matrix **L, double *norm, int *exact, int *k, 
		const double fidbound,  const int rank, const int fidelity, const int forceL,
        const int verbose, const int quiet){
	
	//trivial case
	if(t == 0){
		*L = gsl_matrix_alloc(0, 0);
        *exact = 0;
		*norm = 1;
		return;
	}
	
	double v = cos(M_PI/8);

    //exact case
    *norm = pow(2, floor((float)t/2)/2);
	if (t % 2) *norm *= 2*v;
    if (*exact) return;

	int forceK = 1;
	if (*k <= 0){
		forceK  = 0;
		//pick unique k such that 1/(2^(k-2)) \geq v^(2t) \delta \geq 1/(2^(k-1))
		*k = ceil(1 - 2*t*log2(v) - log2(fidbound));
        if (verbose) printf("Autopicking k = %d.", *k);
	}
	
	//can achieve k = t/2 by pairs of stabilizer states
	if(*k > t/2 && !forceK && !forceL){
        if (verbose) printf("k > t/2. Reverting to exact decomposition.");
        *exact = 1;
		return;
	}
	
	//prevents infinite loops
    if (*k > t){
        if (forceK && !quiet){
			printf("Can't have k > t. Setting k to %d.\n", t);
		}
		*k = t;
	}
	
	double innerProduct = 0;
	
	gsl_matrix *U, *V;
	gsl_vector *S, *work;
	*L = gsl_matrix_alloc(*k, t);
	U = gsl_matrix_alloc(t, *k);
	V = gsl_matrix_alloc(*k, *k);  
	S = gsl_vector_alloc(*k); 
	work = gsl_vector_alloc(*k);
	
	double Z_L;
	
	while(innerProduct < 1-fidbound || forceK){
	
        // sample random matrix
		for(int i=0;i<*k;i++){
			for(int j=0;j<t;j++){
				gsl_matrix_set(*L, i, j, rand() & 1);	//set to either 0 or 1
			}
		}
		
        // optionally verify rank
		if(rank){
			int currRank = 0;
			//rank of a matrix is the number of non-zero values in its singular value decomposition
			gsl_matrix_transpose_memcpy(U, *L);
			gsl_linalg_SV_decomp(U, V, S, work);
			for(int i=0;i<*k;i++){
				if(fabs(gsl_vector_get(S, i)) > 0.00001){
					currRank++;
				}
			}
			
			//check rank
			if(currRank < *k){
				if (!quiet) printf("L has insufficient rank. Sampling again...\n");
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
				
				gsl_blas_dgemv(CblasTrans, 1., *L, z, 0., x);
				
				double temp = 0;
				for(int i=0;i<t;i++){
					temp += mod((int)gsl_vector_get(x, i), 2);
				}
				
				Z_L += pow(2, -temp/2);
			}
			
			innerProduct = pow(2,*k) * pow(v, 2*t) / Z_L;
			
			if(forceK){
                // quiet can't be set for this
				printf("Inner product <H^t|L>: %lf\n", innerProduct);
				break;
			}
			else if(innerProduct < 1-fidbound){
				if (!quiet) printf("Inner product <H^t|L>: %lf - Not good enough!\n", innerProduct);
			}
			else{
				if (!quiet) printf("Inner product <H^t|L>: %lf\n", innerProduct);
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

	int eps, p, m;
	innerProduct(theta, phi, &eps, &p, &m, innerProd, 0);
    freeStabilizerState(phi);
    gsl_vector_free(tempVector);
}


double sampleProjector(struct Projector *P, gsl_matrix *L, int k, const int exact, const int stateParallel){
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

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
	struct StabilizerState *theta = randomStabilizerState(t);

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

    gsl_vector_free(zeta);
    gsl_vector_free(xi);
    
    gsl_complex total = gsl_complex_rect(0,0);
    gsl_complex innerProd;
    int size;

    // parallel mode
    int inc = 1;
    if (stateParallel == 1) {
        inc = world_size;
         
        for (int dest = 1; dest < world_size; dest++) {
            send_int(0, dest); // tell process to continue
            send_int(exact, dest);
            send_int(t, dest);
            send_int(k, dest);
            if (!exact) send_gsl_matrix(L, dest);
            send_stabilizer_state(theta, dest);
        }
    }

    if (exact) {
        size = pow(2, ceil((double)t / 2));
        for (int i = 0; i < size; i+=inc) {
            evalHcomponent(&innerProd, i, theta, t);
            total = gsl_complex_add(total, innerProd);  
        }
    } else {
        size = pow(2, k);
        for (int i = 0; i < size; i+=inc) {
            evalLcomponent(&innerProd, i, L, theta, k, t);
            total = gsl_complex_add(total, innerProd);
        }
    }

    // resynchronize
    if (stateParallel == 1) {
        for (int src = 1; src < world_size; src++) {
            total = gsl_complex_add(total, recv_gsl_complex(src));
        }
    }
    
    freeStabilizerState(theta);

    double out = pow(2, t) * gsl_complex_abs2(gsl_complex_mul_real(total, projfactor));
    return out;
}
