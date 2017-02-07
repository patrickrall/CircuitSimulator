#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_blas.h>

#include "mpi.h"
#include "utils/comms.h"


/********************* Prototypes *********************/

double multiSampledProjector(struct Projector *P, gsl_matrix *L, int exact, double norm, int samples, int bins);
double singleProjectorSample(struct Projector *P, gsl_matrix *L, int exact);

double exactProjector(struct Projector *P, gsl_matrix *L, int exact, double norm);
gsl_complex exactProjectorWork(int i, struct Projector *P, gsl_matrix *L, int exact);

void master(int argc, char* argv[]);
void slave(void);

void decompose(const int t, gsl_matrix **L, double *norm, int *exact, int *k, 
		const double fidbound,  const int rank, const int fidelity, const int forceL,
        const int verbose, const int quiet);

/********************* MAIN *********************/
int main(int argc, char* argv[]) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    gsl_set_error_handler_off();
   
    if (world_rank == 0) {
        clock_t begin = clock(); 
        master(argc, argv);
        clock_t end = clock(); 
        printf("time elapsed: %f sec\n", (double)(end - begin)/CLOCKS_PER_SEC);
    } else slave();

    MPI_Finalize();
    return 0;
}


/********************* MASTER THREAD *********************/
void master(int argc, char* argv[]) {
    // print mode, used for IO debugging. 
    // For algorithm related output use verbose
    int print = 0;
    if (print) printf("C backend print mode is on.\n");

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
            return;
        }
    }

    /************* Parse args for decompose *************/

    // Logging
    int quiet;
    fscanf(stream,"%d", &quiet);
    if (print) printf("quiet: %d\n", quiet);

    int verbose;
    fscanf(stream,"%d", &verbose);
    if (print) printf("verbose: %d\n", quiet);

    // Sampling method
    int noapprox;
    fscanf(stream,"%d", &noapprox);
    if (print) printf("noapprox: %d\n", noapprox);

    int samples;
    fscanf(stream,"%d", &samples);
    if (print) printf("samples: %d\n", samples);

    int bins;
    fscanf(stream,"%d", &bins);
    if (print) printf("bins: %d\n", bins);

    // State preparation
    int t;
    fscanf(stream,"%d", &t);
    if (print) printf("t: %d\n", t);

    int k;
    fscanf(stream,"%d", &k);  
    if (print) printf("k: %d\n", k);

    int exact;
    fscanf(stream,"%d", &exact); 
    if (print) printf("exact: %d\n", exact);

    double fidbound;
    fscanf(stream,"%lf", &fidbound);
    if (print) printf("fidbound: %f\n", fidbound);

    int fidelity;
    fscanf(stream,"%d", &fidelity); 
    if (print) printf("fidelity: %d\n", fidelity);

    int rank;
    fscanf(stream,"%d", &rank);
    if (print) printf("rank: %d\n", rank);

    // Debug
    int forceL;
    fscanf(stream,"%d", &forceL);
    if (print) printf("forceL: %d\n", forceL);

    // Projectors
    struct Projector *G = readProjector(stream);
    if (print) printf("G:\n");
    if (print) printProjector(G);
    struct Projector *H = readProjector(stream);
    if (print) printf("H:\n");
    if (print) printProjector(H);

    /************** Call decompose, send data to workers *************/
   
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double norm;
    gsl_matrix *L; 
    decompose(t, &L, &norm, &exact, &k, fidbound, rank, fidelity, forceL, verbose, quiet); 

    // random seed
    int seed = (int)time(NULL);
    srand(seed);


    for (int dest = 1; dest < world_size; dest++) {
        send_int(1, dest); // init command
        send_int(rand(), dest); // random seed
        
        send_int(noapprox, dest);
        send_int(exact, dest);
        if (!exact) send_gsl_matrix(L, dest);
    }

    /********************* Evaluate projectors *********************/

    double numerator, denominator;
    if (noapprox == 0) {
        numerator = multiSampledProjector(G, L, exact, norm, samples, bins);
        denominator = multiSampledProjector(H, L, exact, norm, samples, bins);
    } else {
        numerator = exactProjector(G, L, exact, norm);
        denominator = exactProjector(H, L, exact, norm);
    }

    // tell procs to terminate
    for (int dest = 1; dest < world_size; dest++) send_int(0, dest);

    if (print == 1) {
        printf("|| Gprime |H^t> ||^2 ~= %f\n", numerator);
        printf("|| Hprime |H^t> ||^2 ~= %f\n", denominator);
        if (denominator > 0) printf("Output: %f\n", numerator/denominator);
    } 
        
    printf("%f\n", numerator);
    printf("%f\n", denominator);

    free(G);
    free(H);
    if (!exact) gsl_matrix_free(L);
}


/********************* SLAVE THREAD *********************/
void slave(void) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    struct Projector *P;

    int seed, noapprox, exact; 
    gsl_matrix *L; 

    int size;

    while (1) {
        int cmd = recv_int(0);
        // 0 - exit
        // 1 - init
        // 2 - eval projector

        switch (cmd) {
            case 0: // exit
                if (!exact) gsl_matrix_free(L);
                return;

            case 1: // init
                seed = recv_int(0);     
                srand(seed);
                
                noapprox = recv_int(0);
                exact = recv_int(0);
                if (!exact) L = recv_gsl_matrix(0);
                
                break;
            case 2: // eval projector
                P = recv_projector(0);
                size = recv_int(0);
                
                if (noapprox == 0) { // add samples
                    double total = 0;

                    for (int i = world_rank; i < size; i += world_size) {
                        total += singleProjectorSample(P, L, exact);
                    }

                    send_double(total, 0);
                } else { // add inner products
                    gsl_complex total = gsl_complex_rect(0,0);
                    gsl_complex part;

                    for (int i = world_rank; i < size; i += world_size) {
                        part = exactProjectorWork(i, P, L, exact);
                        total = gsl_complex_add(total, part);
                    }

                    send_gsl_complex(total, 0);
                }

                free(P);
                break;
            default:
                continue;
        }
    }
}


/************* Decompose: calculate L, or decide on exact decomp *************/
// helper that respects sign
int mod(int a, int b);

//if k <= 0, the function finds k on its own
void decompose(const int t, gsl_matrix **L, double *norm, int *exact, int *k, 
		const double fidbound,  const int rank, const int fidelity, const int forceL,
        const int verbose, const int quiet) {
	
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
	
	*L = gsl_matrix_alloc(*k, t);
	
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
            // Make identity matrix
            gsl_matrix * Null = gsl_matrix_alloc(t, t);
            gsl_matrix_set_identity(Null);
            int nullsize = t;

            // memory for temporary rows
            gsl_matrix * Good = gsl_matrix_alloc(t, t);
            gsl_matrix * Bad = gsl_matrix_alloc(t, t);
            int goodsize, badsize;
            
            for (int i = 0; i < *k; i++) {
                goodsize = 0;
                badsize = 0;

                for (int j = 0; j < nullsize; j++) {
                    int inner = 0;
                    for (int l = 0; l < t; l++) {
                        inner += gsl_matrix_get(*L, i, l) * gsl_matrix_get(Null, j, l);
                    }
                    
                    if (inner % 2 == 0) {
                        for (int l = 0; l < t; l++) {
                            gsl_matrix_set(Good, goodsize, l, gsl_matrix_get(Null, j, l));
                        }
                        goodsize += 1;
                    } else {
                        for (int l = 0; l < t; l++) {
                            gsl_matrix_set(Bad, badsize, l, gsl_matrix_get(Null, j, l));
                        }
                        badsize += 1;
                    }
                }

                nullsize = 0;
                for (int j = 0; j < goodsize; j++) {
                    for (int l = 0; l < t; l++) {
                        gsl_matrix_set(Null, nullsize, l, gsl_matrix_get(Good, j, l));
                    }
                    nullsize += 1;
                }
 
                for (int j = 1; j < badsize; j++) {
                    for (int l = 0; l < t; l++) {
                        double val = gsl_matrix_get(Bad, 0, l) + gsl_matrix_get(Bad, j, l);
                        gsl_matrix_set(Null, nullsize, l, (double)((int)val % 2));
                    }
                    nullsize += 1;
                }           
            }
		
            gsl_matrix_free(Null);
            gsl_matrix_free(Good);
            gsl_matrix_free(Bad);

			//check rank
			if(t - nullsize < *k){
				if (!quiet) printf("L has insufficient rank. Sampling again...\n");
				continue;
			}
		}
		
		if (fidelity) {
			//compute Z(L) = sum_x 2^{-|x|/2}
			Z_L = 0;
			
			gsl_vector *z, *x;
			z = gsl_vector_alloc(*k);
			x = gsl_vector_alloc(t);
			int *zArray = (int *)calloc(*k, sizeof(int));
			int currPos, currTransfer;
			for (int i=0; i<pow(2,*k); i++) {
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
			
			if(forceK) {
                // quiet can't be set for this
				printf("Inner product <H^t|L>: %lf\n", innerProduct);
				break;
			} else if (innerProduct < 1-fidbound) {
				if (!quiet) printf("Inner product <H^t|L>: %lf - Not good enough!\n", innerProduct);
			} else {
				if (!quiet) printf("Inner product <H^t|L>: %lf\n", innerProduct);
			}
		}
		else break;
	}
	
	if(fidelity) *norm = sqrt(pow(2,*k) * Z_L);
}
