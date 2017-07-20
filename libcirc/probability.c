#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#include "mpi.h"
#include "utils/comms.h"


/********************* Prototypes *********************/

double multiSampledProjector(struct Projector* P, struct BitMatrix* L, int exact, double norm, int samples, int bins);
double singleProjectorSample(struct Projector* P, struct BitMatrix* L, int exact);

double exactProjector(struct Projector* P, struct BitMatrix* L, int exact, double norm);
Complex exactProjectorWork(int i, struct Projector* P, struct BitMatrix* L, int exact);

void master(int argc, char* argv[]);
void slave(void);

void decompose(const int t, struct BitMatrix **L, double *norm, int *exact, int *k, 
		const double fidbound,  const int rank, const int fidelity, const int forceL,
        const int verbose, const int quiet);

char *binrep(unsigned int val, char *buff, int sz);

/********************* MAIN *********************/
int main(int argc, char* argv[]) {
    MPI_Init(NULL, NULL);
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
   
    if (world_rank == 0) master(argc, argv);
    else slave();

    MPI_Finalize();
    return 0;
}


/********************* MASTER THREAD *********************/
void master(int argc, char* argv[]) {
    // print mode, used for IO debugging. 
    // For algorithm related output use verbose
    int print = 1;
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
    if (print) printf("verbose: %d\n", verbose);

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

    int forceSample;
    fscanf(stream,"%d", &forceSample);
    if (print) printf("forceSample: %d\n", forceSample);


    // Projectors
    struct Projector *G = readProjector(stream);
    if (print) printf("G:\n");
    if (print) printProjector(G);
    struct Projector *H = readProjector(stream);
    if (print) printf("H:\n");
    if (print) printProjector(H);

    /************** BLAS or custom back end *************/

    #ifdef BLAS
        if (verbose) printf("Using BLAS for matrix operations.\n");
    #else 
        if (verbose) printf("Using custom code for matrix operations.\n");
    #endif

    /************** Call decompose, send data to workers *************/
   
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double norm;
    struct BitMatrix* L; 
    decompose(t, &L, &norm, &exact, &k, fidbound, rank, fidelity, forceL, verbose, quiet); 

    if (verbose) {
        if (exact) printf("Using exact decomposition of |H^t>: 2^%d\n", (t+1)/2);
        else printf("Stabilizer rank of |L>: 2^%d\n", k);
    }

    if (noapprox == 0 && forceSample == 0) {
        if (exact) {
            if (samples*bins*2 > (pow(2,(t+1)/2) - 1)) {
                noapprox = 1;
                if (verbose) printf("More samples than terms in exact calculation. Disabling sampling.\n");
            }
        } else {
            if (samples*bins*2 > (pow(2,k) - 1)) {
                noapprox = 1;
                if (verbose) printf("More samples than terms in exact calculation. Disabling sampling.\n");
            }
        }
    }

    if (exact == 0 && print) {
        printf("L:\n");
        BitMatrixPrint(L);
    }

    // random seed
    int seed = (int)time(NULL);
    srand(seed);


    for (int dest = 1; dest < world_size; dest++) {
        sendInt(1, dest); // init command
        sendInt(rand(), dest); // random seed
        
        sendInt(noapprox, dest);
        sendInt(exact, dest);
        if (!exact) sendBitMatrix(L, dest);
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
    for (int dest = 1; dest < world_size; dest++) sendInt(0, dest);

    int sigfigs = 17;  // number of sigfigs

    if (print == 1) {
        printf("|| Gprime |H^t> ||^2 ~= %.*e\n", sigfigs, numerator);
        printf("|| Hprime |H^t> ||^2 ~= %.*e\n", sigfigs, denominator);
        if (denominator > 0) printf("Output: %.*e\n", sigfigs, numerator/denominator);
    } 
        
    printf("%.*e\n", sigfigs, numerator);
    printf("%.*e\n", sigfigs, denominator);

    freeProjector(G);
    freeProjector(H);
    if (!exact) BitMatrixFree(L);
}


/********************* SLAVE THREAD *********************/
void slave(void) {
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    struct Projector *P;

    int seed, noapprox, exact; 
    struct BitMatrix* L; 

    int size;

    while (1) {
        int cmd = recvInt(0);
        // 0 - exit
        // 1 - init
        // 2 - eval projector

        switch (cmd) {
            case 0: // exit
                if (!exact) BitMatrixFree(L);
                return;

            case 1: // init
                seed = recvInt(0);     
                srand(seed);
                
                noapprox = recvInt(0);
                exact = recvInt(0);
                if (!exact) L = recvBitMatrix(0);
                
                break;
            case 2: // eval projector
                P = recvProjector(0);
                size = recvInt(0);
                
                if (noapprox == 0) { // add samples
                    double total = 0;

                    for (int i = world_rank; i < size; i += world_size) {
                        total += singleProjectorSample(P, L, exact);
                    }

                    sendDouble(total, 0);
                } else { // add inner products
                    Complex total = {0,0};

                    for (int l = world_rank; l < size; l += world_size) {
                        Complex part = exactProjectorWork(l, P, L, exact);
                        total = ComplexAdd(total, part);
                    }

                    sendComplex(total, 0);
                }

                freeProjector(P);
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
void decompose(const int t, struct BitMatrix **L, double *norm, int *exact, int *k, 
		const double fidbound,  const int rank, const int fidelity, const int forceL,
        const int verbose, const int quiet) {
	
	//trivial case
	if(t == 0){
		*L = 0x00;
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
        if (verbose) printf("Autopicking k = %d to achieve delta = %f.\n", *k, fidbound);
	}
	
	//can achieve k = t/2 by pairs of stabilizer states
	if(*k > t/2 && !forceK && !forceL){
        if (verbose) printf("k > t/2. Reverting to exact decomposition.\n");
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
	
	*L = newBitMatrixZero(*k, t);
	
	double Z_L;
	
	while(innerProduct < 1-fidbound || forceK){
	
        // sample random matrix
	    BitMatrixSetRandom(*L);	
		
        // optionally verify rank
		if(rank){
            int thisrank = BitMatrixRank(*L); 

			//check rank
			if(thisrank < *k){
				if (!quiet) printf("L has insufficient rank. Sampling again...\n");
				continue;
			}
		}
		
		if (fidelity) {
			//compute Z(L) = sum_x 2^{-|x|/2}
			Z_L = 0;
		
            char buff[*k+1];
			for (int i=0; i<pow(2,*k); i++) {
                char *Lbits = binrep(i,buff,*k);
                
                struct BitVector *bitstring = newBitVector(t);

                int j=0;
                while(Lbits[j] != '\0'){
                    if(Lbits[j] == '1'){
                        struct BitVector *tempVector = BitMatrixGetRow(*L, j);
                        BitVectorXorSet(bitstring, tempVector);
                        BitVectorFree(tempVector);
                    }
                    j++;
                }
                
                int hamming = 0;
                for (int j = 0; j<t; j++) hamming += BitVectorGet(bitstring, j);
                
                BitVectorFree(bitstring);
                
			    Z_L += pow(2, -hamming/2);
            }

            innerProduct = pow(2,*k) * pow(v, 2*t) / Z_L;
			
			if(forceK) {
                // quiet can't be set for this
				printf("delta = 1 - <H^t|L>: %lf\n", 1 - innerProduct);
				break;
			} else if (innerProduct < 1-fidbound) {
				if (!quiet) printf("delta = 1 - <H^t|L>: %lf - Not good enough!\n", 1 - innerProduct);
			} else {
				if (!quiet) printf("delta = 1 - <H^t|L>: %lf\n", 1 - innerProduct);
			}
		}
		else break;
	}
	
	if (fidelity) *norm = sqrt(pow(2,*k) * Z_L);
}
