#include <stdlib.h>
#include <math.h>

#include "stabilizer/stabilizer.h"
#include "mpi.h"

/***************** Prototypes *******************/
// linked from stateprep.c
struct StabilizerState* prepH(int i, int t);
struct StabilizerState* prepL(int i, int t, struct BitMatrix* L);

double sampledProjector(struct Projector *P, struct BitMatrix *L, int exact, double norm, int samples);
double singleProjectorSample(struct Projector *P, struct BitMatrix *L, int exact);
Complex exactProjectorWork(int i, struct Projector *P, struct BitMatrix *L, int exact);

// sort compare function
int cmpfunc(const void* a, const void* b) {
   return *(double*)a - *(double*)b;
}

/******************* Code **********************/
// Master function. Calculate median of several means. 
double multiSampledProjector(struct Projector *P, struct BitMatrix *L, int exact, double norm, int samples, int bins) {
    if (bins == 1) return sampledProjector(P, L, exact, norm, samples);
   
    double* binVals = malloc(bins * sizeof(double));
    for (int i = 0; i < bins; i++) {
        binVals[i] = sampledProjector(P, L, exact, norm, samples);
    }

    qsort(binVals, bins, sizeof(double), cmpfunc);

    double out = 0;
    if (bins % 2 == 1) out = binVals[(bins-1)/2];
    else {
        out = (binVals[bins/2] + binVals[bins/2 - 1])/2;
    }
    
    free(binVals);
    return out;
}


// Master function. Calculate mean of several samples.
double sampledProjector(struct Projector* P, struct BitMatrix* L, int exact, double norm, int samples) {
    // empty projector
    if (P->Nstabs == 0) return pow(norm, 2);

    int t = P->Nqubits;

    // clifford circuit
    if (t == 0) {
        double sum = 1; // include identity
        for (int i = 0; i < P->Nstabs; i++) {
            double ph = (double)BitVectorGet(P->phaseSign, i)*2;
            ph += (double)BitVectorGet(P->phaseComplex, i);
            if (ph == 0) sum += 1;
            if (ph == 2) sum -= 1;
        } 

        return pow(norm, 2) * sum/(1 + (double)P->Nstabs);
    }

    // parallel code
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double total = 0;
    for (int dest = 1; dest < world_size; dest++) {
        sendInt(2, dest); // command
        sendProjector(P, dest);
        sendInt(samples, dest);
    }    

    for (int i = 0; i < samples; i += world_size) {
        total += singleProjectorSample(P, L, exact);
    }

    for (int src = 1; src < world_size; src++) {
        total += recvDouble(src);
    }

    return total/samples;
}


// Evaluate 2^t * || <theta| P |H^t> ||^2 for a random theta.
double singleProjectorSample(struct Projector *P, struct BitMatrix *L, int exact) {
    int t = P->Nqubits;

    int k;
    if (!exact) k = L->rows;

    // Sample random stabilizer state
	struct StabilizerState *theta = randomStabilizerState(t);

    // project state onto P
    double projfactor = 1;

    for (int i = 0; i < P->Nstabs; i++) {
        int m = BitVectorGet(P->phaseSign, i)*2;
        m += BitVectorGet(P->phaseComplex, i);
        struct BitVector* zeta = BitMatrixGetRow(P->zs, i);
        struct BitVector* xi = BitMatrixGetRow(P->xs, i);

        double res = measurePauli(theta, m, zeta, xi);
        projfactor *= res;

        BitVectorFree(zeta);
        BitVectorFree(xi);

        if (res == 0) {
            freeStabilizerState(theta);
            return 0;
        }
    } 


    struct StabilizerState *phi;
    Complex total = {0,0};

    if (exact) {
        int size = pow(2, ceil((double)t / 2));
        for (int i = 0; i < size; i+=1) {
            phi = prepH(i, t);
            Complex innerProd = innerProduct(theta, phi);
            total = ComplexAdd(total, innerProd);
            freeStabilizerState(phi);
        }
    } else {
        int size = pow(2, k);
        for (int i = 0; i < size; i+=1) {
            phi = prepL(i, t, L);
            Complex innerProd = innerProduct(theta, phi);
            total = ComplexAdd(total, innerProd);
            freeStabilizerState(phi);
        }
    }
    
    freeStabilizerState(theta);

    double out = pow(2, t) * ComplexMagSquare(ComplexMulReal(total, projfactor));
    return out;
}


// Master function. Calculate projector exactly. 
double exactProjector(struct Projector* P, struct BitMatrix* L, int exact, double norm) {
    // empty projector
    if (P->Nstabs == 0) return pow(norm, 2);

    int t = P->Nqubits;
    int k;
    if (!exact) k = L->rows;

    // clifford circuit
    if (t == 0) {
        double sum = 1; // include identity
        for (int i = 0; i < P->Nstabs; i++) {
            double ph = (double)BitVectorGet(P->phaseSign, i)*2;
            ph += (double)BitVectorGet(P->phaseComplex, i);
            if (ph == 0) sum += 1;
            if (ph == 2) sum -= 1;
        } 

        return sum/(1 + (double)P->Nstabs);
    }
    
    // parallel code
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int size;
    if (exact) size = ceil((double)t / 2);
    else size = k;

    int kRange = pow(2,size - 1) * (pow(2, size) +1);  // chi * (chi + 1)/2

    Complex total = {0,0};
    Complex part;
    for (int dest = 1; dest < world_size; dest++) {
        sendInt(2, dest); // command
        sendProjector(P, dest);
        sendInt(kRange, dest);
    }    

    for (int l = 0; l < kRange; l += world_size) {
        part = exactProjectorWork(l, P, L, exact);
        total = ComplexAdd(total, part);
    }

    for (int src = 1; src < world_size; src++) {
        part = recvComplex(src);
        total = ComplexAdd(total, part);
    }


    return ComplexMag(total);
}


// Work function for exactProjector.
Complex exactProjectorWork(int l, struct Projector* P, struct BitMatrix* L, int exact) {
    int t = P->Nqubits;

    int chi;
    if (exact) chi = pow(2, ceil((double)t / 2));
    else chi = pow(2, L->rows);

    int i = 0;
    while (l >= chi - i) {
        l -= chi - i;
        i += 1;
    }
    int j = l + i;

    struct StabilizerState* theta;
    if (exact) theta = prepH(i, t);
    else theta = prepL(i, t, L);

    // Project theta
    double projfactor = 1;

    for (int r = 0; r < P->Nstabs; r++) {
        int m = BitVectorGet(P->phaseSign, r)*2;
        m += BitVectorGet(P->phaseComplex, r);
        struct BitVector* zeta = BitMatrixGetRow(P->zs, r);
        struct BitVector* xi = BitMatrixGetRow(P->xs, r);
       
        double res = measurePauli(theta, m, zeta, xi);
        projfactor *= res;
        
        BitVectorFree(zeta);
        BitVectorFree(xi);

        if (res == 0) {
            freeStabilizerState(theta);
            Complex out = {0,0};
            return out;
        }
    } 
    // Evaluate other components
    struct StabilizerState* phi;

    // Diagonal term
    if (exact) phi = prepH(j, t);
    else phi = prepL(j, t, L);

    Complex innerProd = innerProduct(theta, phi);

    freeStabilizerState(phi);
    freeStabilizerState(theta);
   
    if (i == j) {
        return ComplexMulReal(innerProd, projfactor); 
    } else {
        innerProd.re *= 2*projfactor;
        innerProd.im = 0;
        return innerProd; 
    }
}
