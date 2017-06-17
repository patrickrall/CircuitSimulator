#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "comms.h"
#include "mpi.h"


/************* Helpers for reading and printing projectors *************/
struct Projector* readProjector(FILE* stream) {
    struct Projector *P = (struct Projector *)malloc(sizeof(struct Projector));
    fscanf(stream,"%d", &(P->Nstabs));
    fscanf(stream,"%d", &(P->Nqubits));

    if (P->Nstabs == 0) return P;

    P->phaseSign = newBitVector(P->Nstabs);
    P->phaseComplex = newBitVector(P->Nstabs);
    P->xs = newBitMatrixZero(P->Nstabs, P->Nqubits);
    P->zs = newBitMatrixZero(P->Nstabs, P->Nqubits);
    
    int v;
    for (int i = 0; i < P->Nstabs; i++) {
        fscanf(stream,"%d", &v);
        BitVectorSet(P->phaseComplex, i, v % 2);
        BitVectorSet(P->phaseSign, i, (v/2) % 2);

        for (int j = 0; j < P->Nqubits; j++) {
            fscanf(stream,"%d", &v);
            BitMatrixSet(P->xs, i, j, v);
            
            fscanf(stream,"%d", &v);
            BitMatrixSet(P->zs, i, j, v);
        }
    }
    return P;
}

void printProjector(struct Projector *P) {
    for (int i = 0; i < P->Nstabs; i++) {
        int tmpphase = 2*BitVectorGet(P->phaseSign, i) + BitVectorGet(P->phaseComplex, i);

        struct BitVector* xs = BitMatrixGetRow(P->xs, i);
        struct BitVector* zs = BitMatrixGetRow(P->zs, i);

        char stab[20];
        stab[0] ='\0';

        for (int j = 0; j < P->Nqubits; j++) {
            int x = BitVectorGet(xs, j);
            int z = BitVectorGet(zs, j);

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

        BitVectorFree(xs);
        BitVectorFree(zs);
    }
}


void freeProjector(struct Projector *P) {

    if (P->Nstabs > 0) {
        BitVectorFree(P->phaseSign);
        BitVectorFree(P->phaseComplex);
        BitMatrixFree(P->xs);
        BitMatrixFree(P->zs);
    }
    
    free(P);
}

// Comms
//----------- int macro
void sendInt(int i, int dest) {
    MPI_Send(&i, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
}
int recvInt(int src) {
    int buff;
    MPI_Recv(&buff, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return buff;
}

//----------- double macro
void sendDouble(double i, int dest) {
    MPI_Send(&i, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
}
double recvDouble(int src) {
    double buff;
    MPI_Recv(&buff, 1, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return buff;
}


//----------- BitVector
void sendBitVector(struct BitVector* vec, int dest) {
    int size = (int)vec->size;
    sendInt(size, dest); 
    for (int i = 0; i < size; i++) {
        sendInt(BitVectorGet(vec, i), dest);
    }
}
struct BitVector* recvBitVector(int src) {
    int size = recvInt(src);
    struct BitVector* vec = newBitVector(size);
    for (int i = 0; i < size; i++) {
        BitVectorSet(vec, i, recvInt(src));
    }
    return vec;
}

//----------- gsl_matrix
void sendBitMatrix(struct BitMatrix* mat, int dest){
    int rows = (int)mat->rows;
    int cols = (int)mat->cols;

    sendInt(rows, dest); 
    sendInt(cols, dest); 

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            sendInt(BitMatrixGet(mat, i, j), dest);
        }
    }
} 
struct BitMatrix* recvBitMatrix(int src) {
    int rows = recvInt(src);
    int cols = recvInt(src);

    struct BitMatrix* mat = newBitMatrixZero(rows, cols);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            BitMatrixSet(mat, i, j, recvInt(src));
        }
    }
    return mat;
}


//---------- gsl_complex
void sendComplex(Complex z, int dest) {
    sendDouble(z.re, dest);
    sendDouble(z.im, dest);
}
Complex recvComplex(int src) {
    double real = recvDouble(src);
    double imag = recvDouble(src);
    Complex out = {real, imag};
    return out;
}

//---------- projectors
void sendProjector(struct Projector* P, int dest) {
    sendInt(P->Nstabs, dest);
    sendInt(P->Nqubits, dest);
    if (P->Nstabs == 0) return;

    sendBitVector(P->phaseSign, dest);
    sendBitVector(P->phaseComplex, dest);
    sendBitMatrix(P->xs, dest);
    sendBitMatrix(P->zs, dest);
}
struct Projector* recvProjector(int src) {
    struct Projector *P = (struct Projector *)malloc(sizeof(struct Projector));
    P->Nstabs = recvInt(src);
    P->Nqubits = recvInt(src);
    if (P->Nstabs == 0) return P;

    P->phaseSign = recvBitVector(src);
    P->phaseComplex = recvBitVector(src);
    P->xs = recvBitMatrix(src);
    P->zs = recvBitMatrix(src);
    return P;
}

//---------- stabilizer states
// implemented, but not needed anywhere
/*
void sendStabilizerState(void* statein, int dest) {
    struct StabilizerState* state = (struct StabilizerState*)statein;
    sendInt(state->n, dest);
    sendInt(state->k, dest);

    sendBitVector(state->h, dest);
    sendBitMatrix(state->G, dest);
    sendBitMatrix(state->Gbar, dest);

    sendInt(state->Q, dest);
    sendBitVector(state->D, dest);
    sendBitMatrix(state->J, dest);
}
void* recvStabilizerState(int src) {
    int n = recvInt(src);
    int k = recvInt(src);

    struct StabilizerState *state = allocStabilizerState(n, k);
    state->h = recvBitVector(src);
    state->G = recvBitMatrix(src);
    state->Gbar = recvBitMatrix(src);

    state->Q = recvInt(src);
    state->D = recvBitVector(src);
    state->J = recvBitMatrix(src);
    return (void*)state;
}
*/
