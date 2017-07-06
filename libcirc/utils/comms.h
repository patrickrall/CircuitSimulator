#include "matrix.h"

// Projectors
struct Projector {
    int Nstabs;
    int Nqubits;
    struct BitVector* phaseSign;
    struct BitVector* phaseComplex;
    struct BitMatrix* xs;
    struct BitMatrix* zs; 
};

struct Projector* readProjector(FILE* stream);
void printProjector(struct Projector* P);
void freeProjector(struct Projector* P);


// Communication
void sendInt(int i, int dest);
int recvInt(int src);

void sendDouble(double i, int dest);
double recvDouble(int src);

void sendBitVector(struct BitVector* vec, int dest);
struct BitVector* recvBitVector(int src);

void sendBitMatrix(struct BitMatrix* mat, int dest);
struct BitMatrix* recvBitMatrix(int src);

void sendComplex(Complex z, int dest);
Complex recvComplex(int src);

void sendProjector(struct Projector* P, int dest);
struct Projector* recvProjector(int src);

// implemented, but not needed anywhere
// uses void* to avoid inclusion difficulties
// void sendStabilizerState(void* state, int dest);
// void* recvStabilizerState(int src);
