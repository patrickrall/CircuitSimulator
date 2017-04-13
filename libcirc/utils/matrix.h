#include <stdio.h>
#include <stdlib.h>

struct BitVector {
    unsigned int size;
    unsigned char* data;
};

struct BitVector* newBitVector(unsigned int size);  // alloc and init to 0
struct BitVector* newBitVectorRandom(unsigned int size);  // alloc and init randomly

void BitVectorFree(struct BitVector* vec);
void BitVectorCopy(struct BitVector* vec1, struct BitVector* vec2); // vec2 <- vec1
void BitVectorPrint(struct BitVector* vec);

unsigned int BitVectorGet(struct BitVector* vec, unsigned int loc);
void BitVectorSet(struct BitVector* vec, unsigned int loc, unsigned int value);

// number of positions where they coincide. Do the mod 2 yourself.
unsigned int BitVectorInner(struct BitVector* vec1, struct BitVector* vec2);

void BitVectorXorSet(struct BitVector* vec1, struct BitVector* vec2); // vec2 <- vec1 ^ vec2

struct BitMatrix {
    unsigned int rows;
    unsigned int cols;
    unsigned char* data;
};

struct BitMatrix* newBitMatrixZero(unsigned int rows, unsigned int cols);  // alloc and init to 0
struct BitMatrix* newBitMatrixIdentity(unsigned int rows);  // alloc and init to identity
struct BitMatrix* newBitMatrixRandom(unsigned int rows, unsigned int cols);  // alloc and init randomly

void BitMatrixFree(struct BitMatrix* mat);
void BitMatrixSetZero(struct BitMatrix* mat);
void BitMatrixCopy(struct BitMatrix* mat1, struct BitMatrix* mat2); // mat2 <- mat1
void BitMatrixPrint(struct BitMatrix* mat);

unsigned int BitMatrixGet(struct BitMatrix* mat, unsigned int row, unsigned int col);
void BitMatrixSet(struct BitMatrix* mat, unsigned int row, unsigned int col, unsigned int value);

void BitMatrixSetCol(struct BitMatrix* mat, struct BitVector* vec, unsigned int col);
void BitMatrixSetRow(struct BitMatrix* mat, struct BitVector* vec, unsigned int row);

struct BitVector* BitMatrixGetCol(struct BitMatrix* mat, unsigned int col);
struct BitVector* BitMatrixGetRow(struct BitMatrix* mat, unsigned int row);

void BitMatrixSwapCols(struct BitMatrix* mat, unsigned int col1, unsigned int col2);
void BitMatrixSwapRows(struct BitMatrix* mat, unsigned int row1, unsigned int row2);

struct BitMatrix* BitMatrixTranspose(struct BitMatrix* mat);
void BitMatrixTransposeSet(struct BitMatrix* mat);

struct BitMatrix* BitMatrixMulMatrix(struct BitMatrix* mat1, struct BitMatrix* mat2);
void BitMatrixMulMatrixSet(struct BitMatrix* mat1, struct BitMatrix* mat2); // mat2 <- mat1*mat2;

struct BitVector* BitMatrixMulVector(struct BitMatrix* mat, struct BitVector* vec);
void BitMatrixMulVectorSet(struct BitMatrix* mat, struct BitVector* vec); // vec <- mat*vec

unsigned int BitMatrixRank(struct BitMatrix* mat);
