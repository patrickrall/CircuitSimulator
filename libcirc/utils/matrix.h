#include <stdio.h>
#include <stdlib.h>

struct BitVector {
    int size;
    int* data;
}

struct BitVector* newBitVector(int size);  // alloc and init to 0
void freeBitVector(struct BitVector* vec);
void copyBitVector(struct BitVector* vec1, struct BitVector* vec2);

void BitVectorGet(struct BitVector* vec, int loc);
void BitVectorSet(struct BitVector* vec, int loc, int value);

void BitVectorInner(struct BitVector* vec1, struct BitVector* vec2);

struct BitMatrix {
    int size1;
    int size2;
    int* data;
}

struct BitMatrix* newBitMatrixZero(int size1, int size2);  // alloc and init to 0
struct BitMatrix* newBitMatrixEye(int size1, int size2);  // alloc and init to identity

void freeBitMatrix(struct BitMatrix* mat);
void copyBitMatrix(struct BitMatrix* mat1, struct BitMatrix* mat2);

void BitMatrixGet(struct BitMatrix* mat, int row, int col);
void BitMatrixSet(struct BitMatrix* mat, int row, int col, int value);

void getBitMatrixCol(struct BitMatrix* mat, int col);
void getBitMatrixRow(struct BitMatrix* mat, int row);

void BitMatrixSwapCols(struct BitMatrix* mat, int col1, int col2);
void BitMatrixSwapRows(struct BitMatrix* mat, int row1, int row2);
