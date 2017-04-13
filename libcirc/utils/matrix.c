#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

/*************************** Bit Vector *************************/

/*
struct BitVector {
    unsigned int size;
    unsigned char* data;
};
*/

// alloc and init to 0
struct BitVector* newBitVector(unsigned int size) {
    assert(size > 0);

    struct BitVector* vec = (struct BitVector*)malloc(sizeof(struct BitVector));
    vec->size = size;
    
    unsigned int bytes = (size + 7)/8; // round up

    vec->data = (unsigned char*)malloc(bytes);
    for (int i = 0; i < bytes; i++) {
        vec->data[i] = 0x00;
    }

    return vec;
}


void BitVectorFree(struct BitVector* vec) {
    free(vec->data);
    free(vec);
}


// copy vec1 to vec2
void BitVectorCopy(struct BitVector* vec1, struct BitVector* vec2) {
    assert(vec1->size == vec2->size);
    unsigned int bytes = (vec1->size + 7)/8; // round up

    for (int i = 0; i < bytes; i++) {
        vec2->data[i] = vec1->data[i];
    }

}

unsigned int BitVectorGet(struct BitVector* vec, unsigned int loc) {
    assert(vec->size > loc);
    
    unsigned int byte = loc/8;
    unsigned int mask = 0x80;
    mask >>= loc % 8;
    return (vec->data[byte] & mask) > 0;
};


void BitVectorSet(struct BitVector* vec, unsigned int loc, unsigned int value) {
    assert(vec->size > loc);

    unsigned int byte = loc/8;
    unsigned int mask = 0x80;
    mask >>= loc % 8;

    if (value % 2 == 1) {
        vec->data[byte] |= mask;
    } else {
        vec->data[byte] &= mask ^ 0xFF;
    }
}


void BitVectorPrint(struct BitVector* vec) {
    printf("[");
    for (int i = 0; i < vec->size; i++) {
        printf("%d", BitVectorGet(vec, i));
    } 
    printf("]\n");
}


unsigned int BitVectorInner(struct BitVector* vec1, struct BitVector* vec2) {
    assert(vec1->size == vec2->size);
    unsigned int inner = 0;
    unsigned int mask, byte;

    unsigned int bytes = vec1->size/8; // round down
    for (byte = 0; byte < bytes; byte++) {
        mask = 0x80;
        while (mask > 0) {
            if ((vec1->data[byte] & mask) > 0 && (vec2->data[byte] & mask) > 0) {
                inner += 1;
            }
            mask >>= 1;
        }
    } 

    unsigned int extra = vec1->size - 8*bytes; 
    mask = 0x80;
    for (int bit = 0; bit < extra; bit++) {
        if ((vec1->data[byte] & mask) > 0 && (vec2->data[byte] & mask) > 0) {
            inner += 1;
        }
        mask >>= 1;
    }


    return inner;
}



/*************************** Bit Matrix *************************/

/*
struct BitMatrix {
    unsigned int rows;
    unsigned int cols;
    unsigned char* data;
};
*/

// alloc and init to 0
struct BitMatrix* newBitMatrixZero(unsigned int rows, unsigned int cols) {
    assert(rows > 0);
    assert(cols > 0);

    struct BitMatrix* mat = (struct BitMatrix*)malloc(sizeof(struct BitMatrix));
    mat->rows = rows;
    mat->cols = cols;
    
    unsigned int bytes = (rows*cols + 7)/8; // round up

    mat->data = (unsigned char*)malloc(bytes);
    for (int i = 0; i < bytes; i++) {
        mat->data[i] = 0x00;
    }

    return mat;
}

// alloc and init to identity
struct BitMatrix* newBitMatrixIdentity(unsigned int rows) {
    struct BitMatrix* mat = newBitMatrixZero(rows, rows);

    for (int i = 0; i < rows; i++) {
        BitMatrixSet(mat, i, i, 1);
    }

    return mat;
}

void BitMatrixFree(struct BitMatrix* mat) {
    free(mat->data);
    free(mat);
}

void BitMatrixCopy(struct BitMatrix* mat1, struct BitMatrix* mat2) {
    assert(mat1->rows == mat2->rows);
    assert(mat1->cols == mat2->cols);

    unsigned int bytes = ((mat1->rows * mat1->cols) + 7)/8; // round up

    for (int i = 0; i < bytes; i++) {
        mat2->data[i] = mat1->data[i];
    }
}


void BitMatrixPrint(struct BitMatrix* mat) {
    printf("[");
    for (int i = 0; i < mat->rows; i++) {
        if (i != 0) printf("\n ");
        printf("[");
        for (int j = 0; j < mat->cols; j++) {
            printf("%d", BitMatrixGet(mat, i, j));
        }
        printf("]");
    } 
    printf("]\n");

}

unsigned int BitMatrixGet(struct BitMatrix* mat, unsigned int row, unsigned int col) {
    assert(mat->cols > col);
    assert(mat->rows > row);
   
    unsigned int loc = row*mat->cols + col;
    unsigned int byte = loc / 8;
    unsigned int mask = 0x80;
    mask >>= loc % 8;
    return (mat->data[byte] & mask) > 0;
}

void BitMatrixSet(struct BitMatrix* mat, unsigned int row, unsigned int col, unsigned int value) {
    assert(mat->cols > col);
    assert(mat->rows > row);

    unsigned int loc = row*mat->cols + col;
    unsigned int byte = loc / 8;
    unsigned int mask = 0x80;
    mask >>= loc % 8;

    if (value % 2 == 1) {
        mat->data[byte] |= mask;
    } else {
        mat->data[byte] &= mask ^ 0xFF;
    }
}

void BitMatrixColGet(struct BitMatrix* mat, struct BitVector* vec, unsigned int col);
void BitMatrixColSet(struct BitMatrix* mat, struct BitVector* vec, unsigned int col);
void BitMatrixRowGet(struct BitMatrix* mat, struct BitVector* vec, unsigned int row);
void BitMatrixRowSet(struct BitMatrix* mat, struct BitVector* vec, unsigned int row);

void BitMatrixSwapCols(struct BitMatrix* mat, unsigned int col1, unsigned int col2);
void BitMatrixSwapRows(struct BitMatrix* mat, unsigned int row1, unsigned int row2);

void BitMatrixTranspose(struct BitMatrix* mat);

struct BitMatrix* BitMatrixMulMatrix(struct BitMatrix* mat1, struct BitMatrix* mat2);
void BitMatrixMulMatrixSet(struct BitMatrix* mat1, struct BitMatrix* mat2); // mat2 <- mat1*mat2;

struct BitVector* BitVectorMulMatrix(struct BitMatrix* mat, struct BitVector* vec);
void BitVectorMulMatrixSet(struct BitMatrix* mat, struct BitVector* vec); // vec <- mat*vec
