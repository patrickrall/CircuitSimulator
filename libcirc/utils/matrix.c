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

// alloc and init randomly
struct BitVector* newBitVectorRandom(unsigned int size) {
    assert(size > 0);

    struct BitVector* vec = (struct BitVector*)malloc(sizeof(struct BitVector));
    vec->size = size;
    
    unsigned int bytes = (size + 7)/8; // round up

    vec->data = (unsigned char*)malloc(bytes);
    for (int i = 0; i < bytes; i++) {
        vec->data[i] = rand() % 256;
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

 // vec2 <- vec1 ^ vec2
void BitVectorXorSet(struct BitVector* vec1, struct BitVector* vec2) {
    assert(vec1->size == vec2->size);
    
    unsigned int bytes = (vec1->size + 7)/8; // round up
    for (unsigned int byte = 0; byte < bytes; byte++) {
        vec2->data[byte] = vec1->data[byte] ^ vec2->data[byte];
    }
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

// alloc and init randomly
struct BitMatrix* newBitMatrixRandom(unsigned int rows, unsigned int cols) {
    assert(rows > 0);
    assert(cols > 0);

    struct BitMatrix* mat = (struct BitMatrix*)malloc(sizeof(struct BitMatrix));
    mat->rows = rows;
    mat->cols = cols;
    
    unsigned int bytes = (rows*cols + 7)/8; // round up

    mat->data = (unsigned char*)malloc(bytes);
    for (int i = 0; i < bytes; i++) {
        mat->data[i] = rand() % 256;
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

void BitMatrixSetZero(struct BitMatrix* mat) {
    unsigned int bytes = (mat->rows*mat->cols + 7)/8; // round up
    for (int i = 0; i < bytes; i++) {
        mat->data[i] = 0x00;
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

void BitMatrixSetCol(struct BitMatrix* mat, struct BitVector* vec, unsigned int col) {
    assert(vec->size == mat->rows);
    assert(col < mat->cols);

    unsigned int bytes = (vec->size + 7)/8; // round up
    unsigned int row = 0;
    unsigned int mask, byte;
    for (byte = 0; byte < bytes; byte++) {
        mask = 0x80;
        while (mask > 0) {
            if ((vec->data[byte] & mask) > 0) {
                BitMatrixSet(mat, row, col, 1);
            } else {
                BitMatrixSet(mat, row, col, 0);
            }
            row += 1;
            if (row >= vec->size) break;
            mask >>= 1;
        }
    } 
}


void BitMatrixSetRow(struct BitMatrix* mat, struct BitVector* vec, unsigned int row) {
    assert(vec->size == mat->cols);
    assert(row < mat->rows);

    unsigned int vecbytes = (vec->size + 7)/8; // round up
    unsigned int matbyte, vecbyte, matmask, vecmask, vecloc;

    matbyte = (mat->cols*row) / 8;  
    matmask = 0x80 >> (mat->cols * row) % 8;

    vecloc = 0;
    for (vecbyte = 0; vecbyte < vecbytes; vecbyte++) {
        vecmask = 0x80;
        while (vecmask > 0) {
            if ((vec->data[vecbyte] & vecmask) > 0) {
                mat->data[matbyte] |= matmask;
            } else {
                mat->data[matbyte] &= matmask ^ 0xFF;
            }
                     
            vecmask >>= 1;
            matmask >>= 1;
            if (!(matmask > 0)) {
                matmask = 0x80; 
                matbyte += 1;
            }
            vecloc += 1;
            if (vecloc >= vec->size) break;
        }
        if (vecloc >= vec->size) break;
    }
}

struct BitVector* BitMatrixGetCol(struct BitMatrix* mat, unsigned int col) {
    assert(col < mat->cols);
    struct BitVector* vec = newBitVector(mat->rows);
    
    for (unsigned int row = 0; row < mat->rows; row++) {
        BitVectorSet(vec, row, BitMatrixGet(mat, row, col));
    }

    return vec;
}


struct BitVector* BitMatrixGetRow(struct BitMatrix* mat, unsigned int row) {
    assert(row < mat->rows);
    struct BitVector* vec = newBitVector(mat->cols);

    unsigned int vecbytes = (vec->size + 7)/8; // round up
    unsigned int matbyte, vecbyte, matmask, vecmask, vecloc;
    
    matbyte = (mat->cols*row) / 8;  
    matmask = 0x80 >> (mat->cols * row) % 8;

    vecloc = 0;
    for (vecbyte = 0; vecbyte < vecbytes; vecbyte++) {
        vecmask = 0x80;
        while (vecmask > 0) {
            if ((mat->data[matbyte] & matmask) > 0) {
                vec->data[vecbyte] |= vecmask;
            }
                     
            vecmask >>= 1;
            matmask >>= 1;
            if (!(matmask > 0)) {
                matmask = 0x80; 
                matbyte += 1;
            }
            vecloc += 1;
            if (vecloc >= vec->size) break;
        }
        if (vecloc >= vec->size) break;
    }

    return vec;
}


void BitMatrixSwapCols(struct BitMatrix* mat, unsigned int col1, unsigned int col2) {
    struct BitVector* col1v = BitMatrixGetCol(mat, col1);
    struct BitVector* col2v = BitMatrixGetCol(mat, col2);

    BitMatrixSetCol(mat, col2v, col1);
    BitMatrixSetCol(mat, col1v, col2);

    BitVectorFree(col1v);
    BitVectorFree(col2v);
}

void BitMatrixSwapRows(struct BitMatrix* mat, unsigned int row1, unsigned int row2) {
    struct BitVector* row1v = BitMatrixGetRow(mat, row1);
    struct BitVector* row2v = BitMatrixGetRow(mat, row2);

    BitMatrixSetRow(mat, row2v, row1);
    BitMatrixSetRow(mat, row1v, row2);

    BitVectorFree(row1v);
    BitVectorFree(row2v);
}

struct BitMatrix* BitMatrixTranspose(struct BitMatrix* mat) {
    struct BitMatrix* matT = newBitMatrixZero(mat->cols, mat->rows);

    unsigned int bytes = (mat->rows*mat->cols + 7)/8; // round up
    unsigned int row = 0;
    unsigned int col = 0;
    unsigned int mask;
    for (unsigned int byte = 0; byte < bytes; byte++) {
        mask = 0x80;
        while (mask > 0) {
            if ((mat->data[byte] & mask) > 0) {
                BitMatrixSet(matT, col, row, 1);
            } 

            col += 1;
            if (col >= mat->cols) {
                col = 0;
                row += 1;
                if (row >= mat->rows) break;
            }

            mask >>= 1;
        }
    }

    return matT;
}

void BitMatrixTransposeSet(struct BitMatrix* mat) {
    struct BitMatrix* matT = BitMatrixTranspose(mat);

    mat->rows = matT->rows;
    mat->cols = matT->cols;
    BitMatrixCopy(matT, mat);

    BitMatrixFree(matT);
}

struct BitMatrix* BitMatrixMulMatrix(struct BitMatrix* mat1, struct BitMatrix* mat2) {
    assert(mat1->cols == mat2->rows);
    struct BitMatrix* matOut = newBitMatrixZero(mat1->rows, mat2->cols);

    struct BitVector* vecTmp1;
    struct BitVector* vecTmp2;

    unsigned int bytes = (matOut->cols*matOut->rows + 7)/8; // round up
    unsigned int col = 0;
    unsigned int row = 0;
    unsigned int mask;
    vecTmp1 = BitMatrixGetRow(mat1, row);
    for (unsigned int byte = 0; byte < bytes; byte++) {
        mask = 0x80;
        while (mask > 0) {
            vecTmp2 = BitMatrixGetCol(mat2, col);
            if (BitVectorInner(vecTmp1, vecTmp2) % 2 == 1) {
                matOut->data[byte] |= mask;
            } 
            BitVectorFree(vecTmp2);

            col += 1;
            if (col >= matOut->cols) {
                col = 0;
                row += 1;
                if (row >= matOut->rows) break;
                BitVectorFree(vecTmp1);
                vecTmp1 = BitMatrixGetRow(mat1, row);
            }

            mask >>= 1;
        }
    } 
    BitVectorFree(vecTmp1);

    return matOut;
}

// mat2 <- mat1*mat2;
void BitMatrixMulMatrixSet(struct BitMatrix* mat1, struct BitMatrix* mat2) {
    assert(mat1->cols == mat2->rows); // Multiplication valid
    assert(mat1->rows == mat1->cols); // Input square
    assert(mat1->rows == mat2->cols); // Output same
    
    struct BitMatrix* matTmp = BitMatrixMulMatrix(mat1, mat2);
    BitMatrixCopy(matTmp, mat2);
    BitMatrixFree(mat2);
}

struct BitVector* BitMatrixMulVector(struct BitMatrix* mat, struct BitVector* vec) {
    assert(mat->cols == vec->size);
    struct BitVector* vecOut = newBitVector(mat->rows);
    struct BitVector* vecTmp;

    unsigned int bytes = (vecOut->size + 7)/8; // round up
    unsigned int pos = 0;
    unsigned int mask;
    for (unsigned int byte = 0; byte < bytes; byte++) {
        mask = 0x80;
        while (mask > 0) {
            vecTmp = BitMatrixGetRow(mat, pos);

            if (BitVectorInner(vecTmp, vec) % 2 == 1) {
                vecOut->data[byte] |= mask;
            } 

            BitVectorFree(vecTmp);

            pos += 1;
            if (pos >= vecOut->size) break;
            mask >>= 1;
        }
    } 

    return vecOut;
}

// vec <- mat*vec
void BitMatrixMulVectorSet(struct BitMatrix* mat, struct BitVector* vec) {
    assert(mat->cols == vec->size);
    assert(mat->cols == mat->rows);
    
    struct BitVector* vecTmp = BitMatrixMulVector(mat, vec);
    BitVectorCopy(vecTmp, vec);

    BitVectorFree(vecTmp);
}


unsigned int BitMatrixRank(struct BitMatrix* mat) {
    struct BitMatrix* null = newBitMatrixIdentity(mat->cols);
    unsigned int rank = mat->cols;

    struct BitMatrix* good = newBitMatrixZero(mat->rows, mat->cols);
    struct BitMatrix* bad = newBitMatrixZero(mat->rows, mat->cols);
    unsigned int numgood, numbad, mask;

    for (unsigned int row = 0; row < mat->rows; row++) {
        struct BitVector* rowVec = BitMatrixGetRow(mat, row);
        numgood = 0;
        numbad = 0;
   
        for (unsigned int i = 0; i < rank; i++) {
            struct BitVector* nullVec = BitMatrixGetRow(null, i);
            
            if (BitVectorInner(nullVec, rowVec) % 2 == 0) {
                BitMatrixSetRow(good, nullVec, numgood);
                numgood += 1;
            } else {
                BitMatrixSetRow(bad, nullVec, numbad);
                numbad += 1;
            }
        }

        // copy good into null
        for (rank = 0; rank < numgood; rank++) {
            struct BitVector* goodVec = BitMatrixGetRow(good, rank);
            BitMatrixSetRow(null, goodVec, rank);
        }

        if (numbad > 1) {
            struct BitVector* flipVec = BitMatrixGetRow(bad, 0);

            for (; rank+1 < numbad+numgood; rank++) {
                struct BitVector* badVec = BitMatrixGetRow(bad, 1+rank-numgood);
                BitVectorXorSet(flipVec, badVec);
                BitMatrixSetRow(null, badVec, rank);
            }
        }
    }
    
    BitMatrixFree(null);
    BitMatrixFree(good);
    BitMatrixFree(bad);

    rank = mat->cols - rank;
    return rank;
}
