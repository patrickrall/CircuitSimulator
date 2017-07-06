#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include <gsl/gsl_vector.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_linalg.h>

/*************************** Complex Number *************************/

/*
typedef struct {
    double dat[2]; 
} Complex;
*/

Complex ComplexPolar(double r, double theta) {
    Complex z = {r * cos(theta), r * sin(theta)};
    return z;
}

void ComplexPrint(Complex z) {
    printf("%f + %fi\n", z.re, z.im);
}

Complex ComplexAdd(Complex z, Complex w) {
    Complex x = {z.re + w.re, z.im + w.im};
    return x;
}

Complex ComplexMul(Complex z, Complex w) {
    Complex x = {z.re*w.re - z.im*w.im, z.re*w.im + w.re*z.im};
    return x;
}

Complex ComplexMulReal(Complex z, double r) {
    Complex x = {z.re*r, z.im*r};
    return x;
}

double ComplexMag(Complex z) {
    return sqrt(ComplexMagSquare(z));
}

double ComplexMagSquare(Complex z) {
    return z.re*z.re + z.im*z.im;
}

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

    vec->data = (unsigned char*)gsl_vector_alloc(vec->size);
    gsl_vector_set_zero((gsl_vector*)vec->data);

    return vec;
}

// alloc and init randomly
struct BitVector* newBitVectorRandom(unsigned int size) {
    assert(size > 0);

    struct BitVector* vec = (struct BitVector*)malloc(sizeof(struct BitVector));
    vec->size = size;
    vec->data = (unsigned char*)gsl_vector_alloc(vec->size);

    BitVectorSetRandom(vec);

    return vec;
}

void BitVectorSetRandom(struct BitVector* vec) {
    for (unsigned int i = 0; i < vec->size; i++) {
        BitVectorSet(vec, i, rand());
    }
}


void BitVectorSetZero(struct BitVector* vec) {
    gsl_vector_set_zero((gsl_vector*)vec->data);
}




void BitVectorFree(struct BitVector* vec) {
    gsl_vector_free((gsl_vector*)vec->data);
    free(vec);
}


// copy vec2 to vec1
void BitVectorCopy(struct BitVector* vec1, struct BitVector* vec2) {
    assert(vec1->size == vec2->size);

    gsl_vector_memcpy((gsl_vector*)vec1->data, (gsl_vector*)vec2->data);

}

unsigned int BitVectorGet(struct BitVector* vec, unsigned int loc) {
    assert(vec->size > loc);
    
    return (int)gsl_vector_get((gsl_vector*)vec->data, loc) % 2; 
};


void BitVectorSet(struct BitVector* vec, unsigned int loc, unsigned int value) {
    assert(vec->size > loc);

    gsl_vector_set((gsl_vector*)vec->data, loc, (double) (value % 2)); 
}

void BitVectorFlip(struct BitVector* vec, unsigned int loc) {
    assert(vec->size > loc);

    BitVectorSet(vec, loc, 1-BitVectorGet(vec, loc));
}

int BitVectorSame(struct BitVector* vec1, struct BitVector* vec2) {
    assert(vec1->size == vec2->size);

    for (unsigned int i = 0; i < vec1->size; i++) {
        if (BitVectorGet(vec1, i) != BitVectorGet(vec2, i)) return 0;
    }
    return 1;
}


void BitVectorPrint(struct BitVector* vec) {
    printf("[");
    for (unsigned int i = 0; i < vec->size; i++) {
        printf("%d", BitVectorGet(vec, i));
    } 
    printf("]\n");
}


unsigned int BitVectorInner(struct BitVector* vec1, struct BitVector* vec2) {
    assert(vec1->size == vec2->size);
   
    double inner;
    gsl_blas_ddot((gsl_vector*)vec1->data, (gsl_vector*)vec2->data, &inner);

    return (unsigned int)inner;
}

 // vec1 <- vec1 ^ vec2
void BitVectorXorSet(struct BitVector* vec1, struct BitVector* vec2) {
    assert(vec1->size == vec2->size);
   
    for (unsigned int i = 0; i < vec1->size; i++) {
        BitVectorSet(vec1, i, BitVectorGet(vec1,i) + BitVectorGet(vec2,i) % 2);
    }
}

// helper
void BitVectorMod2(struct BitVector* vec) {
    for (unsigned int i = 0; i < vec->size; i++) {
        BitVectorSet(vec, i, BitVectorGet(vec, i));
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
    
    mat->data = (unsigned char*)gsl_matrix_alloc(mat->rows, mat->cols);
    gsl_matrix_set_zero((gsl_matrix*)mat->data);

    return mat;
}

// alloc and init to identity
struct BitMatrix* newBitMatrixIdentity(unsigned int rows) {
    struct BitMatrix* mat = newBitMatrixZero(rows, rows);

    gsl_matrix_set_identity((gsl_matrix*)mat->data);

    return mat;
}

// alloc and init randomly
struct BitMatrix* newBitMatrixRandom(unsigned int rows, unsigned int cols) {
    assert(rows > 0);
    assert(cols > 0);

    struct BitMatrix* mat = newBitMatrixZero(rows, rows);
    BitMatrixSetRandom(mat); 
    return mat;
}


void BitMatrixFree(struct BitMatrix* mat) {
    gsl_matrix_free((gsl_matrix*)mat->data);
    free(mat);
}

void BitMatrixCopy(struct BitMatrix* mat1, struct BitMatrix* mat2) {
    assert(mat1->rows == mat2->rows);
    assert(mat1->cols == mat2->cols);

    gsl_matrix_memcpy((gsl_matrix*)mat1->data, (gsl_matrix*)mat2->data);
}

void BitMatrixSetZero(struct BitMatrix* mat) {
    gsl_matrix_set_zero((gsl_matrix*)mat->data);
}

void BitMatrixSetRandom(struct BitMatrix* mat) {
    for (unsigned int i = 0; i < mat->rows; i++) {
        for (unsigned int j = 0; j < mat->cols; j++) {
            BitMatrixSet(mat, i, j, rand());
        }
    }
}

void BitMatrixSetIdentity(struct BitMatrix* mat) {
    gsl_matrix_set_identity((gsl_matrix*)mat->data);
}

void BitMatrixPrint(struct BitMatrix* mat) {
    printf("[");
    for (unsigned int i = 0; i < mat->rows; i++) {
        if (i != 0) printf("\n ");
        printf("[");
        for (unsigned int j = 0; j < mat->cols; j++) {
            printf("%d", BitMatrixGet(mat, i, j));
        }
        printf("]");
    } 
    printf("]\n");

}

unsigned int BitMatrixGet(struct BitMatrix* mat, unsigned int row, unsigned int col) {
    assert(mat->cols > col);
    assert(mat->rows > row);
   
    return (int)gsl_matrix_get((gsl_matrix*)mat->data, row, col) % 2; 
}

void BitMatrixSet(struct BitMatrix* mat, unsigned int row, unsigned int col, unsigned int value) {
    assert(mat->cols > col);
    assert(mat->rows > row);

    gsl_matrix_set((gsl_matrix*)mat->data, row, col, (double)(value % 2)); 
}

void BitMatrixFlip(struct BitMatrix* mat, unsigned int row, unsigned int col) {
    assert(mat->cols > col);
    assert(mat->rows > row);

    BitMatrixSet(mat, row, col, 1-BitMatrixGet(mat, row, col)); 
}

int BitMatrixSame(struct BitMatrix* mat1, struct BitMatrix* mat2) {
    assert(mat1->rows == mat2->rows);
    assert(mat1->cols == mat2->cols);

    for (unsigned int i = 0; i < mat1->rows; i++) {
        for (unsigned int j = 0; j < mat1->cols; j++) {
            if (BitMatrixGet(mat1, i, j) != BitMatrixGet(mat2, i, j)) return 0;
        }
    }
    return 1;
}

void BitMatrixXorSet(struct BitMatrix* mat1, struct BitMatrix* mat2) { // mat1 <- mat1 ^ mat2
    assert(mat1->cols == mat2->cols);
    assert(mat1->rows == mat2->cols);

    for (unsigned int i = 0; i < mat1->rows; i++) {
        for (unsigned int j = 0; j < mat1->cols; j++) {
            BitMatrixSet(mat1, i, j, BitMatrixGet(mat1,i,j) + BitMatrixGet(mat2,i,j) % 2);
        }
    }
}

void BitMatrixSetCol(struct BitMatrix* mat, struct BitVector* vec, unsigned int col) {
    assert(vec->size == mat->rows);
    assert(col < mat->cols);

    gsl_matrix_set_col((gsl_matrix*)mat->data, col, (gsl_vector*)vec->data);
}


void BitMatrixSetRow(struct BitMatrix* mat, struct BitVector* vec, unsigned int row) {
    assert(vec->size == mat->cols);
    assert(row < mat->rows);


    gsl_matrix_set_row((gsl_matrix*)mat->data, row, (gsl_vector*)vec->data);
}

struct BitVector* BitMatrixGetCol(struct BitMatrix* mat, unsigned int col) {
    assert(col < mat->cols);
    
    struct BitVector* vec = newBitVector(mat->rows);
    gsl_matrix_get_col((gsl_vector*)vec->data, (gsl_matrix*)mat->data, col);

    return vec;
}


struct BitVector* BitMatrixGetRow(struct BitMatrix* mat, unsigned int row) {
    assert(row < mat->rows);
    
    struct BitVector* vec = newBitVector(mat->cols);

    gsl_matrix_get_row((gsl_vector*)vec->data, (gsl_matrix*)mat->data, row);

    return vec;
}


void BitMatrixSwapCols(struct BitMatrix* mat, unsigned int col1, unsigned int col2) {
    gsl_matrix_swap_columns((gsl_matrix*)mat->data, col1, col2); 
}

void BitMatrixSwapRows(struct BitMatrix* mat, unsigned int row1, unsigned int row2) {
    gsl_matrix_swap_rows((gsl_matrix*)mat->data, row1, row2); 

}

// helper
void BitMatrixMod2(struct BitMatrix* mat) {
    for (unsigned int i = 0; i < mat->rows; i++) {
        for (unsigned int j = 0; j < mat->cols; j++) {
            BitMatrixSet(mat, i, j, BitMatrixGet(mat,i,j));
        }
    }
}


struct BitMatrix* BitMatrixTranspose(struct BitMatrix* mat) {
    struct BitMatrix* matT = newBitMatrixZero(mat->cols, mat->rows);
    
    gsl_matrix_transpose_memcpy((gsl_matrix*)matT->data, (gsl_matrix*)mat->data);

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
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1, (gsl_matrix*)mat1->data, (gsl_matrix*)mat2->data, 0, (gsl_matrix*)matOut->data);
    
    BitMatrixMod2(matOut);
   
    return matOut;
}

// mat2 <- mat1*mat2;
void BitMatrixMulMatrixLeft(struct BitMatrix* mat1, struct BitMatrix* mat2) {
    struct BitMatrix* matTmp = BitMatrixMulMatrix(mat1, mat2);
    BitMatrixCopy(mat2, matTmp);
    BitMatrixFree(matTmp);
}


// mat1 <- mat1*mat2;
void BitMatrixMulMatrixRight(struct BitMatrix* mat1, struct BitMatrix* mat2) {
    struct BitMatrix* matTmp = BitMatrixMulMatrix(mat1, mat2);
    BitMatrixCopy(mat1, matTmp);
    BitMatrixFree(matTmp);
}


struct BitVector* BitMatrixMulVector(struct BitMatrix* mat, struct BitVector* vec) {
    assert(mat->cols == vec->size);
   
    struct BitVector* vecOut = newBitVector(mat->rows);
    gsl_blas_dgemv(CblasNoTrans, 1, (gsl_matrix*)mat->data, (gsl_vector*)vec->data, 0, (gsl_vector*)vecOut->data);

    BitVectorMod2(vecOut);

    return vecOut;
}

// vec <- mat*vec
void BitMatrixMulVectorSet(struct BitMatrix* mat, struct BitVector* vec) {
    assert(mat->cols == vec->size);
    assert(mat->cols == mat->rows);
   
    gsl_blas_dgemv(CblasNoTrans, 1, (gsl_matrix*)mat->data, (gsl_vector*)vec->data, 0, (gsl_vector*)vec->data);
    BitVectorMod2(vec);
}


unsigned int BitMatrixRank(struct BitMatrix* mat) {
    struct BitMatrix* null = newBitMatrixIdentity(mat->cols);
    unsigned int rank = mat->cols; // rank of the null space

    struct BitMatrix* good = newBitMatrixZero(mat->cols, mat->cols);
    struct BitMatrix* bad = newBitMatrixZero(mat->cols, mat->cols);
    unsigned int numgood, numbad;

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
                BitVectorXorSet(badVec, flipVec);
                BitMatrixSetRow(null, badVec, rank);
            }
        }
    }
    
    BitMatrixFree(null);
    BitMatrixFree(good);
    BitMatrixFree(bad);

    rank = mat->cols - rank; // rank of null space to rank of mat
    return rank;
}
