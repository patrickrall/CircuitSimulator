#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main(void) {
    /*
    struct BitVector* vec = newBitVector(200);

    BitVectorSet(vec, 13, 1);
    BitVectorSet(vec, 12, 1);
    BitVectorSet(vec, 11, 1);
    BitVectorSet(vec, 7, 1);
    BitVectorSet(vec, 2, 2);

    struct BitVector* vec2 = newBitVector(200);
    BitVectorCopy(vec, vec2);

    BitVectorPrint(vec2);
    BitVectorPrint(vec);

    printf("Inner: %d\n", BitVectorInner(vec, vec2));
    BitVectorFree(vec);
    srand(10);

    struct BitMatrix* mat1 = newBitMatrixRandom(3, 3);
    BitMatrixPrint(mat1);

    printf("Rank: %d\n", BitMatrixRank(mat1));
    */

    Complex z = ComplexPolar(1, M_PI/3);
    z = ComplexMul(z, z);
    ComplexPrint(z);
    printf("%f\n", ComplexMag(z));

    return 0;
}
