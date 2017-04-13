#include <stdio.h>
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
    */

    struct BitMatrix* mat = newBitMatrixIdentity(10);
    BitMatrixSet(mat, 4,2,1);

    BitMatrixPrint(mat);

    BitMatrixFree(mat);
    return 0;
}
