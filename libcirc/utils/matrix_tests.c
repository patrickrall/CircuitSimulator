#include <stdio.h>
#include <stdlib.h>
#include "matrix.h"

int main(void) {
   /* 
    struct BitVector* vec = newBitVectorRandom(20);


    struct BitVector* vec2 = newBitVectorRandom(20);

    BitVectorPrint(vec2);
    BitVectorPrint(vec);

    printf("Inner: %d\n", BitVectorInner(vec, vec2));
    BitVectorFree(vec);
    srand(10);
    */
   
    
    struct BitMatrix* mat1 = newBitMatrixIdentity(2);
    struct BitMatrix* eye = newBitMatrixIdentity(2);
    BitMatrixSet(mat1, 1,0,1);

    //BitMatrixPrint(mat1);
    //BitMatrixPrint(eye);
    
    struct BitMatrix* out = BitMatrixMulMatrix(mat1, eye);
    //BitMatrixPrint(out);


   // printf("Rank: %d\n", BitMatrixRank(mat1));

    /*
    Complex z = ComplexPolar(1, M_PI/3);
    z = ComplexMul(z, z);
    ComplexPrint(z);
    printf("%f\n", ComplexMag(z));
    */

    return 0;
}
