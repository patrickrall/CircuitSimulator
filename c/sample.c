#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_linalg.h>
#include "stabilizer.h"


// Main: read inputs from stdin, and sample an inner product
// Arguments:
// <parallel:1 or 0>
// <NSamples> 
// <Norm>
// p
// zzz
// xxx
// p
// zzz
// xxx
// p
// zzz
// xxx
// L L L
// L L L
// L L L
int main(){
    // parallel
    int parallel;
    scanf("%d\n", &parallel);

    // samples
    int samples;
    scanf("%d\n", &samples);

    // norm
    double norm;
    scanf("%lg\n", &norm);



    if (parallel == 1) printf("Parallel\n");
    else printf("Not parallel\n");

    printf("Samples: %d\n", samples);
    
    printf("Norm: %lg\n", norm);

    return 0;
}
