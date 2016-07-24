#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_linalg.h>
#include "stabilizer.h"

struct Projector {
    int Nstabs;
    int Nqubits;
    gsl_vector* phases;
    gsl_matrix* xs;
    gsl_matrix* zs; 
};

void printProjector(struct Projector *P);
struct Projector* readProjector(void);

int main(int argc, char* argv[]){
    struct Projector *G = readProjector();
    printf("Gprime:\n");
    printProjector(G);
    
    struct Projector *H = readProjector();
    printf("Hprime:\n");
    printProjector(H);
    return 0;
}

struct Projector* readProjector(void) {
    struct Projector *P = (struct Projector *)malloc(sizeof(struct Projector));
    scanf("%d", &(P->Nstabs));
    scanf("%d", &(P->Nqubits));
    P->phases = gsl_vector_alloc(P->Nstabs);
    P->xs = gsl_matrix_alloc(P->Nstabs, P->Nqubits);
    P->zs = gsl_matrix_alloc(P->Nstabs, P->Nqubits);
    
    int v;
    for (int i = 0; i < P->Nstabs; i++) {
        scanf("%d", &v);
        gsl_vector_set(P->phases, i, (double)v);

        for (int j = 0; j < P->Nqubits; j++) {
            scanf("%d", &v);
            gsl_matrix_set(P->xs, i, j, (double)v);
            
            scanf("%d", &v);
            gsl_matrix_set(P->zs, i, j, (double)v);
        }
    }

    return P;
}

void printProjector(struct Projector *P) {
    for (int i = 0; i < P->Nstabs; i++) {
        int tmpphase = (int)gsl_vector_get(P->phases, i);

        gsl_vector *xs = gsl_vector_alloc(P->Nqubits);
        gsl_vector *zs = gsl_vector_alloc(P->Nqubits);
        gsl_matrix_get_row(xs, P->xs, i);
        gsl_matrix_get_row(zs, P->zs, i);

        char stab[20];
        stab[0] ='\0';

        for (int j = 0; j < P->Nqubits; j++) {
            double x = gsl_vector_get(xs, j);
            double z = gsl_vector_get(zs, j);

            if (x == 1 && z == 1) {
                tmpphase -= 1;
                strcat(stab, "Y");
                continue;
            }

            if (x == 1) {
                strcat(stab, "X");
                continue;
            }

            if (z == 1) {
                strcat(stab, "Z");
                continue;
            }
            strcat(stab, "_");
        }
        while (tmpphase < 0) tmpphase += 4;

        char* lookup[4] = {" +", " i", " -", "-i"};
        printf("%s%s\n", lookup[tmpphase], stab);
    }
}
