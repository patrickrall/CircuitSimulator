#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "comms.h"
#include "mpi.h"


/************* Helpers for reading and printing projectors *************/
struct Projector* readProjector(FILE* stream) {
    struct Projector *P = (struct Projector *)malloc(sizeof(struct Projector));
    fscanf(stream,"%d", &(P->Nstabs));
    fscanf(stream,"%d", &(P->Nqubits));

    if (P->Nstabs == 0) return P;

    P->phases = gsl_vector_alloc(P->Nstabs);
    P->xs = gsl_matrix_alloc(P->Nstabs, P->Nqubits);
    P->zs = gsl_matrix_alloc(P->Nstabs, P->Nqubits);
    
    int v;
    for (int i = 0; i < P->Nstabs; i++) {
        fscanf(stream,"%d", &v);
        gsl_vector_set(P->phases, i, (double)v);

        for (int j = 0; j < P->Nqubits; j++) {
            fscanf(stream,"%d", &v);
            gsl_matrix_set(P->xs, i, j, (double)v);
            
            fscanf(stream,"%d", &v);
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

        gsl_vector_free(xs);
        gsl_vector_free(zs);
    }
}

// Comms
//----------- int macro
void send_int(int i, int dest) {
    MPI_Send(&i, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
}
int recv_int(int src) {
    int buff;
    MPI_Recv(&buff, 1, MPI_INT, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return buff;
}

//----------- double macro
void send_double(double i, int dest) {
    MPI_Send(&i, 1, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
}
double recv_double(int src) {
    double buff;
    MPI_Recv(&buff, 1, MPI_DOUBLE, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return buff;
}


//----------- gsl_vector
void send_gsl_vector(gsl_vector* vec, int dest) {
    int size = (int)vec->size;
    send_int(size, dest); 
    for (int i = 0; i < size; i++) {
        send_double(gsl_vector_get(vec, i), dest);
    }
}
gsl_vector* recv_gsl_vector(int src) {
    int size = recv_int(src);
    gsl_vector* vec = gsl_vector_alloc(size);
    for (int i = 0; i < size; i++) {
        gsl_vector_set(vec, i, recv_double(src));
    }
    return vec;
}

//----------- gsl_matrix
void send_gsl_matrix(gsl_matrix* mat, int dest){
    int size1 = (int)mat->size1;
    int size2 = (int)mat->size2;

    send_int(size1, dest); 
    send_int(size2, dest); 

    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            send_double(gsl_matrix_get(mat, i, j), dest);
        }
    }
} 
gsl_matrix* recv_gsl_matrix(int src) {
    int size1 = recv_int(src);
    int size2 = recv_int(src);

    gsl_matrix* mat = gsl_matrix_alloc(size1, size2);

    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            gsl_matrix_set(mat, i, j, recv_double(src));
        }
    }
    return mat;
}


//---------- gsl_complex
void send_gsl_complex(gsl_complex z, int dest) {
    send_double(GSL_REAL(z), dest);
    send_double(GSL_IMAG(z), dest);
}
gsl_complex recv_gsl_complex(int src) {
    double real = recv_double(src);
    double imag = recv_double(src);
    return gsl_complex_rect(real, imag);
}

//---------- projectors
void send_projector(struct Projector* P, int dest) {
    send_int(P->Nstabs, dest);
    send_int(P->Nqubits, dest);
    if (P->Nstabs == 0) return;

    send_gsl_vector(P->phases, dest);
    send_gsl_matrix(P->xs, dest);
    send_gsl_matrix(P->zs, dest);
}
struct Projector* recv_projector(int src) {
    struct Projector *P = (struct Projector *)malloc(sizeof(struct Projector));
    P->Nstabs = recv_int(src);
    P->Nqubits = recv_int(src);
    if (P->Nstabs == 0) return P;

    P->phases = recv_gsl_vector(src);
    P->xs = recv_gsl_matrix(src);
    P->zs = recv_gsl_matrix(src);
    return P;
}

//---------- stabilizer states
// implemented, but not needed anywhere
/*
void send_stabilizer_state(void* statein, int dest) {
    struct StabilizerState* state = (struct StabilizerState*)statein;
    send_int(state->n, dest);
    send_int(state->k, dest);

    send_gsl_vector(state->h, dest);
    send_gsl_matrix(state->G, dest);
    send_gsl_matrix(state->Gbar, dest);

    send_int(state->Q, dest);
    send_gsl_vector(state->D, dest);
    send_gsl_matrix(state->J, dest);
}
void* recv_stabilizer_state(int src) {
    int n = recv_int(src);
    int k = recv_int(src);

    struct StabilizerState *state = allocStabilizerState(n, k);
    state->h = recv_gsl_vector(src);
    state->G = recv_gsl_matrix(src);
    state->Gbar = recv_gsl_matrix(src);

    state->Q = recv_int(src);
    state->D = recv_gsl_vector(src);
    state->J = recv_gsl_matrix(src);
    return (void*)state;
}
*/
