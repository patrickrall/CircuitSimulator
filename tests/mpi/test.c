#include <mpi.h>
#include <stdio.h>
#include "../../libcirc/stabilizer/stabilizer.h"

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_errno.h>

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
void recv_gsl_matrix(gsl_matrix** mat, int src) {
    int size1 = recv_int(src);
    int size2 = recv_int(src);

    *mat = gsl_matrix_alloc(size1, size2);

    for (int i = 0; i < size1; i++) {
        for (int j = 0; j < size2; j++) {
            gsl_matrix_set(*mat, i, j, recv_double(src));
        }
    }
}

int main(int argc, char* argv[]){
    // Initialize the MPI environment
    MPI_Init(NULL, NULL);
    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int tasks = 10;
    double out = 0;
    
    gsl_matrix* mat = NULL;
    int buff;

    if (world_rank == 0) {
        for (int i = 0; i < argc; i++) {
            printf("Arg %d: %s\n", i, argv[i]);
        }
       
        int size1;
        int size2;
        printf("Enter array sizes: ");
        fflush(stdout);
        scanf("%d", &size1);
        scanf("%d", &size2);

        mat = gsl_matrix_alloc(size1, size2);

        printf("Enter %d numbers:\n", size1*size2);
        for (int i = 0; i < size1; i++) {
            for (int j = 0; j < size2; j++) {
                scanf("%d", &buff);
                gsl_matrix_set(mat, i, j, buff);
            }
        }

        for (int i = 1; i < world_size; i++) {
            send_gsl_matrix(mat, i); 
        }

    } else {
        recv_gsl_matrix(&mat, 0);
    }

    printf("Thread %d has [", world_rank);
    for (int i = 0; i < (int)mat->size1; i++) {
        printf("[");
        for (int j = 0; j < (int)mat->size2; j++) {
            printf("%f,", gsl_matrix_get(mat, i, j));
        }
        printf("] ");
    }
    printf("]\n");

    int taskidx = world_rank;
    while (taskidx < tasks) {
        printf("Thread %d performing task %d.\n", world_rank, taskidx);
        out += (double)(taskidx);

        taskidx += world_size;
    }

    if (world_rank != 0) {
        MPI_Send(&out, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    } else {
        double buffer;

        for (int i = 1; i < world_size; i++) {
            MPI_Recv(&buffer, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            out += buffer;
        }

        printf("Output: %f\n", out);
    }

    gsl_matrix_free(mat);

    MPI_Finalize();
    return 0;
}
