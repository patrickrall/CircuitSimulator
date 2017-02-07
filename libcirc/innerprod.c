#include <stdlib.h>
#include <math.h>

#include <gsl/gsl_sort_vector.h>

#include "stabilizer/stabilizer.h"
#include "mpi.h"
#include "utils/comms.h"


/***************** Prototypes *******************/
// linked from stateprep.c
struct StabilizerState* prepH(int i, int t);
struct StabilizerState* prepL(int i, int t, gsl_matrix* L);

double sampledProjector(struct Projector *P, gsl_matrix *L, int exact, double norm, int samples);
double singleProjectorSample(struct Projector *P, gsl_matrix *L, int exact);
gsl_complex exactProjectorWork(int i, struct Projector *P, gsl_matrix *L, int exact);

/******************* Code **********************/
// Master function. Calculate median of several means. 
double multiSampledProjector(struct Projector *P, gsl_matrix *L, int exact, double norm, int samples, int bins) {
    if (bins == 1) return sampledProjector(P, L, exact, norm, samples);
    
    gsl_vector * bin_vals = gsl_vector_alloc(bins);
    for (int i = 0; i < bins; i++) {
        double value = sampledProjector(P, L, exact, norm, samples);
        gsl_vector_set(bin_vals, i, value);
    }

    gsl_sort_vector(bin_vals);

    double out = 0;
    if (bins % 2 == 1) { 
        out = gsl_vector_get(bin_vals, (bins - 1)/2);
    } else {
        out = (gsl_vector_get(bin_vals, bins/2) + gsl_vector_get(bin_vals, bins/2 - 1))/2;
    }

    gsl_vector_free(bin_vals);
    return out;
}


// Master function. Calculate mean of several samples.
double sampledProjector(struct Projector *P, gsl_matrix *L, int exact, double norm, int samples) {
    // empty projector
    if (P->Nstabs == 0) return pow(norm, 2);

    int t = P->Nqubits;

    // clifford circuit
    if (t == 0) {
        double sum = 1; // include identity
        for (int i = 0; i < P->Nstabs; i++) {
            double ph = gsl_vector_get(P->phases, i);
            if (ph == 0) sum += 1;
            if (ph == 2) sum -= 1;
        } 

        return pow(norm, 2) * sum/(1 + (double)P->Nstabs);
    }

    // parallel code
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    double total = 0;
    for (int dest = 1; dest < world_size; dest++) {
        send_int(2, dest); // command
        send_projector(P, dest);
        send_int(samples, dest);
    }    

    for (int i = 0; i < samples; i += world_size) {
        total += singleProjectorSample(P, L, exact);
    }

    for (int src = 1; src < world_size; src++) {
        total += recv_double(src);
    }

    return total/samples;
}


// Evaluate 2^t * || <theta| P |H^t> ||^2 for a random theta.
double singleProjectorSample(struct Projector *P, gsl_matrix *L, int exact) {
    int t = P->Nqubits;

    int k;
    if (!exact) k = L->size1;

    // Sample random stabilizer state
	struct StabilizerState *theta = randomStabilizerState(t);

    // project state onto P
    double projfactor = 1;
    gsl_vector *zeta = gsl_vector_alloc(P->Nqubits);
    gsl_vector *xi = gsl_vector_alloc(P->Nqubits);

    for (int i = 0; i < P->Nstabs; i++) {
        int m = gsl_vector_get(P->phases, i);
        gsl_matrix_get_row(zeta, P->zs, i);
        gsl_matrix_get_row(xi, P->xs, i);

        double res = measurePauli(theta, m, zeta, xi);
        projfactor *= res;

        if (res == 0) {
            freeStabilizerState(theta);
            gsl_vector_free(zeta);
            gsl_vector_free(xi);
            return 0;
        }
    } 

    gsl_vector_free(zeta);
    gsl_vector_free(xi);

    struct StabilizerState *phi;
    gsl_complex total = gsl_complex_rect(0,0);
    gsl_complex innerProd;
    int eps, p, m;
    int size;

    if (exact) {
        size = pow(2, ceil((double)t / 2));
        for (int i = 0; i < size; i+=1) {
            phi = prepH(i, t);
            innerProduct(theta, phi, &eps, &p, &m, &innerProd, 0);
            total = gsl_complex_add(total, innerProd);
            freeStabilizerState(phi);
        }
    } else {
        size = pow(2, k);
        for (int i = 0; i < size; i+=1) {
            phi = prepL(i, t, L);
            innerProduct(theta, phi, &eps, &p, &m, &innerProd, 0);
            total = gsl_complex_add(total, innerProd);
            freeStabilizerState(phi);
        }
    }
    
    freeStabilizerState(theta);

    double out = pow(2, t) * gsl_complex_abs2(gsl_complex_mul_real(total, projfactor));
    return out;
}


// Master function. Calculate projector exactly. 
double exactProjector(struct Projector *P, gsl_matrix *L, int exact, double norm) {
    // empty projector
    if (P->Nstabs == 0) return pow(norm, 2);

    int t = P->Nqubits;
    int k;
    if (!exact) k = L->size1;

    // clifford circuit
    if (t == 0) {
        double sum = 1; // include identity
        for (int i = 0; i < P->Nstabs; i++) {
            double ph = gsl_vector_get(P->phases, i);
            if (ph == 0) sum += 1;
            if (ph == 2) sum -= 1;
        } 

        return sum/(1 + (double)P->Nstabs);
    }
    
    // parallel code
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int size;
    if (exact) size = pow(2, ceil((double)t / 2));
    else size = pow(2, k);

    gsl_complex total = gsl_complex_rect(0,0);
    gsl_complex part;
    for (int dest = 1; dest < world_size; dest++) {
        send_int(2, dest); // command
        send_projector(P, dest);
        send_int(size, dest);
    }    

    for (int i = 0; i < size; i += world_size) {
        part = exactProjectorWork(i, P, L, exact);
        total = gsl_complex_add(total, part);
    }

    for (int src = 1; src < world_size; src++) {
        part = recv_gsl_complex(src);
        total = gsl_complex_add(total, part);
    }

    return gsl_complex_abs(total);
}


// Work function for exactProjector.
gsl_complex exactProjectorWork(int i, struct Projector *P, gsl_matrix *L, int exact) {
    int t = P->Nqubits;

    int k;
    if (!exact) k = L->size1;

    struct StabilizerState* theta;
    if (exact) theta = prepH(i, t);
    else theta = prepL(i, t, L);

    // Project theta
    double projfactor = 1;
    gsl_vector *zeta = gsl_vector_alloc(P->Nqubits);
    gsl_vector *xi = gsl_vector_alloc(P->Nqubits);

    for (int j = 0; j < P->Nstabs; j++) {
        int m = gsl_vector_get(P->phases, j);
        gsl_matrix_get_row(zeta, P->zs, j);
        gsl_matrix_get_row(xi, P->xs, j);
        
        double res = measurePauli(theta, m, zeta, xi);
        projfactor *= res;

        if (res == 0) return gsl_complex_rect(0,0);
    } 

    gsl_vector_free(zeta);
    gsl_vector_free(xi);

    // Evaluate other components
    gsl_complex total = gsl_complex_rect(0,0);
    int eps, p, m;
    gsl_complex innerProd;
    struct StabilizerState* phi;

    int size;
    if (exact) size = pow(2, ceil((double)t / 2));
    else size = pow(2, k);

    for (int j = 0; j < size; j++) {
        if (exact) phi = prepH(j, t);
        else phi = prepL(j, t, L);
        
        innerProduct(theta, phi, &eps, &p, &m, &innerProd, 0);
        total = gsl_complex_add(total, gsl_complex_mul_real(innerProd, projfactor));  

        freeStabilizerState(phi);
    }

    freeStabilizerState(theta);
    return total;
}
