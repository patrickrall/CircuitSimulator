#include "../stabilizer/stabilizer.h"

// Basics
struct Projector {
    int Nstabs;
    int Nqubits;
    gsl_vector* phases;
    gsl_matrix* xs;
    gsl_matrix* zs; 
};
struct Projector* readProjector(FILE* stream);
void printProjector(struct Projector *P);

char* binrep(unsigned int val, char *buff, int sz);

// Communication
void send_int(int i, int dest);
int recv_int(int src);

void send_double(double i, int dest);
double recv_double(int src);

void send_gsl_vector(gsl_vector* vec, int dest);
gsl_vector* recv_gsl_vector(int src);

void send_gsl_matrix(gsl_matrix* mat, int dest);
gsl_matrix* recv_gsl_matrix(int src);

void send_gsl_complex(gsl_complex z, int dest);
gsl_complex recv_gsl_complex(int src);

void send_projector(struct Projector* P, int dest);
struct Projector* recv_projector(int src);

void send_stabilizer_state(struct StabilizerState* state, int dest);
struct StabilizerState *recv_stabilizer_state(int src);


