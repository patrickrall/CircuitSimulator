C=mpicc
CFLAGS=-lm -std=gnu99 -g -W -Wall
LFLAGS=-I/usr/include -L/usr/lib 

mpibackend:
	$(C) $(LFLAGS) -c libcirc/utils/matrix.c -o matrix.o $(CFLAGS)
	$(C) $(LFLAGS) -c libcirc/utils/comms.c -o comms.o $(CFLAGS)
	$(C) $(LFLAGS) -c libcirc/stabilizer/stabilizer.c -o stabilizer.o $(CFLAGS)
	$(C) $(LFLAGS) -c libcirc/stateprep.c -o stateprep.o $(CFLAGS)
	$(C) $(LFLAGS) -c libcirc/innerprod.c -o innerprod.o $(CFLAGS)
	$(C) $(LFLAGS) -c libcirc/probability.c -o probability.o $(CFLAGS)
	$(C) $(LFLAGS) matrix.o comms.o stabilizer.o stateprep.o innerprod.o probability.o -o libcirc/mpibackend $(CFLAGS)
	-@rm *.o 2>/dev/null

BFLAGS=-lgsl -lgslcblas 

mpibackendblas:
	$(C) $(LFLAGS) -c libcirc/utils/matrix-blas.c -o matrix.o $(CFLAGS) $(BFLAGS)
	$(C) $(LFLAGS) -c libcirc/utils/comms.c -o comms.o $(CFLAGS)
	$(C) $(LFLAGS) -c libcirc/stabilizer/stabilizer.c -o stabilizer.o $(CFLAGS)
	$(C) $(LFLAGS) -c libcirc/stateprep.c -o stateprep.o $(CFLAGS)
	$(C) $(LFLAGS) -c libcirc/innerprod.c -o innerprod.o $(CFLAGS)
	$(C) $(LFLAGS) -D BLAS -c libcirc/probability.c -o probability.o $(CFLAGS)
	$(C) $(LFLAGS) matrix.o comms.o stabilizer.o stateprep.o innerprod.o probability.o -o libcirc/mpibackend $(CFLAGS) $(BFLAGS)
