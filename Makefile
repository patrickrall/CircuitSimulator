C=mpicc
CFLAGS=-lgsl -lgslcblas -lm -std=gnu99 -g -W -Wall
LFLAGS=-I/usr/include -L/usr/lib 

mpibackend:
	$(C) $(LFLAGS) -c libcirc/utils/comms.c -o comms.o $(CFLAGS)
	$(C) $(LFLAGS) -c libcirc/stabilizer/stabilizer.c -o stabilizer.o $(CFLAGS)
	$(C) $(LFLAGS) -c libcirc/stateprep.c -o stateprep.o $(CFLAGS)
	$(C) $(LFLAGS) -c libcirc/innerprod.c -o innerprod.o $(CFLAGS)
	$(C) $(LFLAGS) -c libcirc/probability.c -o probability.o $(CFLAGS)
	$(C) $(LFLAGS) comms.o stabilizer.o stateprep.o innerprod.o probability.o -o libcirc/mpibackend $(CFLAGS)
	-@rm *.o 2>/dev/null
