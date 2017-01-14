C=mpicc
CFLAGS=-lgsl -lgslcblas -lm -std=gnu99 -g -lpthread -W -Wall
LFLAGS=-I/usr/include -L/usr/lib 

test: 
	$(C) $(LFLAGS) -c test.c -o test.o $(CFLAGS)
	$(C) $(LFLAGS) -c ../../libcirc/stabilizer/stabilizer.c -o stabilizer.o $(CFLAGS)
	$(C) $(LFLAGS) test.o stabilizer.o -o testmpi $(CFLAGS)
