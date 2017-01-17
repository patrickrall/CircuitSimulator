C=mpicc
CFLAGS=-lgsl -lgslcblas -lm -std=gnu99 -g -lpthread -W -Wall
LFLAGS=-I/usr/include -L/usr/lib 

sample:
	$(C) $(LFLAGS) -c libcirc/sample.c -o sample.o $(CFLAGS)
	$(C) $(LFLAGS) -c libcirc/stabilizer/stabilizer.c -o stabilizer.o $(CFLAGS)
	$(C) $(LFLAGS) sample.o stabilizer.o -o libcirc/sample $(CFLAGS)
	-@rm *.o 2>/dev/null
