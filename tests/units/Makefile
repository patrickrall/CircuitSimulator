C=gcc
CFLAGS=-lm -std=gnu99 -g -W -Wall
LFLAGS=-I/usr/include -L/usr/lib 

stabtests:
	$(C) $(LFLAGS) -c ../../libcirc/utils/matrix.c -o matrix.o $(CFLAGS)
	$(C) $(LFLAGS) -c ../../libcirc/stabilizer/stabilizer.c -o stabilizer.o $(CFLAGS)
	$(C) $(LFLAGS) -c stabtests.c -o stabtests.o $(CFLAGS)
	$(C) $(LFLAGS) stabtests.o stabilizer.o matrix.o -o stabtests $(CFLAGS)
	-@rm *.o 2>/dev/null || true
	./stabtests
	rm stabtests

BFLAGS=-lgsl -lgslcblas 

stabtestsblas:
	$(C) $(LFLAGS) -c ../../libcirc/utils/matrix-blas.c -o matrix.o $(CFLAGS) $(BFLAGS)
	$(C) $(LFLAGS) -c ../../libcirc/stabilizer/stabilizer.c -o stabilizer.o $(CFLAGS)
	$(C) $(LFLAGS) -c stabtests.c -o stabtests.o $(CFLAGS)
	$(C) $(LFLAGS) stabtests.o stabilizer.o matrix.o -o stabtests $(CFLAGS) $(BFLAGS)
	-@rm *.o 2>/dev/null || true
	./stabtests
	rm stabtests
