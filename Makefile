CC=gcc
CFLAGS=-I${TACC_GSL_INC} -I${TACC_GSL_INC}/gsl -std=gnu99 -g -W -Wall
LDFLAGS=-L${TACC_GSL_LIB} -lgsl -lgslcblas -lm -lpthread

sample: clean
	$(CC) $(CFLAGS) -c libcirc/sample.c -o sample.o
	$(CC) $(CFLAGS) -c libcirc/stabilizer/stabilizer.c -o stabilizer.o
	$(CC) $(CFLAGS) $(LDFLAGS) sample.o stabilizer.o -o libcirc/sample
	-@rm *.o 2>/dev/null

stabtests: 
	$(CC) $(LDFLAGS) -c tests/stabtests.c -o stabtests.o 
	$(CC) $(LDFLAGS) -c libcirc/stabilizer/stabilizer.c -o stabilizer.o
	$(CC) $(LDFLAGS) stabtests.o stabilizer.o -o stabtests
	-@rm *.o 2>/dev/null || true
	./stabtests
	rm stabtests

clean:
	-@rm *.o stabtests libcirc/sample 2>/dev/null || true
