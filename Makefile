C=gcc
CFLAGS=-lgsl -lgslcblas -lm -std=gnu99 -g -lpthread -W -Wall
LFLAGS=-I/usr/include -L/usr/lib 

sample: clean
	$(C) $(LFLAGS) -c libcirc/sample.c -o sample.o $(CFLAGS)
	$(C) $(LFLAGS) -c libcirc/stabilizer/stabilizer.c -o stabilizer.o $(CFLAGS)
	$(C) $(LFLAGS) sample.o stabilizer.o -o libcirc/sample $(CFLAGS)
	-@rm *.o 2>/dev/null

stabtests: 
	$(C) $(LFLAGS) -c tests/stabtests.c -o stabtests.o $(CFLAGS)
	$(C) $(LFLAGS) -c libcirc/stabilizer/stabilizer.c -o stabilizer.o $(CFLAGS)
	$(C) $(LFLAGS) stabtests.o stabilizer.o -o stabtests $(CFLAGS)
	-@rm *.o 2>/dev/null || true
	./stabtests
	rm stabtests

clean:
	-@rm *.o stabtests libcirc/sample 2>/dev/null || true
