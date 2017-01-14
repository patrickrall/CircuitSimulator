C=gcc
CFLAGS=-lgsl -lgslcblas -lm -std=gnu99 -g -lpthread -W -Wall
LFLAGS=-I/usr/include -L/usr/lib 

stabtests:
	$(C) $(LFLAGS) -c stabtests.c -o stabtests.o $(CFLAGS)
	$(C) $(LFLAGS) -c ../../libcirc/stabilizer/stabilizer.c -o stabilizer.o $(CFLAGS)
	$(C) $(LFLAGS) stabtests.o stabilizer.o -o stabtests $(CFLAGS)
	-@rm *.o 2>/dev/null || true
	./stabtests
	rm stabtests
