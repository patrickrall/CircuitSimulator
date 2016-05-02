C=gcc
CFLAGS=-lgsl -lgslcblas -lm -std=gnu99
LFLAGS=-I/usr/include -L/usr/lib 

stabtests: 
	$(C) $(LFLAGS) -c stabtests.c -o stabtests.o $(CFLAGS)
	$(C) $(LFLAGS) -c stabilizer.c -o stabilizer.o $(CFLAGS)
	$(C) $(LFLAGS) stabtests.o stabilizer.o -o stabtests $(CFLAGS)
	./stabtests
	rm *.o stabtests

clean:
	rm *.o stabtests
