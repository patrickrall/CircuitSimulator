C=gcc
CFLAGS=-lgsl -lgslcblas -lm -std=gnu99
LFLAGS=-I/usr/include -L/usr/lib 

sample:
	$(C) $(LFLAGS) -c sample.c -o sample.o $(CFLAGS)
	$(C) $(LFLAGS) -c stabilizer.c -o stabilizer.o $(CFLAGS)
	$(C) $(LFLAGS) sample.o stabilizer.o -o sample $(CFLAGS)
	rm *.o
	./sample
	rm sample

stabtests: 
	$(C) $(LFLAGS) -c stabtests.c -o stabtests.o $(CFLAGS)
	$(C) $(LFLAGS) -c stabilizer.c -o stabilizer.o $(CFLAGS)
	$(C) $(LFLAGS) stabtests.o stabilizer.o -o stabtests $(CFLAGS)
	rm *.o
	./stabtests
	rm stabtests

clean:
	rm *.o stabtests
