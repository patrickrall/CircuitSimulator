CFLAGS=-lgsl -lgslcblas 
LFLAGS=-I/usr/include -L/usr/lib 

matrixtests:
	gcc $(LFLAGS) -c matrix.c -o matrix.o -lm $(CFLAGS)
	gcc -c matrix_tests.c -o matrix_tests.o
	gcc $(LFLAGS) matrix.o matrix_tests.o -o test -lm $(CFLAGS)
	-@rm *.o 2>/dev/null
	./test
	rm test
