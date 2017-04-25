
matrixtests:
	gcc -c matrix.c -o matrix.o -lm
	gcc -c matrix_tests.c -o matrix_tests.o
	gcc matrix.o matrix_tests.o -o test -lm
	-@rm *.o 2>/dev/null
	./test
	rm test
