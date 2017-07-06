mpibackend:
	$(C) $(LDFLAGS) -c libcirc/utils/comms.c -o comms.o $(CFLAGS)
	$(C) $(LDFLAGS) -c libcirc/stabilizer/stabilizer.c -o stabilizer.o $(CFLAGS)
	$(C) $(LDFLAGS) -c libcirc/stateprep.c -o stateprep.o $(CFLAGS)
	$(C) $(LDFLAGS) -c libcirc/innerprod.c -o innerprod.o $(CFLAGS)
	$(C) $(LDFLAGS) -c libcirc/probability.c -o probability.o $(CFLAGS)
	$(C) $(LDFLAGS) comms.o stabilizer.o stateprep.o innerprod.o probability.o -o libcirc/mpibackend $(CFLAGS)
	-@rm *.o 2>/dev/null

