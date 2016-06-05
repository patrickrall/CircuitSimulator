#
# file: main.py
# Interface for the python implementation. Also implements sampling
# algorithm.
#

import numpy as np
import sys
import re
from datetime import datetime
import circuit.compile
import circuit.gadgetize
import projectors
from sample import decompose, sampleProjector
from multiprocessing import Pool


# calculate probability circuit yielding a measurement
def probability(circ, measure, Nsamples=5000, Lbound=0.0001, verbose=False, parallel=True, exact=False):
    # get projectors
    G, H, n, t = projectors.projectors(circ, measure, verbose=verbose, zeros=False)

    # truncate projectors
    Gprime, u = projectors.truncate(n, G)
    Hprime, v = projectors.truncate(n, H)

    # print projector data
    if verbose:
        print("Gadgetize circuit:")
        print("G:")
        printProjector(G)
        print("H:")
        printProjector(H)

        print("Truncate projectors to magic state space:")
        print("Gprime: (truncated by %d)" % u)
        printProjector(Gprime)
        print("Hprime: (trunacted by %d)" % v)
        printProjector(Hprime)

    # calculate |L> ~= |H>
    L, Lnorm = decompose(t, Lbound, exact=True)

    if verbose and L is None:
        print("Using exact decomposition of |H^t>: 2^%d" % int(np.ceil(t/2)))
    elif verbose:
        print("Stabilizer rank of |L>: 2^%d" % len(L))

    # parallelization over samples, or over elements of |L>?
    # Can't do both because of yucky "child thread of child thread" handling
    if L is None:
        Lparallel = 2**(np.ceil(t/2)) > Nsamples
    else:
        Lparallel = 2**len(L) > Nsamples
    if not parallel: Lparallel = False

    # helper for preventing needless sampling trivial projectors or circuits
    def calcProjector(P, Lparallel, pool=False):
        if len(P[0]) == 0 or len(P[1][0]) == 0:
            if verbose: print("Empty projector found. Not sampling.")
            # empty projector or Clifford circuit. No sampling needed.
            return Nsamples * sampleProjector((P, L, Lnorm, 0, False))

        queries = [(P, L, Lnorm, seed, Lparallel) for seed in range(0, Nsamples)]
        if pool is not False: return sum(pool.map(sampleProjector, queries))
        return sum(map(sampleProjector, queries))

    # calculate || Gprime |L> ||^2 and || Hprime |L> ||^2 up to a constant
    if parallel and not Lparallel:
        if verbose: print("Parallelizing over %d samples" % Nsamples)

        # set up thread pool
        pool = Pool()

        numerator = calcProjector(Gprime, False, pool=pool)
        denominator = calcProjector(Hprime, False, pool=pool)

        pool.close()
        pool.join()

    else:
        if verbose and Lparallel and L is None: print("Parallelizing over %d stabilizers in |H^t>" % 2**np.ceil(t/2))
        elif verbose and Lparallel: print("Parallelizing over %d stabilizers in |L>" % 2**len(L))

        numerator = calcProjector(Gprime, Lparallel)
        print("H")
        denominator = calcProjector(Hprime, Lparallel)

    if verbose:
        print("|| Gprime |H^t> ||^2 ~= ", numerator/Nsamples)
        print("|| Hprime |H^t> ||^2 ~= ", denominator/Nsamples)

    return 2**(v-u) * (numerator/denominator)


# sample from a set of qubits
def sampleQubits(circ, measure, sample, Nsamples=5000, Lbound=0.0001, verbose=False, parallel=True):

    def prob(measure):  # shortcut
        return probability(circ, measure, Nsamples=Nsamples, Lbound=Lbound, parallel=parallel, verbose=verbose)

    # recursive implementation: need the probability of sampling a smaller
    # set of qubits in order to sample next one
    def recursiveSample(qubits, measure):
        if len(qubits) == 1:  # base case
            # is a sample even possible? Some qubit constraints are unsatisfiable.
            if len(measure.keys()) != 0:  # other measurements present
                P = prob(measure)
                if verbose: print("Constraints present. Probability of satifying: %f\n" % P)
                if P == 0: return "", {}, 0  # not satisfiable
                if P < 1e-5:
                    print("Warning: probability of satisfying qubit constraints very low (%f). Could be impossible." % P)

            # compute probability
            measure[qubits[0]] = 0
            P0 = prob(measure)

            if verbose: print("Measuring qubit %d: P(0) = %f" % (qubits[0], P0))

            # sample
            qubit = 0 if np.random.random() < P0 else 1
            if verbose: print("-> Sampled", qubit, "\n")

            measure[qubits[0]] = qubit

            # return sample and probability of sample occurring
            return str(qubit), measure, (P0 if qubit == 0 else (1 - P0))
        else:
            qubitssofar, measure, Psofar = recursiveSample(qubits[:-1], measure)

            if qubitssofar == "": return "", {}, 0  # not satisfiable

            # compute conditional probability
            measure[qubits[-1]] = 0
            P = prob(measure)
            P0 = P/Psofar
            if verbose:
                print("Probability with qubit %d: %f" % (qubits[-1], P))
                print("Conditional probability of qubit %d: P(0) = %f" % (qubits[-1], P0))

            # sample
            qubit = 0 if np.random.random() < P0 else 1
            if verbose:
                print("-> Sampled", qubit)
                print("-> Sample so far:", qubitssofar + str(qubit), "\n")

            measure[qubits[-1]] = qubit

            return qubitssofar + str(qubit), measure, (P0 if qubit == 0 else (1 - P0))

    output, _, _ = recursiveSample(sample, measure)
    return output


def main(argv):
    if len(argv) < 2:
        return usage("Wrong number of arguments.")

    if argv[1] == "-h":
        return help()

    if len(argv) < 3 or len(argv) > 8:
        return usage("Wrong number of arguments.")

    config = {
        "verbose": False,
        "reference": "circuit/reference",
        "samples": int(1e3),
        "Lbound": 1e-5,
        "parallel": True,
    }

    # parse optional arguments
    for i in range(3, len(argv)):

        if argv[i] == "-v": config["verbose"] = True
        elif argv[i] == "-np": config["parallel"] = False
        elif argv[i][:10] == "reference=": config["reference"] = argv[i][10:]
        elif argv[i][:8] == "samples=": config["samples"] = int(float(argv[i][8:]))
        elif argv[i][:7] == "Lbound=": config["Lbound"] = float(argv[i][7:])
        else: raise ValueError("Invalid argument: " + argv[i])

    if not re.match("^(1|0|M|_)*$", argv[2]):
        return usage("Measurement string must consists only of '1', '0', 'M' or '_'.")

    # load input circuit
    f = open(argv[1], "r")
    infile = circuit.compile.removeComments(f.read())
    f.close()

    n = len(circuit.compile.standardGate(infile.splitlines()[0], lineMode=True))

    # measurement string correct length
    if (len(argv[2]) != n):
        return usage("Measurement not the right length:\n" +
                     "Expected %d, but input is %d long." % (n, len(argv[2])))

    # measurement string can't be all "_"'s
    if argv[2] == "_"*n: return usage("Must measure at least one bit.")

    # load reference
    f = open(config["reference"])
    reference = circuit.compile.parseReference(circuit.compile.removeComments(f.read()))
    f.close()

    # compile circuit
    circ = circuit.compile.compileRawCircuit(infile, reference)

    # get measurements
    measure = {}
    sample = []
    for i in range(len(argv[2])):
        if list(argv[2])[i] == '0':
            measure[i] = 0
        if list(argv[2])[i] == '1':
            measure[i] = 1
        if list(argv[2])[i] == 'M':
            sample.append(i)

    # start timer
    if config["verbose"]: starttime = datetime.now()

    if len(sample) == 0:  # algorithm 1: compute probability
        if config["verbose"]: print("Probability mode: calculate probability of measurement outcome")

        P = probability(circ, measure, Nsamples=config["samples"], Lbound=config["Lbound"],
                        parallel=config["parallel"], verbose=config["verbose"], exact=True)
        print("Probability:", P)

    else:  # algorithm 2: sample bits
        if config["verbose"]: print("Sample mode: sample marked bits")

        X = sampleQubits(circ, measure, sample, Nsamples=config["samples"], Lbound=config["Lbound"],
                         parallel=config["parallel"], verbose=config["verbose"])
        if len(X) == 0:  # no sample possible
            print("Error: circuit output state cannot produce a measurement with these constraints.")
        else:
            print("Sample:", X)

    if config["verbose"]:
        print("Time elapsed: ", str(datetime.now() - starttime))


def printProjector(projector):
    phases, xs, zs = projector
    for g in range(len(phases)):
        phase, x, z = phases[g], xs[g], zs[g]
        tmpphase = phase
        genstring = ""
        for i in range(len(x)):
            if (x[i] == 1 and z[i] == 1):
                # tmpphase -= 1  # divide phase by i
                genstring += "(XZ)"
                continue
            if (x[i] == 1):
                genstring += "X"
                continue
            if (z[i] == 1):
                genstring += "Z"
                continue
            genstring += "_"
        tmpphase = tmpphase % 4
        lookup = {0: " +", 1: " i", 2: " -", 3: "-i"}
        print(lookup[tmpphase] + genstring)
    if len(phases) == 0:
        print("Projector is empty.")


def usage(error):
    print(error + """\n\nStabilizer simulator usage:
main.py <circuitfile> <measurement> [reference=circuits/reference]

Full help statement: main.py -h""")


def help():
    print("""Stabilizer simulator usage:
main.py <circuitfile> <measurement> [reference=circuit/reference] [samples=1e3] [Lbound=1e-5] [-v] [-np]

Example:
python main.py examples/controlledH.circ _M samples=2e6
Execute controlled H gate on state |00>, then measure 2nd qubit.
Expected result: 0 always.

Note: Python 3 is recommended, but not strictly necessary. Some
print statements look less pretty with python 2.

Arguments:
<circuitfile>: A file specifying the circuit you want to simulate.
Each line corresponds to a gate in the circuit. For example, the
line CCX would denote the toffoli gate controlled on qubits 1 and 2
and acting on qubit 3.

Each line length must be equal to the number of qubits. Omitted qubits
on each line can be denoted with "_", so "__H" would run the Hadamard
gate on the third qubit.

The standard gate set is H (Hadamard), S (Phase), X (Pauli-X),
CX (CNOT) and T. More sophisticated gates can be be used in the input
file provided the reference file defines them.

<measurement>
A string consisting of the characters "_", "M", "0" and "1".
Must have length equal to the number of qubits in the input file.

If the string consists solely of "_", "0" and "1", the application
outputs the probability of a measurement result such that the qubits
have the values specified in the string.
E.g. "_01" would give the probability of measuring 001 or 101.

If the string contains the character "M", the application samples
a measurement on the qubits marked with "M". The output is a
string with length of the number of "M"'s in the input.

E.g. the circuit controlledH.circ creates the two-qubit state |1>|+>.
The inputs "_M" or "1M" would have 50-50 chance of yielding 0 or 1.
The input "0M" would return an error message, as the first qubit
is in the state |1>.


The following arguments can be specified in any order, or ommitted.
The identity of variable is necessary, i.e. write "samples=1e3", not "1e3".

[reference=circuits/reference]
Location of a reference file defining new gates in terms of the
standard gate set. Please read the documentation in the provided
reference file at circuit/reference for details.

[samples=1e5]
Number of samples in calculating || P |H^t> ||^2 for projectors P.
Larger numbers are more accurate. If samples = 1/(p * e^2) then
there is a 1-p probability of the result being within +-e of
the correct value. Must be an integer.

[Lbound=1e-3]
Rather than using a magic state |H^t> we use a state |L> which
can be written as a linear combination of fewer stabilizer states.
The inner product <H^t|L> will be greater than (1 - Lbound).
This variable only becomes relevant for about 10 T gates or more.

[-v]
Verbose mode. Prints lots of intermediate calculation information.
Useful for debugging, or understanding the application's inner workings.

[-np]
Non-parallel mode. Parallelization may cause problems with random
number generation on some machines. Since the algorithm is highly
concurrent, non-parallel mode is very slow.
""")

if __name__ == "__main__":
    main(sys.argv)
