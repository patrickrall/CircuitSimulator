#
# file: main.py
# Command line interface.
#

import sys
import re
from datetime import datetime
import libcirc.compilecirc as compilecirc
from libcirc.probability import probability, sampleQubits


def main(argv):
    if len(argv) > 1 and argv[1] == "-h":
        return help()

    if len(argv) < 3:
        return usage("Wrong number of arguments.")

    config = {
        "verbose": False,
        "silenceprojectors": False,
        "parallel": True,
        "samples": int(1e3),
        "fidbound": 1e-5,
        "k": None,
        "exact": False,
        "rank": False,
        "fidelity": False,
        "y": None,
        "x": None,
        "python": False,
        "cpath": "libcirc/sample",
    }

    # parse optional arguments
    for i in range(3, len(argv)):

        if argv[i] == "-v": config["verbose"] = True
        elif argv[i] == "-np": config["parallel"] = False
        elif argv[i][:8] == "samples=": config["samples"] = int(float(argv[i][8:]))
        elif argv[i][:9] == "fidbound=": config["fidbound"] = float(argv[i][9:])
        elif argv[i][:2] == "k=": config["k"] = int(float(argv[i][2:]))
        elif argv[i][:2] == "y=": config["y"] = argv[i][2:]
        elif argv[i][:2] == "x=": config["x"] = argv[i][2:]
        elif argv[i] == "-exact": config["exact"] = True
        elif argv[i] == "-rank": config["rank"] = True
        elif argv[i] == "-fidelity": config["fidelity"] = True
        elif argv[i] == "-py": config["python"] = True
        elif argv[i] == "-sp": config["silenceprojectors"] = True
        elif argv[i][:6] == "cpath=": config["cpath"] = argv[i][6:]
        else: raise ValueError("Invalid argument: " + argv[i])

    if not re.match("^(1|0|M|_)*$", argv[2]):
        return usage("Measurement string must consists only of '1', '0', 'M' or '_'.")

    # load input circuit
    infile = argv[1]
    circ = compilecirc.compileCircuit(fname=infile)

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
        config["exact"] = True

        P = probability(circ, measure, config)
        print("Probability:", P)

    else:  # algorithm 2: sample bits
        if config["verbose"]: print("Sample mode: sample marked bits")

        X = sampleQubits(circ, measure, sample, config)
        if len(X) == 0:  # no sample possible
            print("Error: circuit output state cannot produce a measurement with these constraints.")
        else:
            print("Sample:", X)

    if config["verbose"]:
        print("Time elapsed: ", str(datetime.now() - starttime))


def usage(error):
    print(error + """\n\nStabilizer simulator usage:
main.py <circuitfile> <measurement>

Full help statement: main.py -h""")


def help():
    print("""Stabilizer simulator usage:
main.py <circuitfile> <measurement> [samples=1e3] [-v] [-np] [-py] [-sp] [-exact]
        [-rank] [k=?] [fidbound=1e-5] [-fidelity]

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

[-v]
Verbose mode. Prints lots of intermediate calculation information.
Useful for debugging, or understanding the application's inner workings.

[-np]
Non-parallel mode. Parallelization may cause problems with random
number generation on some machines. Since the algorithm is highly
concurrent, non-parallel mode is very slow.

[-py]
Use python backend. The python backend is slower than the c backend,
but the code may be easier to read.

[-sp]
Silence projector printing despite verbosity. Useful for big circuits.

[cpath="/path/to/sample"]
Path to the compiled "sample" executable.

--- L selection parameters ---
The algorithm must construct a magic state |H^t>, where t is the number
of T-gates in the circuit. This state is resolved as a linear combination
of several stabilizer states. An exact decomposition is achieved by
decomposing |H^2> into a sum two stabilizer states, resulting in a
decomposition of |H^t> into about 2^(t/2) stabilizer states.
An exact decomposition of this form is necessary for calculating
probability queries like "_01".

For sampling queries like "_0M" we can get away with a more efficient
decomposition of |H^t> into 2^(0.23t) stabilizer states. Rather than
constructing |H^t> we construct a state |L> that is very similar to
|H^t>. |L> is defined by a k*t matrix L, where k is an input parameter.
L is randomly chosen.

[-exact]
Pass this option to always use an exact decomposition with scaling 2^(t/2).
This is only makes a difference with sampling queries like "_0M", where
an |L> decomposition would be used otherwise.

[-rank]
Force rank verification.
The k*t matrix L should have rank k. Verifiying this for large k and t
can take a long time, so this is not done by default. A random k*t
matrix usually has rank k.

[k=?]
The number of rows in the k*t matrix L. If k > t then k is set to t.
This parameter overrides the fidbound parameter. It is off by default,
so fidbound=1e-3 is used if neither option is specified.

[fidbound=1e-3]
k can also be chosen according to:
    k = ceil(1 - 2*t*log2(cos(pi/8)) - log2(fidbound))
This usually ensures that the inner product <H^t|L> is greater than
(1 - fidbound). By default, this parameter only affects the choice of
k, and the inner product <H^t|L> is not computed, so as to verify
that it is greater than (1 - fidbound). If k > t/2 then an exact
decomposition is used instead.

[-fidelity]
Inner product <H^t|L> display (and verification).
Pass this option and <H^t|L> is computed in time 2^(0.23t).
If the k option is specified, then <H^t|L> is simply printed.
If the fidbound option is specified, L is sampled until <H^t|L> > 1-fidbound.


--- Debugging ---
[y=?], [x=?]
Set the postselection of the T gates and other measurements, e.g., y=0010.
Output should be independent of the postselection, at least
up to some small error. If different y give different results
then there is a problem somewhere.
""")

if __name__ == "__main__":
    main(sys.argv)
