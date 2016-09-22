#
# file: hiddenshift.py
# Tests the probability routine using the hidden shift algorithm,
# as described in appendix F.
#

from probability import probability
from circuit.compile import compileRawCircuit, removeComments, parseReference
import numpy as np
from datetime import datetime

# ######## CONFIG ######## #
n = 40  # length of shift string
toff = 2  # number of toffoli's per O_f
randcliff = 200  # number of random Cliffords in between CCZ's

# probability algorithm precision parameters
samples = 100
k = 11

# output config
printCirc = False
plot = True

# random seed
np.random.seed(0)

halfn = int(np.ceil(n/2))
n = halfn*2
s = np.random.randint(0, 2, size=n)  # hidden shift string

# ####### CIRCUIT ######## #

# Circuit takes form H^n O_f' H^n O_f H^n
# where O_f |x> = f(x) |x> are oracle circuits
# for functions f, f': F^n_2 -> {-1, 1}
# For details see appendix F.

# Construct random O_g |x> = (-1)^g(x) |x>, on |x| = n/2
Og = []  # list of gates


# pick cliffords Z and CZ in random locs
def addCliffords():
    for i in range(randcliff):
        line = ["_"]*halfn
        if np.random.randint(0, 2) == 1:
            line[np.random.randint(0, n/2)] = "Z"
        else:
            loc1 = np.random.randint(0, n/2)
            loc2 = np.random.randint(0, n/2)
            while loc2 == loc1: loc2 = np.random.randint(0, n/2)

            line[loc1] = "C"
            line[loc2] = "Z"
        Og.append("".join(line))

addCliffords()
for i in range(toff):  # pick toffolis
    line = ["_"]*halfn

    loc1 = np.random.randint(0, n/2)
    loc2 = np.random.randint(0, n/2)
    while loc2 == loc1: loc2 = np.random.randint(0, n/2)
    loc3 = np.random.randint(0, n/2)
    while loc3 == loc1 or loc3 == loc2: loc3 = np.random.randint(0, n/2)

    line[loc1] = "C"
    line[loc2] = "C"
    line[loc3] = "Z"

    Og.append("".join(line))

    addCliffords()

# build circuit
circuit = ""

# Hadamard all qubits
for i in range(n):
    line = "_"*i + "H" + "_"*(n-1-i)
    circuit += line + "\n"

# Apply O_f' = X(s) (\prod^{n/2}_{i=1} CZ_{i, i+n/2}) (I \otimes O_g) X(s)
# X(s)
for i in range(n):
    if s[i] == 1:
        line = "_"*i + "X" + "_"*(n-1-i)
        circuit += line + "\n"
# (I \otimes O_g)
for gate in Og:
    line = halfn*"_" + gate
    circuit += line + "\n"
# (\prod^{n/2}_{i=1} CZ_{i, i+n/2})
for i in range(halfn):
    line = i*"_" + "C" + "_"*(halfn - 1) + "Z" + "_"*(halfn-i-1)
    circuit += line + "\n"
# X(s)
for i in range(n):
    if s[i] == 1:
        line = "_"*i + "X" + "_"*(n-1-i)
        circuit += line + "\n"

# Hadamard all qubits
for i in range(n):
    line = "_"*i + "H" + "_"*(n-1-i)
    circuit += line + "\n"

# Apply O_f = (\prod^{n/2}_{i=1} CZ_{i, i+n/2}) (O_g \otimes I)
# (O_g \otimes I)
for gate in Og:
    line = gate + halfn*"_"
    circuit += line + "\n"
# (\prod^{n/2}_{i=1} CZ_{i, i+n/2})
for i in range(halfn):
    line = i*"_" + "C" + "_"*(halfn - 1) + "Z" + "_"*(halfn-i-1)
    circuit += line + "\n"

# Hadamard all qubits
for i in range(n):
    line = "_"*i + "H" + "_"*(n-1-i)
    circuit += line + "\n"

# compile
f = open("circuit/reference")
reference = parseReference(removeComments(f.read()))
f.close()
compiled = compileRawCircuit(circuit, reference)

# ####### INFORMATION ######## #
# Show circuit
if (printCirc):
    print("Circuit:")
    print(circuit)

print("Hidden shift string:")
print("".join(list(s.astype(str))))

print("%d hidden shift bits, %d qubits, %d T gates, %d samples, k=%d" %
      (n, len(compiled.splitlines()[0]), toff*8, samples, k))

# ####### RUN ALGORITHM ######## #

config = {
    "verbose": False,
    "parallel": True,
    "samples": samples,
    "fidbound": 1e-5,
    "k": k,
    "exact": False,
    "rank": False,
    "fidelity": True,
    "y": None,
    "x": None,
}


starttime = datetime.now()

results = []
for i in range(n):
    measure = {i: 1}
    result = probability(compiled, measure, config)

    print("Bit %d: %f" % (i, result))
    results.append(result)

print("Time elapsed: ", str(datetime.now() - starttime))

if plot:
    import matplotlib.pyplot as plt
    plt.bar(range(n), results, 0.9, align="center")
    plt.title("Hidden shift algorithm output")
    plt.xticks(range(n), s)
    plt.ylabel("Probability")
    plt.xlabel("Hidden shift bit")
    plt.show()
