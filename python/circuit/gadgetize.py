#
# file: gadgetize.py
# input a file with a compiled circuit (h,s,CNOT,t) only
# output stabilizer projectors g(x,y) and h(x)
#

from circuit.compile import standardGate, removeComments
import numpy as np


# count the number of random bits to generate
# this is the number of Ts plus measured ancillas
# assumes circuit consists only of gates in the gate set
def countY(circuit):
    Ts = 0
    Tcache = []  # recent T gates to check for any consecutive Ts
    Ms = []

    MTs = []
    # measurements depending on which a T is performed
    # depending on postselection we might not need those T
    # these are not counted in Ts variable

    lineNr = 0
    for line in circuit.splitlines():
        lineNr += 1
        gate = standardGate(line, lineMode=True)

        # identify measured qubits
        if 'M' in gate:
            Midx = gate.index('M')
            if Midx not in Ms:
                Ms.append(Midx)

        # Check for operations performed on already measured qubits
        for Midx in Ms:
            if gate[Midx] not in ['M', '_']:
                raise SyntaxError("Line %d: Cannot perform operations to measured qubit %d" % (lineNr, Midx))

        if 'T' in gate and 'M' not in gate:
            Ts += 1  # count T gate

            # Put qubit into Tcache
            Tidx = gate.index('T')
            if Tidx in Tcache:  # already there: consecutive Ts
                raise SyntaxError("Line %d: Consecutive Ts on qubit %d" % (lineNr, Midx))
            Tcache.append(Tidx)

        # remove qubits from Tcache if a gate is encountered
        for Tidx in Tcache:
            if gate[Tidx] != '_':
                Tcache.remove(Tidx)

        # T that is applied conditionally on a measurement
        if 'T' in gate and 'M' in gate:
            Midx = gate.index('M')
            MTs.append(Midx)

    return Ts, Ms, MTs


# contract circuit into single projector
# y: postselected T ancilla measurements
# Mdict: postselected other measurements
# Xdict: output string
def gadgetize(circuit, Xdict, Mdict, y):
    n = len(standardGate(circuit.split('\n', 1)[0], lineMode=True))
    size = n + len(y)

    phases = []  # list of numbers in Z_4
    xs = []  # lists of bitstrings
    zs = []

    def gen(idx, val, phases, xs, zs):
        xs.append(np.zeros(size).astype(int))
        z = np.zeros(size)
        z[idx] = 1
        zs.append(z)

        phase = 0
        if val == 1: phase = 2
        phases.append(phase)
        return phases, xs, zs

    # set output qubits
    for X in Xdict.keys(): phases, xs, zs = gen(X, Xdict[X], phases, xs, zs)

    # set measured ancillas
    for M in Mdict.keys(): phases, xs, zs = gen(M, Mdict[M], phases, xs, zs)

    # set T ancillas
    for i in range(n, size): phases, xs, zs = gen(i, y[i-n], phases, xs, zs)

    # return phases, xs, zs  # debug

    # conjugate stabilizer generators:
    tpos = n
    for line in reversed(circuit.splitlines()):
        gate = standardGate(line, lineMode=True)

        if line == "": continue

        #  skip if contains a measurement
        if 'M' in gate:
            Midx = gate.index('M')
            if Mdict[Midx] == 0: continue

        std = standardGate(line)

        idxs = []
        for letter in std:
            idxs.append(gate.index(letter))

        lookup = {
            "H": {(0, 0): (0, 0, 0), (1, 0): (0, 0, 1), (0, 1): (0, 1, 0), (1, 1): (2, 1, 1)},  # X->Z, Z->X
            "X": {(0, 0): (0, 0, 0), (1, 0): (0, 1, 0), (0, 1): (2, 0, 1), (1, 1): (2, 1, 1)},  # X->X, Z->-Z
            "S": {(0, 0): (0, 0, 0), (1, 0): (1, 1, 1), (0, 1): (0, 0, 1), (1, 1): (1, 1, 0)},  # X->Y, Z->Z
            "CX": {(0, 0, 0, 0): (0, 0, 0, 0, 0), (0, 0, 1, 0): (0, 0, 0, 1, 0), (0, 0, 0, 1): (0, 0, 1, 0, 1), (0, 0, 1, 1): (0, 0, 1, 1, 1),
                   (1, 0, 0, 0): (0, 1, 0, 1, 0), (1, 0, 1, 0): (0, 1, 0, 0, 0), (1, 0, 0, 1): (0, 1, 1, 1, 1), (1, 0, 1, 1): (0, 1, 1, 0, 1),
                   (0, 1, 0, 0): (0, 0, 1, 0, 0), (0, 1, 1, 0): (2, 1, 1, 1, 0), (0, 1, 0, 1): (0, 0, 0, 0, 1), (0, 1, 1, 1): (0, 0, 0, 1, 1),
                   (1, 1, 0, 0): (0, 1, 1, 1, 0), (1, 1, 1, 0): (0, 1, 1, 0, 1), (1, 1, 0, 1): (0, 1, 0, 1, 1), (1, 1, 1, 1): (0, 1, 0, 0, 1)}
            # "CX": {"IX": "IX",       "IZ": "ZZ",       "I(XZ)": "Z(XZ)",
            #       "XX": "XI",       "XZ": "(XZ)(XZ)", "X(XZ)": "(XZ)Z",
            #       "ZX": "-(XZ)X",   "ZZ": "IZ",       "Z(XZ)": "I(XZ)",
            #       "(XZ)X": "(XZ)Z", "(XZ)Z": "X(XZ)", "(XZ)(XZ)": "XZ",
            #    }
        }

        def conjugateLookup(std, idxs, phase, xs, zs):
            mapping = lookup[std]
            if (len(std) == 1):
                state = (xs[idxs[0]], zs[idxs[0]])
                ph, xs[idxs[0]], zs[idxs[0]] = mapping[state]
                phase = (phase + ph) % 4
            else:
                state = (xs[idxs[0]], zs[idxs[0]], xs[idxs[1]], zs[idxs[1]])
                ph, xs[idxs[0]], zs[idxs[0]], xs[idxs[1]], zs[idxs[1]] = mapping[state]
                phase = (phase + ph) % 4
            return phase, xs, zs

        for g in range(len(phases)):  # update all generators
            if std == "T":
                # execute T gadget
                phases[g], xs[g], zs[g] = conjugateLookup("CX", [idxs[0], tpos], phases[g], xs[g], zs[g])
                if y[tpos-n] == 1:
                    phases[g], xs[g], zs[g] = conjugateLookup("S", idxs, phases[g], xs[g], zs[g])

                # prepend HS^\dagger to t registers in compiled circuit
                for tUpdateGate in ["H", "S", "S", "S"]:
                    phases[g], xs[g], zs[g] = conjugateLookup(tUpdateGate, [tpos], phases[g], xs[g], zs[g])

                continue

            phases[g], xs[g], zs[g] = conjugateLookup(std, idxs, phases[g], xs[g], zs[g])

        if std == "T":
            tpos += 1

    return phases, xs, zs


# turn list of generators to complete list of elements
# relies on fact that generators commute!
def expandGenerators(proj):
    (phases, xs, zs) = proj
    newPhases, newXs, newZs = [], [], []
    k = len(phases)
    if (k == 0):
        return ([], [], [])

    n = len(xs[0])

    def ph(xs1, zs1, xs2, zs2):
        out = 0
        for i in range(n):
            tup = (xs1[i], zs1[i], xs2[i], zs2[i])
            if tup == (0, 1, 1, 0): out += 2  # Z*X
            if tup == (0, 1, 1, 1): out += 2  # Z*XZ
            if tup == (1, 1, 1, 0): out += 2  # XZ*X
            if tup == (1, 1, 1, 1): out += 2  # XZ*XZ
        return out

    for i in range(1, 2**k):  # omit identity
        bitstring = list(np.binary_repr(i, width=k))
        prod = (0, np.zeros(n), np.zeros(n))
        for j in range(k):
            if bitstring[j] == '1':
                phplus = ph(prod[1], prod[2], xs[j], zs[j])
                prod = (prod[0] + phases[j] + phplus, prod[1] + xs[j], prod[2] + zs[j])

        newPhases.append(prod[0] % 4)
        newXs.append(prod[1] % 2)
        newZs.append(prod[2] % 2)

    return (newPhases, newXs, newZs)


def main():
    # read data
    filename = "examples/compiled2.circ"
    f = open(filename, "r")
    circuit = removeComments(f.read())
    f.close()

    # count T gates and measurements
    Ts, Ms, MTs = countY(circuit)

    # postselect measured qubits
    Mselect = np.random.randint(2, size=len(Ms))

    # increment Ts by those depending on measurement results
    for M in MTs:
        idx = Ms.index(M)
        if Mselect[idx] == 1:
            Ts += 1

    # pack measurements into dictionary
    Mdict = {}
    for M in Ms:
        idx = Ms.index(M)
        Mdict[M] = Mselect[idx]

    # postselect ts
    y = np.random.randint(2, size=Ts)

    measure = {0: 1, 1: 1}

    # obtain projectors
    projector = gadgetize(circuit, measure, Mdict, y)
    print(projector)


if __name__ == "__main__":
    main()
