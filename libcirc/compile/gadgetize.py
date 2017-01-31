#
# file: gadgetize.py
# input a file with a compiled circuit (h,s,CNOT,t) only
# output stabilizer projectors g(x,y) and h(x)
#

from libcirc.compile.compilecirc import standardGate
import numpy as np


# count the number of random bits to generate
# this is the number of Ts plus measured ancillas
# assumes circuit consists only of gates in the gate set, i.e. is compiled
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
# Mdict: postselected bits to determine circuit
# y: postselected T ancilla measurements
def gadgetize(circuit, Mdict, y):
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

    # set measured ancillas
    for M in Mdict.keys(): phases, xs, zs = gen(M, Mdict[M], phases, xs, zs)

    # set T ancillas
    for i in range(n, size): phases, xs, zs = gen(i, y[i-n], phases, xs, zs)

    # conjugate stabilizer generators:
    tpos = n

    # from main import printProjector
    # printProjector((phases, xs, zs))
    for line in reversed(circuit.splitlines()):
        if line == "": continue

        gate = standardGate(line, lineMode=True)

        #  skip if contains a measurement
        if 'M' in gate:
            Midx = gate.index('M')
            if Mdict[Midx] == 0: continue

        std = standardGate(line)

        if False:
            pr = "  "
            for i in line:
                if i == "T":
                    pr = pr + "[" + i + str(y[tpos-n]) + "]"
                elif i == "M":
                    pr = pr + "[" + i + "1]"
                else:
                    pr = pr + "[" + i + " ]"
            print(pr)

        idxs = []
        for letter in std:
            idxs.append(gate.index(letter))

        lookup = {
            "H": {(0, 0): (0, 0, 0), (1, 0): (0, 0, 1), (0, 1): (0, 1, 0), (1, 1): (2, 1, 1)},  # X->Z, Z->X
            "X": {(0, 0): (0, 0, 0), (1, 0): (0, 1, 0), (0, 1): (2, 0, 1), (1, 1): (2, 1, 1)},  # X->X, Z->-Z
            "S": {(0, 0): (0, 0, 0), (1, 0): (1, 1, 1), (0, 1): (0, 0, 1), (1, 1): (1, 1, 0)},  # X->Y, Z->Z
        }

        def conjugateLookup(std, idxs, phase, xs, zs):
            if (len(std) == 1):
                mapping = lookup[std]
                state = (xs[idxs[0]], zs[idxs[0]])
                ph, xs[idxs[0]], zs[idxs[0]] = mapping[state]
                phase = (phase + ph) % 4
            else:  # evaluate CNOT
                top = ""
                bot = ""
                if xs[idxs[0]] == 1:
                    top += "X"
                    bot += "X"
                if zs[idxs[0]] == 1: top += "Z"
                if xs[idxs[1]] == 1: bot += "X"
                if zs[idxs[1]] == 1:
                    top += "Z"
                    bot += "Z"

                top = top.replace("ZZ", "")
                bot = bot.replace("XX", "")

                xs[idxs[0]] = 1 if ("X" in top) else 0
                zs[idxs[0]] = 1 if ("Z" in top) else 0
                xs[idxs[1]] = 1 if ("X" in bot) else 0
                zs[idxs[1]] = 1 if ("Z" in bot) else 0

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

        # printProjector(([phases[0]], [xs[0]], [zs[0]]))
        # printProjector((phases, xs, zs))
        # test = input()
        # print("")

    # printProjector((phases, xs, zs))
    # raise ValueError("hi")
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
