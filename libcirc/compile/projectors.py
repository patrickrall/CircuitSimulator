#
# file: projectors.py
# Putting it all together: calculate projectors G and H,
# and truncate them to the magic state subspace.
#

import numpy as np
import libcirc.compile.compilecirc as compilecirc
import libcirc.compile.gadgetize as gadgetize


# obtain projectors G, H
# circ: A compiled circuit string.
# measure: A dictionary mapping qubits indexes to 0 or 1.
#          Not all qubits need to be measured.
# Measurements by the circuit and T gate measurements
# are selected automatically at random. Setting the zeros
# flag turns off randomness and makes all measurements |0>.
def projectors(circ, measure, verbose=False, x=None, y=None):
    n = len(compilecirc.standardGate(circ.splitlines()[0], lineMode=True))

    t, Ms, MTs = gadgetize.countY(circ)
    # t  - number of T gates
    # Ms - qubits measured by circuit
    # MTs - qubits measured by circuit, depending on
    #               which a T gate is performed

    # postselect circuit measured qubits
    if x is None:
        Mselect = np.random.randint(2, size=len(Ms))
    else:
        tmpX = []
        if len(x) != len(Ms): raise ValueError("x needs to have length %d" % len(Ms))
        for l in x:
            if l == "0": tmpX.append(0)
            elif l == "1": tmpX.append(1)
            else: raise ValueError("Only 0s and 1s allowed in x string.")
        Mselect = np.array(tmpX).astype(int)

    # Increment t for measurement-dependent Ts
    for M in MTs:
        idx = Ms.index(M)
        if Mselect[idx] == 1:
            t += 1

    # pack circuit into measurement dictionary
    Mdict = {}
    for M in Ms:
        idx = Ms.index(M)
        Mdict[M] = Mselect[idx]
        if M in measure.keys(): continue
        measure[M] = Mselect[idx]

    # postselect measurement results on Ts
    if y is None:
        y = np.random.randint(2, size=t)
    else:
        tmpY = []
        if len(y) != t: raise ValueError("y needs to have length %d" % t)
        for l in y:
            if l == "0": tmpY.append(0)
            elif l == "1": tmpY.append(1)
            else: raise ValueError("Only 0s and 1s allowed in y string.")
        y = np.array(tmpY).astype(int)

    # Print measurement information:
    if verbose:
        print("n = %d, t = %d" % (n, t))
        print("Measurement info:")

        mstring = ""
        for i in range(n):
            if i in measure:
                mstring += str(measure[i])
            else: mstring += "_"
        print("x: " + mstring)

        print("y: " + "".join(y.astype(str).tolist()))

    G = gadgetize.gadgetize(circ, measure, y)
    H = gadgetize.gadgetize(circ, Mdict, y)

    return G, H, n, t  # also return n and t for convenience


# truncates first n qubits of P by projecting onto |0^n>
def truncate(n, P):
    primePhases = []
    primeXs = []
    primeZs = []

    (phases, xs, zs) = P
    if len(phases) == 0:  # trivial case: empty projector
        return (primePhases, primeXs, primeZs), 0

    t = len(xs[0]) - n

    def nullSpace(A):
        l = len(A)
        X = np.eye(l)

        # modify X for each qubit
        for i in range(len(A[0])):
            # compute rows of X with 1's for that qubit
            y = np.dot(X, A[:, i]) % 2

            good = []  # rows with 0's
            bad = []   # rows with 1's

            for i in range(len(y)):
                if (y[i] == 0): good.append(X[i])
                else: bad.append(X[i])

            # add bad rows to each other to cancel out 1's in y
            bad = (bad + np.roll(bad, 1, axis=0)) % 2
            bad = bad[1:]

            # ensure arrays are properly shaped, even if empty
            good = np.array(good).reshape((len(good), l))
            bad = np.array(bad).reshape((len(bad), l))

            X = np.concatenate((good, bad), axis=0)
        return X

    def mulStabilizers(stab1, stab2):
        (ph1, xs1, zs1) = stab1
        (ph2, xs2, zs2) = stab2

        ph = 0
        for i in range(len(xs1)):
            tup = (xs1[i], zs1[i], xs2[i], zs2[i])
            if tup == (0, 1, 1, 0): ph += 2  # Z*X
            if tup == (0, 1, 1, 1): ph += 2  # Z*XZ
            if tup == (1, 1, 1, 0): ph += 2  # XZ*X
            if tup == (1, 1, 1, 1): ph += 2  # XZ*XZ

        return ((ph1 + ph2 + ph) % 4, (xs1 + xs2) % 2, (zs1 + zs2) % 2)

    A = np.array(xs)[:, :n]
    X = nullSpace(A)

    # debug: verify null space
    for row in X:
        if not np.allclose(np.dot(row, A) % 2, np.zeros(len(A[0]))):
            import pdb; pdb.set_trace()
            raise ValueError("Null space wrong.")

    for row in X:
        gen = (0, np.zeros(n+t), np.zeros(n+t))
        for i in range(len(row)):
            if row[i] == 1: gen = mulStabilizers(gen, (phases[i], xs[i], zs[i]))
        (ph, x, z) = gen

        if ph == 0 and np.allclose(x[n:], np.zeros(t)) and np.allclose(z[n:], np.zeros(t)): continue  # exclude positive identity

        primePhases.append(ph)
        primeXs.append(x[n:])
        primeZs.append(z[n:])

    u = len(phases) - len(X)
    return (primePhases, primeXs, primeZs), u
