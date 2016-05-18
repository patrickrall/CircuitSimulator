#
# file: projectors.py
# Putting it all together: calculate projectors G and H, truncate
# them to the magic state subspace, and sample || <theta| P |H^t> ||^2
#

import numpy as np
import circuit.compile
import circuit.gadgetize
from stabilizer.stabilizer import StabilizerState
from multiprocessing import Pool


# obtain projectors G, H
# circ: A compiled circuit string.
# measure: A dictionary mapping qubits indexes to 0 or 1.
#          Not all qubits need to be measured.
# Measurements by the circuit and T gate measurements
# are selected automatically at random. Setting the zeros
# flag turns off randomness and makes all measurements |0>.
def projectors(circ, measure, verbose=False, zeros=False):
    n = len(circuit.compile.standardGate(circ.splitlines()[0], lineMode=True))

    t, Ms, MTs = circuit.gadgetize.countY(circ)
    # t  - number of T gates
    # Ms - qubits measured by circuit
    # MTs - qubits measured by circuit, depending on
    #               which a T gate is performed

    # postselect circuit measured qubits
    Mselect = np.random.randint(2, size=len(Ms))
    if zeros: Mselect = np.zeros(len(Ms))

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
    y = np.random.randint(2, size=t)
    if zeros: y = np.zeros(t)

    # Print measurement information:
    if verbose:
        print("n = %d, t = %d" % (n, t))
        print("Measurement info:")

        mstring = ""
        for i in range(n):
            if i in measure:
                mstring += str(measure[i])
            else: mstring += "_"
        print("x:", mstring)

        print("y:", "".join(y.astype(str).tolist()))

    G = circuit.gadgetize.gadgetize(circ, measure, y)
    H = circuit.gadgetize.gadgetize(circ, Mdict, y)

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
            y = np.dot(X, A[:, i])

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


# Inner product for some state in |L> ~= |H^t>
def evalLcomponent(args):
    (i, L, theta, t, seed) = args  # unpack arguments (easier for parallel code)

    # set unique seed for this calculation
    np.random.seed((seed * 2**len(L) + i) % 4294967296)

    # compute bitstring by adding rows of l
    Lbits = list(np.binary_repr(i, width=len(L)))
    bitstring = np.zeros(t)
    for idx in range(len(Lbits)):
        if Lbits[idx] == '1':
            bitstring += L[idx]
    bitstring = bitstring.astype(int) % 2

    # Stabilizer state is product of |0> and |+>
    # 1's in bitstring indicate positions of |+>

    # initialize stabilizer state
    phi = StabilizerState(t, t)

    # construct state by measuring paulis
    for xtildeidx in range(t):
        vec = np.zeros(t)
        vec[xtildeidx] = 1
        if bitstring[xtildeidx] == 1:
            # |+> at index, so measure X
            phi.measurePauli(0, np.zeros(t), vec)
        else:
            # |0> at index, so measure Z
            phi.measurePauli(0, vec, np.zeros(t))

    return StabilizerState.innerProduct(theta, phi)


# Evaluate || <theta| P |H^t> ||^2 for a random stabilizer state theta.
def sampleTMeasure(args):
    (P, L, Lnorm, seed, parallel) = args  # unpack arguments (easier for parallel code)
    (phases, xs, zs) = P

    # empty projector
    if len(phases) == 0:
        return 1

    t = len(xs[0])

    # clifford circuit
    if t == 0:
        lookup = {0: 1, 2: -1}
        generators = [1]  # include identity
        for phase in phases: generators.append(lookup[phase])

        # calculate sum of all permutations of generators
        return sum(generators)/len(generators)

    # sample random theta
    theta = StabilizerState.randomStabilizerState(t)

    # project random state to P
    projfactor = 1
    for g in range(len(phases)):
        res = theta.measurePauli(phases[g], zs[g], xs[g])
        projfactor *= res

        if res == 0: return 0  # theta annihilated by P

    if parallel:  # parallelize for large enough L
        pool = Pool()
        total = sum(pool.map(evalLcomponent, [(i, L, theta, t, seed) for i in range(0, 2**len(L))]))
        pool.close()
        pool.join()
    else:
        total = sum(map(evalLcomponent, [(i, L, theta, t, seed) for i in range(0, 2**len(L))]))

    return 2**t * np.abs(projfactor*total/Lnorm)**2
