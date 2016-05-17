#
# file: main.py
# Putting it all together: given an input circuit, sample a set
# of output qubits
#

import numpy as np
import sys
import re
import circuit.compile
import circuit.gadgetize
from stabilizer.stabilizer import StabilizerState
from decompose import decompose
# from multiprocessing.pool import ThreadPool


def main(argv):
    # --------------------- Parse arguments -----------------------
    if len(argv) < 2:
        return usage("Wrong number of arguments.")

    if argv[1] == "-h":
        return help()

    if len(argv) < 3 or len(argv) > 4:
        return usage("Wrong number of arguments.")

    if not re.match("^(M|_)*$", argv[2]):
        return usage("Measurement string must consists only of 'M' or '_'.")

    # load input circuit
    f = open(argv[1], "r")
    infile = circuit.compile.removeComments(f.read())
    f.close()

    n = len(circuit.compile.standardGate(infile.splitlines()[0], lineMode=True))

    # measurement string correct length
    if (len(argv[2]) != n):
        return usage("""Measurement not the right length:
Expected %d, but input is %d long.""" % (n, len(argv[2])))

    # TODO: support for larger measurements
    w = len(argv[2].replace("_", ""))
    # if w > 1: return usage("No support for measurements of multiple qubits.")

    if w == 0: return usage("Must measure at least one bit.")

    # load reference
    referencePath = "circuit/reference"
    if len(argv) > 3: referencePath = argv[2]

    f = open(referencePath)
    reference = circuit.compile.parseReference(circuit.compile.removeComments(f.read()))
    f.close()

    # --------------------- Prepare Projectors ------------------

    # compile, update n to include ancillas
    circ = circuit.compile.compileRawCircuit(infile, reference)
    n = len(circuit.compile.standardGate(circ.splitlines()[0], lineMode=True))

    # postselect measurement results
    Ts, Ms, MTs = circuit.gadgetize.countY(circ)
    # Ts - number of T gates
    # Ms - qubits measured by circuit
    # MTs - qubits measured by circuit, depending on
    #               which a T gate is performed

    # postselect measured qubits
    # Mselect = np.random.randint(2, size=len(Ms))
    Mselect = np.zeros(len(Ms))

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

    # postselect measurement results on Ts
    # y = np.random.randint(2, size=Ts)
    # y = np.ones(Ts).astype(int)
    y = np.zeros(Ts).astype(int)

    print("Measurements: %d" % len(Ms))
    print("Ts: %d" % Ts)

    # assemble measurement pauli
    # TODO: support larger measurements
    measure = {}
    # measure[argv[2].index("M")] = 0
    for i in range(len(argv[2])):
        if list(argv[2])[i] == 'M':
            measure[i] = 0

    print("y: %s" % "".join(list(y.astype(str))))

    # obtain projectors
    # TODO: opportunity for efficiency improvement: H is a subset
    # of G's generators, namely those from the T projectors.
    # could calculate G = H + gadgetize(circ, measure, no t's)
    # In fact, this could be extended to entire calculation.
    G = circuit.gadgetize.gadgetize(circ, measure, Mdict, y)
    H = circuit.gadgetize.gadgetize(circ, {}, Mdict, y)

    # convert list of generators to list of all projectors
    # G = circuit.gadgetize.expandGenerators(G)
    # H = circuit.gadgetize.expandGenerators(H)

    print("G:")
    printProjector(G)
    print("H:")
    printProjector(H)

    # truncate projectors to magic state part
    u = 0  # append normfactor 2**(-u)
    isG = True
    for proj in [G, H]:
        (phases, xs, zs) = proj
        primePhases = []
        primeXs = []
        primeZs = []

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
            gen = (0, np.zeros(n+Ts), np.zeros(n+Ts))
            for i in range(len(row)):
                if row[i] == 1: gen = mulStabilizers(gen, (phases[i], xs[i], zs[i]))
            (ph, x, z) = gen
            primePhases.append(ph)
            primeXs.append(x[-Ts:])
            primeZs.append(z[-Ts:])

        du = len(phases) - len(X)
        print("du: ", du)
        if isG:
            u -= du
            Gprime = (primePhases, primeXs, primeZs)
        else:
            u += du
            Hprime = (primePhases, primeXs, primeZs)

        isG = False

    # --------------------- Evaluate Inner Products ---------------

    print("Gprime:")
    printProjector(Gprime)
    print("Hprime:")
    printProjector(Hprime)
    print("u: %d" % u)

    # TODO: if Gprime = Hprime result is 1

    isG = True
    decomposed = False
    for proj in [Gprime, Hprime]:
        (phases, xs, zs) = proj

        # empty projector
        if len(phases) == 0:
            if isG:
                numerator = 1
            else:
                denominator = 1
            isG = False
            continue

        if Ts == 0:  # all cliffords circuit
            lookup = {0: 1, 2: -1}
            generators = [1]  # include identity
            for phase in phases: generators.append(lookup[phase])

            # calculate sum of all permutations of generators
            result = sum(generators)/len(generators)

        else:
            # compute decomposition if not already done so
            if (type(decomposed) == bool):
                norm, decomposed = decompose(Ts, 0.0001)
                print(decomposed, norm)

            # Evaluate the following expression:
            # || \Pi |H^{\otimes t}> ||^2 ~= (2^t / L) \sum^L_(i=1) || <\theta_i| \Pi |H^(\otimes t)> ||^2
            # = (2^t / L) \sum^L_(i=1) || (1/norm)  <\theta_i| \Pi (\sum^\chi_a |\phi_a>) ||^2

            L = int(5000)
            # number of random stabilizer states = 1/(p_f \eps^2), such that with probability (1-p_f) the  result
            # \alpha satisfies ||\Pi |H^{\otimes t}> ||^2 (1-\eps) < \alpha < ||\Pi |H^{\otimes t}> ||^2 (1+\eps):

            np.random.seed(0)  # make random numbers the same every time
            np.seterr(all='raise')  # make numpy crash on warnings

            def sampleState():
                theta = StabilizerState.randomStabilizerState(Ts)

                projfactor = 1
                # project theta down to \Pi |\theta>

                def printState(x):
                    outstring = "["
                    for comp in x:
                        outstring += " %0.3f+%0.3fi " % (comp.real, comp.imag)
                    outstring += "]"
                    print(outstring)

                # print("\n\nTry:")
                # printState(theta.unpack())
                factors = []
                for g in range(len(phases)):
                    # print((phases[g], zs[g], xs[g]))
                    res = theta.measurePauli(phases[g], zs[g], xs[g])
                    # printState(theta.unpack())
                    # print("Out: ", res)

                    projfactor *= res
                    factors.append(res)

                    if (res == 0):  # don't need to consider more
                        break

                # if (len(factors) == 3 and factors[2] != 1):
                #     print(factors)

                subtotal = 0  # <\theta_i| \Pi (\sum^\chi_a |\phi_a>)

                for i in range(0, 2**len(decomposed)):
                    Lbits = list(np.binary_repr(i, width=len(decomposed)))
                    bitstring = np.zeros(Ts)
                    for idx in range(len(Lbits)):
                        if Lbits[idx] == '1':
                            bitstring += decomposed[idx]
                    bitstring = bitstring.astype(int) % 2

                    # initialize stabilizer state
                    phi = StabilizerState(Ts, Ts)

                    # construct state by measuring paulis
                    for xtildeidx in range(Ts):
                        vec = np.zeros(Ts)
                        vec[xtildeidx] = 1
                        if bitstring[xtildeidx] == 1:
                            # |+> at index, so measure X
                            phi.measurePauli(0, np.zeros(Ts), vec)
                        else:
                            # |0> at index, so measure Z
                            phi.measurePauli(0, vec, np.zeros(Ts))

                    # old code: matrix inverse method
                    # insert basis of affine space into G
                    # space is supported whenever xtilde = 1
                    # rowidx = 0
                    # for xtildeidx in range(Ts):
                    #     if bitstring[xtildeidx] == 1:
                    #         phi.G[rowidx] = np.zeros(Ts)
                    #         phi.G[rowidx][xtildeidx] = 1
                    #         rowidx += 1

                    # make sure matrix has determinant 1, ensuring existence of inverse
                    # count = 0
                    # while np.abs(np.linalg.det(phi.G)) != 1:
                    #     count += 1
                    #     if count > 1e3:
                    #         print("k: ", k)
                    #         print("bitstring: ", bitstring)
                    #         print("Lbits: ", Lbits)

                    #         print(phi.G)
                    #         import pdb; pdb.set_trace()
                    #         raise ValueError("can't make non-singular")

                        # randomly sample remaining rows
                    #     for i in range(k, Ts):
                    #         phi.G[i] = np.random.random_integers(0, 1, (Ts))

                    # phi.Gbar = np.linalg.inv(phi.G).T % 2

                    # if not np.allclose(np.dot(phi.G, phi.Gbar.T) % 2, np.eye(Ts)):
                    #     print(phi.G)
                    #     print(phi.Gbar.T)
                    #     import pdb; pdb.set_trace()
                    #     raise ValueError("bad inverse")

                    # add <\theta_i| \Pi |\phi_a>
                    try:
                        subtotal += StabilizerState.innerProduct(theta, phi)
                    except IndexError:
                        import pdb; pdb.set_trace()

                # add || <\theta_i| \Pi |H^(\otimes t)> ||^2
                return np.abs(projfactor*subtotal)**2, projfactor  # norm necessary?
                return np.abs(projfactor*subtotal/norm)**2

            # parallelize samples over L
            # pool = ThreadPool(processes=len(decomposed)*len(phases))
            # results = []
            # for i in range(L):
            #     results.append(pool.apply_async(sampleState, ()))

            total = 0  # sum without the factor (2^t / L)
            stats = {}
            print("Stats:")
            for i in range(L):
                item, proj = sampleState()
                total += item
                x = np.abs(proj)**2
                if (x in stats): stats[x] += 1
                else: stats[x] = 1

                # total += results[i].get()

            x = 0
            for item in stats:
                x += 1
                print("%0.3f -> %d = %0.3f" % (item, stats[item], 100*stats[item]/L))
            print("So many: ", x)

            # print("total:", total/L)

            # maybe (2**Ts / L) isn't necessary?
            # unless one of the projectors is empty...
            result = (2**Ts / L)*total
            # result = total  # norm necessary?

        if isG:
            numerator = result
        else:
            denominator = result

        isG = False

    print("numerator: %f" % numerator)
    print("denominator: %f" % denominator)

    result = (2**u)*numerator/denominator
    print("Probability of 0: %f" % result)
    print("Rounded Probability of 0: %0.2f" % np.round(result, 2))


def evalStabilizer(m, xi, zeta, bitstring, stateseed):
    return 0


def printProjector(projector):
    phases, xs, zs = projector
    for g in range(len(phases)):
        phase, x, z = phases[g], xs[g], zs[g]
        tmpphase = phase
        genstring = ""
        for i in range(len(x)):
            if (x[i] == 1 and z[i] == 1):
                tmpphase -= 1  # divide phase by i
                genstring += "Y"
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


def usage(error):
    print(error + """\n\nStabilizer simulator usage:
main.py <circuitfile> <measurement> [reference=circuits/reference]

Full help statement: main.py -h""")


def help():
    print("""Stabilizer simulator usage:
main.py <circuitfile> <measurement> [reference=circuit/reference]

Example:
main.py examples/controlledH.circ _M
Execute controlled H gate on state |00>, then measure 2nd qubit.
Expected result: 0 always.


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
A string of "_", "M", with length equal to the number of qubits in
the input file. The algorithm samples from the distribution of
basis states of the qubits marked with "M".

[reference=circuits/reference]
Location of a reference file defining new gates in terms of the
standard gate set. Please read the documentation in the provided
reference file for details.""")

if __name__ == "__main__":
    main(sys.argv)
