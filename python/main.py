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
from multiprocessing.pool import ThreadPool


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

    # compile
    circ = circuit.compile.compileRawCircuit(infile, reference)

    # postselect measurement results
    Ts, Ms, MTs = circuit.gadgetize.countY(circ)
    # Ts - number of T gates
    # Ms - qubits measured by circuit
    # MTs - qubits measured by circuit, depending on
    #               which a T gate is performed

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

    # postselect measurement results on Ts
    # y = np.random.randint(2, size=Ts)
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
    G = circuit.gadgetize.expandGenerators(G)
    H = circuit.gadgetize.expandGenerators(H)

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

        cut = 0
        for g in range(len(phases)):
            if sum(xs[g][:n]) > 0:
                cut += 1
            else:
                # ensure no duplicates
                found = False
                for h in range(len(primePhases)):
                    if primePhases[h] == phases[g] and\
                            np.allclose(primeXs[h], xs[g][n:]) and\
                            np.allclose(primeZs[h], zs[g][n:]):
                                found = True
                                print("cutexisting")
                                break

                # remove identity
                if phases[g] == 0 and\
                        np.allclose(xs[g][n:], np.zeros(Ts)) and\
                        np.allclose(zs[g][n:], np.zeros(Ts)):
                    found = True
                    print("cutidentity")

                if found:
                    pass
                    # cut += 1
                else:
                    primePhases.append(phases[g])
                    primeXs.append(xs[g][n:])
                    primeZs.append(zs[g][n:])

        cutdim = np.log2(len(phases)+1-cut)
        if (np.round(cutdim) != cutdim):
            print("Previous size: %d, cut away %d" % (len(phases)+1, cut))
            printProjector((primePhases, primeXs, primeZs))
            raise ValueError("Truncated projector size not a power of two!")
        du = np.log2(len(phases)+1) - cutdim
        print("was %d, remove %d, du %d" % (len(phases)+1, cut, du))
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
                norm, decomposed = decompose(Ts, 0.001)
                print(decomposed, norm)

            # Evaluate the following expression:
            # || \Pi |H^{\otimes t}> ||^2 ~= (2^t / L) \sum^L_(i=1) || <\theta_i| \Pi |H^(\otimes t)> ||^2
            # = (2^t / L) \sum^L_(i=1) || (1/norm)  <\theta_i| \Pi (\sum^\chi_a |\phi_a>) ||^2

            L = int(2000)
            # number of random stabilizer states = 1/(p_f \eps^2), such that with probability (1-p_f) the  result
            # \alpha satisfies ||\Pi |H^{\otimes t}> ||^2 (1-\eps) < \alpha < ||\Pi |H^{\otimes t}> ||^2 (1+\eps):

            np.random.seed(0)  # make random numbers the same every time
            np.seterr(all='raise')  # make numpy crash on warnings

            def sampleState():
                theta = StabilizerState.randomStabilizerState(Ts)

                projfactor = 1
                # project theta down to \Pi |\theta>
                for g in range(len(phases)):
                    res = theta.measurePauli(phases[g], zs[g], xs[g])
                    projfactor *= res
                    # if res == 0:
                    #     return 0, 0

                subtotal = 0  # <\theta_i| \Pi (\sum^\chi_a |\phi_a>)

                for i in range(0, 2**len(decomposed)):
                    Lbits = list(np.binary_repr(i, width=len(decomposed)))
                    bitstring = np.zeros(Ts)
                    k = 0
                    for idx in range(len(Lbits)):
                        if Lbits[idx] == '1':
                            k += 1
                            bitstring += decomposed[idx]
                    bitstring = bitstring.astype(int)
                    # print("nya1", bitstring)

                    # initialize stabilizer state
                    phi = StabilizerState(Ts, k)

                    # insert basis of affine space into G
                    # space is supported whenever xtilde = 1
                    rowidx = 0
                    for xtildeidx in range(k):
                        if bitstring[xtildeidx] == 1:
                            phi.G[rowidx] = np.zeros(Ts)
                            phi.G[rowidx][xtildeidx] = 1
                            rowidx += 1

                    # make sure matrix isn't singular
                    count = 0
                    while np.linalg.matrix_rank(phi.G) != Ts:
                        count += 1
                        if count > 1e3:
                            print(phi.G)
                            raise ValueError("can't make non-singular")

                        # randomly sample remaining rows
                        for i in range(k, Ts):
                            phi.G[i] = np.random.random_integers(0, 1, (Ts))

                    phi.Gbar = np.linalg.inv(phi.G).T

                    # add <\theta_i| \Pi |\phi_a>
                    subtotal += StabilizerState.innerProduct(theta, phi)

                # add || <\theta_i| \Pi |H^(\otimes t)> ||^2
                return np.abs(projfactor*subtotal/norm)**2

            # parallelize samples over L
            # pool = ThreadPool(processes=len(decomposed)*len(phases))
            # results = []
            # for i in range(L):
            #     results.append(pool.apply_async(sampleState, ()))

            total = 0  # sum without the factor (2^t / L)
            for i in range(L):
                total += sampleState()
                # total += results[i].get()

            print("total:", total/L)

            result = (2**Ts / L)*total

        if isG:
            numerator = result
        else:
            denominator = result

        isG = False

    print("numerator: %f" % numerator)
    print("denominator: %f" % denominator)

    result = (2**u)*numerator/denominator
    print("Probability of 0: %f" % result)


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
