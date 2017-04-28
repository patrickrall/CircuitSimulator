#
# file: probability.py
# Programmatical interface for application. Implements probability
# subroutine and sampling algorithm.
#

import sys
import numpy as np
from multiprocessing import Pool, cpu_count
from subprocess import PIPE, Popen
import os

from libcirc.sample import decompose, sampleProjector
import libcirc.projectors as projectors


# calculate probability that a compiled circuit yields a measurement
# config is dictionary of the following format:
# config = {
#     "verbose": True/False,  # print useful information
#     "silenceprojectors": True/False, # do not print projectors despite verbose (optional)
#     "parallel": True/False,  # run in parallel
#     "samples": int(1e3),   # number of samples from inner product distribution
#     "fidbound": 1e-5,   # maximum allowed inner product <L|H>
#     "k": None,   # size of L matrix defining |L>
#     "exact": False,   # Use |H> instead of |L>
#     "rank": False,   # Verify rank of L matrix (expensive)
#     "fidelity": False,   # Calculate inner product <L|H>
#     "y": "000",   # Specify postselected T measurements
#     "x": None,   # Specify postselected other measurements
#     "python": False,  # Use python backend
#     "cpath": os.getcwd() + "/../c/sample",  # Location of c executable
# }
# For more details see the command line's help text or the documentation
def probability(circ, measure, config):

    # unpack
    verbose = config["verbose"]
    parallel = config["parallel"]
    Nsamples = config["samples"]

    # get projectors
    G, H, n, t = projectors.projectors(circ, measure, verbose=verbose, x=config["x"], y=config["y"])

    # truncate projectors
    Gprime, u = projectors.truncate(n, G)
    Hprime, v = projectors.truncate(n, H)

    # print projector data
    if verbose and ("silenceprojectors" not in config or not config["silenceprojectors"]):
        print("Gadgetize circuit:")
        print("G:")
        printProjector(G)
        print("H:")
        printProjector(H)

        print("Truncate projectors to magic state space:")
        print("Gprime: (truncated by %d)" % u)
        printProjector(Gprime)
        print("Hprime: (truncated by %d)" % v)
        printProjector(Hprime)

    # check for -I in numerator
    for i in range(len(Gprime[0])):
        if Gprime[0][i] == 2 and \
           np.allclose(np.zeros(len(Gprime[1][i])), Gprime[1][i]) and \
           np.allclose(np.zeros(len(Gprime[2][i])), Gprime[2][i]):
            if verbose: print("Found negative identity. Answer is 0.")
            return 0

    # check if projectors are identical
    same = True
    for i in range(len(Gprime[0])):
        if i > len(Hprime[0])-1: same = False
        same = (same and Gprime[0][i] == Hprime[0][i] and
                np.allclose(Gprime[1][i], Hprime[1][i]) and
                np.allclose(Gprime[2][i], Hprime[2][i]))

    if same:
        if verbose: print("Projectors are identical.")
        return 2**(v-u)

    # any empty projectors? require exact decomposition so we have norm.
    if len(Gprime[0]) == 0 or len(Hprime[0]) == 0 and not config["exact"]:
        print("Empty projectors found. Using exact decomposition to compute norm.")
        config["exact"] = True

    # verify existence of executable
    if not config["python"]:
        if not os.path.isfile(config["cpath"]):
            print("Could not find c executable at", config["cpath"])
            print("Reverting to python implementation")
            config["python"] = True

    if config["python"]:  # python backend
        # calculate |L> ~= |H>
        L, Lnorm = decompose(t, config["fidbound"], config["k"], config["exact"], config["rank"], config["fidelity"])

        if verbose and L is None:
            print("Using exact decomposition of |H^t>: 2^%d" % t)
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
            if Lnorm is not None and (len(P[0]) == 0 or len(P[1][0]) == 0):
                # empty projector or Clifford circuit. No sampling needed.
                return Nsamples * sampleProjector((P, L, 0, False)) * np.abs(Lnorm)**2

            queries = [(P, L, seed, Lparallel) for seed in range(0, Nsamples)]
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
            denominator = calcProjector(Hprime, Lparallel)

    else:  # c backend
        def send(s):
            return (str(s) + "\n").encode()

        def writeProjector(projector):
            ph, xs, zs = projector

            dat = b""

            dat += send(len(ph))  # Nstabs
            if len(ph) > 0: dat += send(len(xs[0]))  # Nqubits
            else: dat += send(0)  # empty projector
            for i in range(len(ph)):
                dat += send(int(ph[i]))
                for j in range(len(xs[i])):
                    dat += send(int(xs[i][j]))
                    dat += send(int(zs[i][j]))

            return dat

        p = Popen(config["cpath"], stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=1)

        indat = b""

        indat += send(len(Gprime[1][0]))  # t
        if config["k"] is None: indat += send(0)  # k
        else: indat += send(config["k"])  # k
        indat += send(config["fidbound"])  # fidbound
        indat += send(1 if config["exact"] else 0)  # exact
        indat += send(1 if config["rank"] else 0)  # rank
        indat += send(1 if config["fidelity"] else 0)  # fidelity

        indat += writeProjector(Gprime)
        indat += writeProjector(Hprime)

        if config["parallel"]:
            indat += send(int(cpu_count()))
        else:
            indat += send(1)  # one core

        # out = p.communicate(input=indat)
        # out = out[0].decode().splitlines()

        p.stdin.write(indat)
        p.stdin.flush()

        out = []
        while True:
            p.stdout.flush()
            line = p.stdout.readline().decode()[:-1]
            if not line:
                break

            if ("Numerator:" in line or "Denominator:" in line):
                sys.stdout.write(line + "\r")
            else:
                out.append(line)
        sys.stdout.write("\033[K")  #This code is messing up the terminal, apparently not universally honored - Antia

        success = True
        try:
            numerator = float(out[-2])
            denominator = float(out[-1])
        except IndexError:
            print("C code encountered error. Aborting.")
            if len(out) > 0:
                print("Begin C code output:")
                for line in out:
                    print(line)
                print("End C code output.")
            else:
                print("C code gave no output.")
            success = False

        if not success: raise RuntimeError

        if verbose:
            for line in out[:-2]:
                print(line)

    if verbose:
        print("|| Gprime |H^t> ||^2 ~= " + str(numerator/Nsamples))
        print("|| Hprime |H^t> ||^2 ~= " + str(denominator/Nsamples))

    if numerator == 0: return 0  # deal with denominator == 0
    prob = 2**(v-u) * (numerator/denominator)
    if (prob > 1):
        prob = 1.0  # no probabilities greater than 1
    return prob


# sample from a set of qubits
def sampleQubits(circ, measure, sample, config):
    # unpack config
    verbose = config["verbose"]

    def prob(measure, exact=None):  # shortcut
        if exact is not None:
            old = config["exact"]
            config["exact"] = exact
        P = probability(circ, measure, config=config)
        if exact is not None: config["exact"] = old
        return P

    # recursive implementation: need the probability of sampling a smaller
    # set of qubits in order to sample next one
    def recursiveSample(qubits, measure):
        if len(qubits) == 1:  # base case
            Psofar = 1  # overridden if constraints present

            # is a sample even possible? Some qubit constraints are unsatisfiable.
            if len(measure.keys()) != 0:  # other measurements present
                Psofar = prob(measure, exact=True)
                if verbose: print("Constraints present. Probability of satifying: %f\n" % Psofar)
                if Psofar == 0: return "", {}, 0  # not satisfiable
                if Psofar < 1e-5:
                    print("Warning: probability of satisfying qubit constraints very low (%f). Could be impossible." % Psofar)

            # compute probability
            measure[qubits[0]] = 0
            P = prob(measure)
            P0 = P/Psofar

            if verbose:
                print("Measuring qubit %d: P(0) = %f" % (qubits[0], P))
                if len(measure.keys()) != 0:
                    print("Conditional probability of qubit %d: P(0) = %f" % (qubits[-1], P0))

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


def printProjector(projector):
    phases, xs, zs = projector
    for g in range(len(phases)):
        phase, x, z = phases[g], xs[g], zs[g]
        tmpphase = phase
        genstring = ""
        for i in range(len(x)):
            if (x[i] == 1 and z[i] == 1):
                tmpphase -= 1  # divide phase by i
                # genstring += "(XZ)"
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
    if len(phases) == 0:
        print("Projector is empty.")
