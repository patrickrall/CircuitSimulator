#
# file: probability.py
# Programmatical interface for application. Implements probability
# subroutine and sampling algorithm.
#

import sys
import numpy as np
from multiprocessing import Pool
from subprocess import PIPE, Popen
import os

from libcirc.sample import decompose, sampleProjector
import libcirc.projectors as projectors
from libcirc.noapprox import calcExactProjector


# calculate probability that a compiled circuit yields a measurement
# circ is a compiled quantum circuit.
#
# measure is a dictionary with zero-indexed qubit indexes and measurement values.
# E.g. to calculate the probabilty of measuring 1 for the first qubit use {0:1}
#
# samples governs the accuracy. 1e4 gives error < 0.01 with 95% chance.
#
# config is dictionary with several optional parameters.
# Read below for details. The value in the comment is the default value.
# config = {
#         Logging parameters
#     "verbose": False,  # print useful information
#     "silenceprojectors": False, # do not print projectors despite verbosity
#     "quiet": False, # silence all warning messages and progress bars. Overridden by verbose.
#     "direct": False, # returns tuple (numerator, denominator) rather than the ratio
#
#         Sampling method specification parameters.
#         exact overrides k, k overrides fidbound.
#         exact=True by default, so set exact=False to use k or fidbound.
#               Note: the main.py front end will set exact=False for you
#                     if k or fidbound options are set.
#         If exact=False then at least one of k or fidbound must be set.
#         fidelity=True is ignored if exact=True or k is set.
#     "exact": True,   # Use |H> instead of |L>
#     "k": None,   # size of L matrix defining |L>
#     "fidbound": 1e-5, # inner product <L|H> must be this close to 1
#     "fidelity": False,   # Calculate and/or verify inner product <L|H> (pretty expensive)
#                          # If k is used, this is only for display purposes,
#                          # so it is disabled by quiet=True.
#     "rank": False,   # Verify rank of L matrix (very expensive)
#
#         Backend configuration.
#         If python=False, then cpath must be specified. Default value of
#         cpath should be good if executing from the repo root directory.
#     "python": False,  # Use python backend
#     "cpath": "libcirc/sample",  # Location of c executable
#     "mpirun": "/usr/bin/mpirun",  # Location of mpirun executable, and any options
#     "procs": None,  # number of processes. If unset python or mpi will pick automatically.
#     "stateParallel": False, # Parallelize over state decomposition.
#                             # More overhead, but gives progress bar.
#     "file": None, # instead of using c backend, print instructions to file
#
#         Debug. x and y determine the projectors, but are arbitrary in the end.
#         If unspecified they are selected at random, as required by the sampling algorithm.
#     "x": None,   # Specify postselected other measurements
#     "y": None,   # Specify postselected T measurements
#     "forceL": False,  # use L sampling even when exact sampling is more efficient
#                           Setting k does this automatically.
# }
# For more details see the command line's help text or the documentation
def probability(circ, measure, samples=1e4, config={}):
    # unpack, configure default values
    verbose = config.get("verbose")
    if verbose: config["quiet"] = False
    quiet = config.get("quiet")
    if config.get("exact") is None: config["exact"] = True
    if config.get("fidbound") is None: config["fidbound"] = 1e-5
    if config.get("cpath") is None: config["cpath"] = "libcirc/sample"
    if config.get("mpirun") is None: config["mpirun"] = "/usr/bin/mpirun"
    if config.get("procs") is 0: config["procs"] = None
    samples = int(samples)

    # if k is used for L calculation, then fidelity=True is only for display.
    # thus set fidelity=False if quiet flag is set.
    if not config.get("exact") and config.get("k") is not None:
        if config.get("quiet") and config.get("fidelity"):
            config["fidelity"] = False

    # get projectors
    G, H, n, t = projectors.projectors(circ, measure, verbose=verbose, x=config.get("x"), y=config.get("y"))

    # truncate projectors
    Gprime, u = projectors.truncate(n, G)
    Hprime, v = projectors.truncate(n, H)

    # print projector data
    if verbose and not config.get("silenceprojectors"):
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
        if not quiet: print("Empty projectors found. Using exact decomposition to compute norm.")
        config["exact"] = True

    # Clifford circuit? Don't bother calling c implementation.
    # if t == 0:
    #     if not quiet: print("Clifford circuit. Reverting to python implementation.")
    #     config["python"] = True

    # verify existence of executable
    if not config.get("python"):
        if config.get("cpath") is None:
            if not quiet: print("C executable unspecified. Reverting to python implementation.")
            config["python"] = True

        elif not os.path.isfile(config.get("cpath")):
            if not quiet: print("Could not find c executable at: " + config.get("cpath"))
            if not quiet: print("Reverting to python implementation")
            config["python"] = True

    # ------------------------------------ Python backend ------------------------------------
    if config.get("python"):
        # calculate |L> ~= |H>
        L, Lnorm = decompose(t, config)

        if verbose:
            if L is None: print("Using exact decomposition of |H^t>: 2^%d" % t)
            else: print("Stabilizer rank of |L>: 2^%d" % len(L))

        # parallelization
        stateParallel = config.get("stateParallel")
        # calculate || Gprime |L> ||^2 and || Hprime |L> ||^2 up to a constant
        if not config.get("noparallel"):
            if not stateParallel:
                if verbose: print("Parallelizing over %d samples" % samples)
                # set up thread pool
                pool = Pool(config.get("procs"))
            else:
                if verbose:
                    if L is None: print("Parallelizing over %d stabilizers in |H^t>" % 2**np.ceil(t/2))
                    else: print("Parallelizing over %d stabilizers in |L>" % 2**len(L))
                pool = None
        else:
            if not quiet: print("Parallelism disabled for debugging.")
            pool = None
            stateParallel = 0

        # helper for preventing needless sampling trivial projectors or circuits
        def calcProjector(P, name=""):
            if Lnorm is not None and (len(P[0]) == 0 or len(P[1][0]) == 0):
                # empty projector or Clifford circuit. No repeated sampling needed.
                return samples * sampleProjector((P, L, 0, False)) * np.abs(Lnorm)**2

            if pool is not None:
                queries = [(P, L, seed, stateParallel) for seed in range(0, samples)]
                return sum(pool.map(sampleProjector, queries))

            # show progress when not parallelizing over samples
            out = 0
            for i in range(0, samples):
                out += sampleProjector((P, L, i, stateParallel))
                print(name + ": %d/%d samples" % (i+1, samples), end="\r")
            return out

        numerator = calcProjector(Gprime, name="Numerator")
        denominator = calcProjector(Hprime, name="Denominator")
        # numerator = calcExactProjector(Gprime, Lnorm)
        # denominator = calcExactProjector(Hprime, Lnorm)

        if pool is not None:
            pool.close()
            pool.join()

    else:
        # --------------------------------------- C backend --------------------------------------
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

        if not config.get("file"):
            executable = config["mpirun"].split()
            if config.get("procs") is not None:
                executable += ["-n", str(config["procs"])]

            executable += [config["cpath"], "stdin"]

            p = Popen(executable, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=1)

        indat = b""

        indat += send(len(Gprime[1][0]))  # t

        if config.get("k") is None: indat += send(0)  # k
        else: indat += send(config["k"])

        indat += send(1 if config.get("quiet") else 0)  # quiet
        indat += send(1 if config.get("verbose") else 0)  # verbose

        indat += send(config.get("fidbound") if config.get("fidbound") else 1e-5)  # fidbound
        indat += send(1 if config.get("exact") else 0)  # exact
        indat += send(1 if config.get("rank") else 0)  # rank
        indat += send(1 if config.get("forceK") else 0)  # forceK
        indat += send(1 if config.get("fidelity") else 0)  # fidelity

        indat += send(1 if config.get("stateParallel") else 0)  # stateParallel

        indat += writeProjector(Gprime)
        indat += writeProjector(Hprime)

        indat += send(samples)  # samples

        if config.get("file"):
            f = open(config.get("file"), "w")
            f.write(indat.decode())
            if not quiet: print("Wrote instruction file to %s." % config.get("file"))

            return None

        p.stdin.write(indat)
        p.stdin.flush()

        out = []
        while True:
            p.stdout.flush()
            line = 0
            line = p.stdout.readline().decode()[:-1]

            if not line:
                break

            if ("Numerator:" in line or "Denominator:" in line):
                if not quiet:
                    sys.stdout.write(line + "\r")
                    sys.stdout.write("\033[K")
            else:
                out.append(line)

        success = True
        try:
            numerator = float(out[-2])
            denominator = float(out[-1])
        except:
            # these are errors, not warnings, so they are not silenced by quiet

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

        # print C code output. Should respect quiet flag.
        for line in out[:-2]:
            print(line)

    # ------------------ end backend-dependent code ------------------

    if verbose:
        print("|| Gprime |H^t> ||^2 ~= " + str(numerator/samples))
        print("|| Hprime |H^t> ||^2 ~= " + str(denominator/samples))

    if config.get("direct"):
        return (2**(v-u) * numerator, denominator)

    if numerator == 0: return 0  # deal with denominator == 0
    prob = 2**(v-u) * (numerator/denominator)
    # if (prob > 1): prob = 1.0  # no probabilities greater than 1
    return prob


# sample output from a subset of the qubits
# arguments are same as in probability
# sample is a list of integer indexes, with 0 being the first qubit.
#
# if config["exact"] is unspecified, the default is now False
# also config["file"] is disabled
def sampleQubits(circ, measure, sample, samples=1e4, config={}):
    # unpack config
    verbose = config.get("verbose")
    if config.get("exact") is None: config["exact"] = False
    config["file"] = None

    def prob(measure, exact=None):  # shortcut
        if exact is not None:
            old = config["exact"]
            config["exact"] = exact
        P = probability(circ, measure, samples=samples, config=config)
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
            if verbose: print("-> Sampled " + str(qubit) + "\n")

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
                print("-> Sampled" + qubit)
                print("-> Sample so far:" + qubitssofar + str(qubit) + "\n")

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
