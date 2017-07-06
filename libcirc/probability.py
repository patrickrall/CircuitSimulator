#
# file: probability.py
# Programmatical interface for the probability algorithm.
#

import numpy as np
from libcirc.innerprod import *
import libcirc.compile.projectors as projectors
import os
from subprocess import PIPE, Popen


# calculate probability that a compiled circuit yields a measurement
# circ is a compiled quantum circuit.
#
# measure is a dictionary with zero-indexed qubit indexes and measurement values.
# E.g. to calculate the probabilty of measuring 1 for the first qubit use {0:1}
#
# config is dictionary with several optional parameters.
# Read below for details. The value in the comment is the default value.
# config = {
#         Logging parameters
#     "verbose": False,  # print useful information
#     "silenceprojectors": False, # do not print projectors despite verbosity
#     "quiet": False,    # silence all warning messages and progress bars. Overridden by verbose.
#     "direct": False,   # returns tuple (numerator, denominator) rather than the ratio
#
#         Sampling method specification parameters.
#         Calculate inner product exactly, or approximately by averaging samples
#         from a probability distribution? How many samples? Use median of means?
#     "noapprox": False,  # Don't sample, and calculate inner product directly.
#     "samples": 2000,    # Mean of 2000 samples gives mult. error 0.1 with 95% probability.
#     "bins": 1,          # Number of groups of samples over which to take the median.
#         Instead of manually setting samples and bins, can also auto-pick. Changing
#         either of these from their default value will override samples and bins options.
#     "error": 0.1,       # Multiplicative error of inner product.
#     "failprob": 0.05    # Probability of output being worse than that error.
#
#         State preparation parameters: Use |H> or |L>? If |L>, how big?
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
#     "cpath": "libcirc/mpibackend",  # Location of c executable
#     "mpirun": "/usr/bin/mpirun",  # Location of mpirun executable, and any options
#     "procs": None,  # number of processes. If unset python or mpi will pick automatically.
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
def probability(circ, measure, config={}):
    # unpack, configure default values
    verbose = config.get("verbose")
    if verbose: config["quiet"] = False
    quiet = config.get("quiet")
    if config.get("exact") is None: config["exact"] = True
    if config.get("fidbound") is None: config["fidbound"] = 1e-5
    if config.get("cpath") is None: config["cpath"] = "libcirc/mpibackend"
    if config.get("mpirun") is None: config["mpirun"] = "/usr/bin/mpirun"
    if config.get("procs") is 0: config["procs"] = None

    # if k is used for L calculation, then fidelity=True is only for display.
    # thus set fidelity=False if quiet flag is set.
    if not config.get("exact") and config.get("k") is not None:
        if config.get("quiet") and config.get("fidelity"):
            config["fidelity"] = False

    # get projectors
    G, H, n, t = projectors.projectors(circ, measure, verbose=verbose,
                                       x=config.get("x"), y=config.get("y"))

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
    if t == 0:
        if not quiet: print("Clifford circuit. Reverting to python implementation.")
        config["python"] = True

    # verify existence of executable
    if not config.get("python"):
        if config.get("cpath") is None:
            if not quiet: print("C executable unspecified. Reverting to python implementation.")
            config["python"] = True

        elif not os.path.isfile(config.get("cpath")):
            if not quiet: print("Could not find c executable at: " + config.get("cpath"))
            if not quiet: print("Reverting to python implementation")
            config["python"] = True

    # configure sampling
    if not config.get("noapprox"):
        if not config.get("samples"): config["samples"] = 2000
        if not config.get("bins"): config["bins"] = 1
        if not config.get("error"): config["error"] = 0.2
        if not config.get("failprob"): config["failprob"] = 0.05

        if config.get("error") is not 0.2 or config.get("failprob") is not 0.05:
            # auto-pick samples and bins.
            chi = (2**t - 1)/(2**t + 1)
            if config.get("failprob") < 0.0076:  # median of means worth it
                config["samples"] = int(np.ceil(6 * chi * config.get("error")**(-2)))
                config["bins"] = int(np.ceil(4.5 * np.log(1/config.get("failprob"))))
            else:  # just take the mean: L = chi/(p * e^2)
                config["samples"] = int(np.ceil(chi * config.get("error")**(-2) *
                                                config.get("failprob")**(-1)))
                config["bins"] = 1

            if not quiet:
                print("Autopicking median of %d bins with %d samples per bin."
                      % (config["bins"], config["samples"]))
                print("Ensure that the number of parallel cores is less than %d."
                      % config["samples"])

        if verbose:
            print("Evaluating median of %d bins with %d samples per bin." % (config["bins"], config["samples"]))

    # ------------------------------------ Python backend ------------------------------------
    if config.get("python"):
        # calculate |L> ~= |H>
        L, norm = decompose(t, config)

        # Don't need norm if both projectors are nontrivial
        if not (len(Gprime[0]) == 0 or len(Hprime[0]) == 0): norm = 1

        if verbose:
            if L is None: print("Using exact decomposition of |H^t>: 2^%d" % np.ceil(t/2))
            else: print("Stabilizer rank of |L>: 2^%d" % len(L))

        if L is None:
            if config.get("samples")*config.get("bins")*2 > (2**np.ceil(t/2) - 1):
                config["noapprox"] = True
                if verbose: print("More samples than terms in exact calculation. Disabling sampling.")
        else:
            if config.get("samples")*config.get("bins")*2 > (2**len(L) - 1):
                config.get["noapprox"] = True
                if verbose: print("More samples than terms in exact calculation. Disabling sampling.")

        if config.get("noapprox"):
            numerator = exactProjector(Gprime, L, norm, procs=config.get("procs"))
            denominator = exactProjector(Hprime, L, norm, procs=config.get("procs"))
        else:
            numerator = multiSampledProjector(Gprime, L, norm, samples=config.get("samples"),
                                              bins=config.get("bins"), procs=config.get("procs"))
            denominator = multiSampledProjector(Hprime, L, norm, samples=config.get("samples"),
                                                bins=config.get("bins"), procs=config.get("procs"))

    else:
        # --------------------------------------- C backend --------------------------------------

        # --------------- Helpers ------------
        def send(s):
            if s is None: s = 0
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

        # --------------- mpirun command ------------
        if not config.get("file"):
            executable = config["mpirun"].split()
            if config.get("procs") is not None:
                executable += ["-n", str(config["procs"])]

            executable += [config["cpath"], "stdin"]

            p = Popen(executable, stdin=PIPE, stdout=PIPE, stderr=PIPE, bufsize=1)

        # --------------- input data ------------
        indat = b""

        # Logging
        indat += send(1 if config.get("quiet") else 0)  # quiet
        indat += send(1 if config.get("verbose") else 0)  # verbose

        # Sampling method
        indat += send(1 if config.get("noapprox") else 0)  # noapprox
        indat += send(config.get("samples"))  # samples
        indat += send(config.get("bins"))  # bins

        # State prep
        indat += send(len(Gprime[1][0]))  # t
        indat += send(config.get("k"))  # k
        indat += send(1 if config.get("exact") else 0)  # exact
        indat += send(config.get("fidbound") if config.get("fidbound") else 1e-5)  # fidbound
        indat += send(1 if config.get("fidelity") else 0)  # fidelity
        indat += send(1 if config.get("rank") else 0)  # rank

        # Debug
        indat += send(1 if config.get("forceL") else 0)  # forceL

        # Projectors
        indat += writeProjector(Gprime)
        indat += writeProjector(Hprime)

        # --------------- write to file ------------
        if config.get("file"):
            f = open(config.get("file"), "w")
            f.write(indat.decode())
            if not quiet: print("Wrote instruction file to %s." % config.get("file"))

            return None

        # --------------- run the backend ------------
        p.stdin.write(indat)
        p.stdin.flush()

        out = []
        while True:
            p.stdout.flush()
            line = 0
            line = p.stdout.readline().decode()[:-1]

            if not line:
                break

            out.append(line)

            try:
                float(line)
            except:
                print(line)

        success = True
        try:
            numerator = float(out[-2])
            denominator = float(out[-1])
        except:
            # these are errors, not warnings, so they are not silenced by quiet

            print("C code encountered error. Aborting.")
            success = False

        if not success: raise RuntimeError

    # ------------------ end backend-dependent code ------------------

    if verbose:
        print("|| Gprime |H^t> ||^2 ~= " + str(numerator))
        print("|| Hprime |H^t> ||^2 ~= " + str(denominator))

    if config.get("direct"):
        return (2**(v-u) * numerator, denominator)

    if numerator == 0: return 0  # deal with denominator == 0
    prob = 2**(v-u) * (numerator/denominator)
    # if (prob > 1): prob = 1.0  # no probabilities greater than 1
    return prob


# Needs from config dict: exact, k, fidbound, rank, fidelity, forceL, verbose, quiet
def decompose(t, config):
    quiet = config.get("quiet")
    verbose = config.get("verbose")

    # trivial case
    if t == 0:
        return None, 1

    v = np.cos(np.pi/8)

    # exact case
    norm = (2)**(np.floor(t/2)/2)
    if (t % 2 == 1): norm *= (2*v)
    if config.get("exact"): return None, norm

    k = config.get("k")
    forceK = (k is not None)  # Was k selected by the user

    if k is None:
        if config.get("fidbound") is None:
            raise ValueError("Need to specify either k or fidbound, or set exact=True to determine sampling method.")
        # pick unique k such that 1/(2^(k-2)) \geq v^(2t) \delta \geq 1/(2^(k-1))
        k = np.ceil(1 - 2*t*np.log2(v) - np.log2(config.get("fidbound")))
        if verbose: print("Autopicking k = %d to achieve delta = %f." % (k, config.get("fidbound")))
    k = int(k)

    # can achieve k = t/2 by pairs of stabilizer states
    # revert to exact norm
    if k > t/2 and not forceK and not config.get("forceL"):
        if verbose: print("k > t/2. Reverting to exact decomposition.")
        return None, norm

    # prevents infinite loops
    if (k > t):
        if forceK and not quiet: print("Can't have k > t. Setting k to %d." % t)
        k = t

    innerProd = 0
    Z_L = None

    while innerProd < 1-config.get("fidbound") or forceK:

        L = np.random.random_integers(0, 1, (k, t))

        if (config.get("rank")):
            # check rank by computing null space

            # init with identity
            Null = []
            for i in range(t):
                row = np.zeros(t)
                row[i] = 1
                Null.append(row)

            # orthogonalize null space for every row in L
            for i in range(len(L)):
                row = L[i]
                good = []  # inner prod 0
                bad = []  # inner prod 1

                for g in Null:
                    if np.inner(row, g) % 2 == 0: good.append(g)
                    else: bad.append(g)

                # add together rows with inner prod 1
                corrected = []
                for i in range(len(bad) - 1):
                    corrected.append((bad[i] + bad[i+1]) % 2)
                Null = good + corrected

            rank = t - len(Null)
            if (rank < k):
                if not quiet: print("L has insufficient rank. Sampling again...")
                continue

        if config.get("fidelity"):
            # compute Z(L) = sum_x 2^{-|x|/2}
            Z_L = 0
            for i in range(2**k):
                z = np.array(list(np.binary_repr(i, width=k))).astype(int)[::-1]
                x = np.dot(z, L) % 2
                Z_L += 2**(-np.sum(x)/2)

            innerProd = 2**k * v**(2*t) / Z_L
            if forceK:
                # quiet can't be set for this
                print("delta = 1 - <H^t|L>: %f" % 1 - innerProd)
                break
            elif innerProd < 1-config.get("fidbound"):
                if not quiet: print("delta = 1 - <H^t|L>: %f - Not good enough!" % 1 - innerProd)
            else:
                if not quiet: print("delta = 1 - <H^t|L>: %f" % 1 - innerProd)
        else: break

    if config.get("fidelity"):
        norm = np.sqrt(2**k * Z_L)
        return L, norm
    else:
        return L, None


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
