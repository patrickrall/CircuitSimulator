#
# file: main.py
# Command line interface.
#

import sys
import re
from datetime import datetime
import libcirc.compile.compilecirc as compilecirc
from libcirc.probability import probability
from libcirc.sample import sampleQubits


def main(argv):
    if len(argv) > 1 and argv[1] == "-h":
        return help()

    if len(argv) == 1:
        return usage("Wrong number of arguments.")

    # built-in test script
    HTstack = -1
    if argv[1][:3] == "HT=":
        HTstack = int(argv[1][3:])

    if len(argv) < 3 and HTstack == -1:
        return usage("Wrong number of arguments.")

    config = {}

    mpiarg = False

    # parse optional arguments
    for i in range(3 if HTstack == -1 else 2, len(argv)):
        if mpiarg:
            if argv[i][-1] == '"':
                mpiarg = False
                config["mpirun"] += " " + argv[i][:-1]
            else: config["mpirun"] += " " + argv[i]

        # HT stack error message
        elif argv[i][:3] == "HT=": return usage("HT=? option ignored. Put instead of input file.")

        # Logging
        elif argv[i] == "-v": config["verbose"] = True
        elif argv[i] == "-sp": config["silenceprojectors"] = True
        elif argv[i] == "-quiet": config["quiet"] = True

        # Sampling
        elif argv[i] == "-nosampling": config["noapprox"] = True
        elif argv[i][:8] == "samples=": config["samples"] = int(float(argv[i][8:]))
        elif argv[i][:5] == "bins=": config["bins"] = int(float(argv[i][5:]))
        elif argv[i][:6] == "error=": config["error"] = float(argv[i][6:])
        elif argv[i][:9] == "failprob=": config["failprob"] = float(argv[i][9:])

        # State preparation
        elif argv[i] == "-exactstate": config["exact"] = True
        elif argv[i][:2] == "k=": config["k"] = int(float(argv[i][2:]))
        elif argv[i][:9] == "fidbound=": config["fidbound"] = float(argv[i][9:])
        elif argv[i] == "-fidelity": config["fidelity"] = True
        elif argv[i] == "-rank": config["rank"] = True

        # Backend
        elif argv[i] == "-py": config["python"] = True
        elif argv[i][:6] == "cpath=": config["cpath"] = argv[i][6:]
        elif argv[i][:8] == 'mpirun="':
            if argv[i][-1] == '"':
                config["mpirun"] = argv[i][8:-1]
            else:
                mpiarg = True
                config["mpirun"] = argv[i][8:]
        elif argv[i][:6] == "procs=": config["procs"] = int(argv[i][6:])
        elif argv[i][:5] == "file=": config["file"] = argv[i][5:]

        # Debug
        elif argv[i][:2] == "y=": config["y"] = argv[i][2:]
        elif argv[i][:2] == "x=": config["x"] = argv[i][2:]
        elif argv[i] == "-forceL": config["forceL"] = True
        elif i == 2: return usage("HT=? option replaces file and measurement argument.")
        else: return usage("Invalid argument: " + argv[i])

    if config.get("verbose"):
        if config.get("quiet"):
            config["quiet"] = False
            print("Warning: -v option overrides -quiet.")

    quiet = config.get("quiet")

    if config.get("error") is not None or config.get("failprob") is not None:
        if config.get("samples") is not None:
            config["samples"] = None
            if not quiet: print("Warning: error and failprob options override samples option.")
        if config.get("bins") is not None:
            config["bins"] = None
            if not quiet: print("Warning: error and failprob options override bins option.")

    if config.get("exact"):
        if config.get("k") is not None:
            config["k"] = None
            if not quiet: print("Warning: -exact option overrides k option.")
        if config.get("fidbound") is not None:
            config["fidbound"] = None
            if not quiet: print("Warning: -exact option overrides fidbound option.")

    if config.get("k") is not None:
        config["exact"] = False
        if config.get("fidbound") is not None:
            if not quiet: print("Warning: k option overrides fidbound option.")
            config["fidbound"] = None

    if config.get("fidbound") is not None:
        config["exact"] = False

    if config.get("forceL") is not None:
        config["exact"] = False

    if HTstack == -1:
        if not re.match("^(1|0|M|_)*$", argv[2]):
            return usage("Measurement string must consists only of '1', '0', 'M' or '_'.")

        # load input circuit
        infile = argv[1]
        circ = compilecirc.compileCircuit(fname=infile)

        # get measurements
        measure = {}
        sample = []
        for i in range(len(argv[2])):
            if list(argv[2])[i] == '0':
                measure[i] = 0
            if list(argv[2])[i] == '1':
                measure[i] = 1
            if list(argv[2])[i] == 'M':
                sample.append(i)
    else:
        circ = ""
        for i in range(HTstack):
            circ += "H\n"
            circ += "T\n"
        circ += "H\n"

        measure = {0: 0}
        sample = []

        if config.get("y") is not None:
            if not quiet: print("Warning: override y to all zero string.")

        config["y"] = "0"*HTstack

    # start timer
    if config.get("verbose"): starttime = datetime.now()

    if len(sample) == 0:  # algorithm 1: compute probability
        if config.get("verbose"): print("Probability mode: calculate probability of measurement outcome")

        if config.get("exact") is not None and config.get("exact") is False:
            if not quiet: print("Warning: Probability calculation requires exact sampling rather than L sampling.\nWill use |L> state anyway, as requested.")

        P = probability(circ, measure, config=config)
        if P is None:
            print("Program gave no output probability")
        else:
            print("Probability: " + str(P))

    else:  # algorithm 2: sample bits
        if config.get("verbose"): print("Sample mode: sample marked bits")

        X = sampleQubits(circ, measure, sample, config=config)
        if len(X) == 0:  # no sample possible
            print("Error: circuit output state cannot produce a measurement with these constraints.")
        else:
            print("Sample: " + str(X))

    if config.get("verbose"):
        print("Time elapsed: " + str(datetime.now() - starttime))

    # calculate reference value for HT stack
    if HTstack != -1 and P is not None:
        import numpy as np
        H = np.array([[1,1],[1,-1]])/np.sqrt(2)
        T = np.array([[1,0],[0,np.exp(1j*np.pi/4)]])

        out = H
        for i in range(HTstack):
            out = np.dot(out,T)
            out = np.dot(out,H)

        Pref = np.abs(out[0,0])**2
        print("HT stack reference value: " + str(Pref))
        print("Error: " + str(np.abs(P - Pref)))


def usage(error):
    print(error + """\n\nStabilizer simulator usage:
main.py <circuitfile> <measurement>

Full help statement: main.py -h""")


def help():
    f = open("helptext.txt")
    print(f.read())


if __name__ == "__main__":
    main(sys.argv)
