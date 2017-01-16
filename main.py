#
# file: main.py
# Command line interface.
#

import sys
import re
from datetime import datetime
import libcirc.compilecirc as compilecirc
from libcirc.probability import probability, sampleQubits


def main(argv):
    if len(argv) > 1 and argv[1] == "-h":
        return help()

    if len(argv) < 3:
        return usage("Wrong number of arguments.")

    samples = 1e4
    config = {}

    # parse optional arguments
    for i in range(3, len(argv)):

        if argv[i] == "-v": config["verbose"] = True
        elif argv[i] == "-sp": config["silenceprojectors"] = True
        elif argv[i] == "-quiet": config["quiet"] = True
        elif argv[i] == "-np": config["parallel"] = False
        elif argv[i][:8] == "samples=": samples = int(float(argv[i][8:]))
        elif argv[i][:9] == "fidbound=": config["fidbound"] = float(argv[i][9:])
        elif argv[i][:2] == "k=": config["k"] = int(float(argv[i][2:]))
        elif argv[i] == "-exact": config["exact"] = True
        elif argv[i][:2] == "y=": config["y"] = argv[i][2:]
        elif argv[i][:2] == "x=": config["x"] = argv[i][2:]
        elif argv[i] == "-forceL": config["forceL"] = True
        elif argv[i] == "-rank": config["rank"] = True
        elif argv[i] == "-fidelity": config["fidelity"] = True
        elif argv[i] == "-py": config["python"] = True
        elif argv[i][:6] == "cpath=": config["cpath"] = argv[i][6:]
        else: raise ValueError("Invalid argument: " + argv[i])

    if config.get("verbose"):
        if config.get("quiet"):
            config["quiet"] = False
            print("Warning: -v option overrides -quiet.")

    quiet = config.get("quiet")

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

    # start timer
    if config.get("verbose"): starttime = datetime.now()

    if len(sample) == 0:  # algorithm 1: compute probability
        if config.get("verbose"): print("Probability mode: calculate probability of measurement outcome")

        if config.get("exact") is not None and config.get("exact") is False:
            if not quiet: print("Warning: Probability calculation requires exact sampling rather than L sampling.\nWill use L sampling anyway, as requested.")

        P = probability(circ, measure, samples=samples, config=config)
        print("Probability: " + str(P))

    else:  # algorithm 2: sample bits
        if config.get("verbose"): print("Sample mode: sample marked bits")

        X = sampleQubits(circ, measure, sample, samples=samples, config=config)
        if len(X) == 0:  # no sample possible
            print("Error: circuit output state cannot produce a measurement with these constraints.")
        else:
            print("Sample: " + str(X))

    if config.get("verbose"):
        print("Time elapsed: " + str(datetime.now() - starttime))


def usage(error):
    print(error + """\n\nStabilizer simulator usage:
main.py <circuitfile> <measurement>

Full help statement: main.py -h""")


def help():
    f = open("helptext.txt")
    print(f.read())


if __name__ == "__main__":
    main(sys.argv)
