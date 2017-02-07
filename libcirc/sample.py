#
# file: sample.py
# Programmatical interface for sampling algorithm.
#

import numpy as np
from libcirc.probability import probability


# sample output from a subset of the qubits
# arguments are same as in probability
# sample is a list of integer indexes, with 0 being the first qubit.
#
# if config["exact"] is unspecified, the default is now False
# also config["file"] is disabled
def sampleQubits(circ, measure, sample, config={}):
    # unpack config
    verbose = config.get("verbose")
    if config.get("exact") is None: config["exact"] = False
    config["file"] = None

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
                print("-> Sampled: " + str(qubit))
                print("-> Sample so far: " + qubitssofar + str(qubit) + "\n")

            measure[qubits[-1]] = qubit

            return qubitssofar + str(qubit), measure, (P0 if qubit == 0 else (1 - P0))

    output, _, _ = recursiveSample(sample, measure)
    return output
