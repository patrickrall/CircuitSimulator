########################################################
# Tests for the python stabilizer state implementation #
########################################################

#
# File: tests.py
# Tests the stabilizer state code, also potentially the simulation itself
#

import numpy as np
from stabilizer import StabilizerState
import matplotlib.pyplot as plt

# ---------------- Helper functions -------------


def decToBitstring(dec, n):
    result = np.array([])
    repres = np.binary_repr(dec)
    if len(repres) > n:
        raise ValueError("Decimal number does not fit into "+str(n)+" bytes")

    while n > len(repres):
        result = np.append(result, 0)
        n -= 1

    for b in repres:
        if b == '1':
            result = np.append(result, 1)
        else:
            result = np.append(result, 0)

    return result

# ----------------- Main -----------------------

n = 8
nsamples = int(1e3)

samples = []

dsamples = {}
for d in range(1, n+1):
    dsamples[d] = []

# ---- Test 3: Some sample stabilizer states

if True:
    def bitsToStr(bits):
        string = ""
        for bit in bits:
            string = string + str(int(bit))
        return string

    for n in range(2, 5):
        print("\n\nStates with n =", n)

        for k in range(5):
            print("\nState %d:" % (k+1))
            state = StabilizerState.randomStabilizerState(n)

            norm = 0
            for i in range(2**n):
                x = decToBitstring(i, n)
                coeff = state.coeff(x)
                norm += np.abs(coeff)**2
                if round(np.abs(coeff)**2, 3) > 0:
                    coeff = round(coeff, 3)
                    print(bitsToStr(x), coeff)
            print("Norm: "+str(norm))


# ---- Test 2: Distribution of k or d versus probability

if False:
    fig, ax = plt.subplots()
    ax.set_title("Distribution of $d,n-k$ in a sample of $%0.0E$ stab. states with $n = %d$" % (nsamples, n))
    ax.set_ylabel("Frequency")
    ax.set_xlabel("$n-k,d$")

    ddist = []
    kdist = []
    for d in range(1, n+1):
        ddist.append(len(dsamples[d]))
        kdist = kdist + dsamples[d]

    width = 0.25

    khist, bins = np.histogram(kdist, bins=np.arange(0, n+1))

    ax.bar(np.arange(0, n), ddist, width, label=("Sampled $b$"), fill=True)
    ax.bar(np.arange(0, n)+width, nsamples*StabilizerState.dDists[n], width, color='y', label="Probable $d$", fill=True)
    ax.bar(np.arange(0, n)+2*width, khist, width, label=("Sampled $n-k$"), fill=True, color='g')

    ax.legend()
    plt.show()

# ---- Test 1: Distribution of k, d

if False:
    plt.clf()
    plt.title("Distribution of $n-k$ in a sample of $%0.0E$ stab. states with $n = %d$" % (nsamples, n))
    plt.ylabel("Frequency")
    plt.xlabel("$n-k$")

    for d in range(1, n+1):
        if len(dsamples[d]) == 0:
            continue
        plt.hist(dsamples[d], bins=range(n+1), label="$d = "+str(d)+"$", histtype="step", fill=True)

    plt.legend()
    plt.show()
