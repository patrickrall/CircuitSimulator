import numpy as np
import os
import sys
sys.path.append(os.path.abspath("."))

import matplotlib.pyplot as plt

from libcirc.probability import probability

# todo = [1,2,3,4,5,6,7]
# todo = [1,2,3]
todo = [6,7]

keys = ["success", "confidence", "output", "error", "std", "targeterr"]
data = {"num":{}, "denom":{}}

for key in keys:
    data["num"][key] = []
    data["denom"][key] = []

# title = "Numerical error suppression"
title = "Cheese random 2"
verbose = False

# number of samples
logL = 3
L = int(np.ceil(10**logL))

# probability of failure
logpf = -1
pf = 10**logpf

# number of tests
ntests = 1
# ntests = int(np.ceil(10**(-logpf+1.5)))  # autopick

title += "\n Samples: " + str(L) + ", Failure probability: " + str(np.round(pf*100,3)) + "%"
# python
python = False
if python: title += ", Python backend"
else: title += ", C backend"

print(title)

for T in todo:
    # t gates
    print("T gates:", T)

    # calculate expected output
    Tgate = np.array([[1+0j, 0+0j], [0+0j, np.exp(1j*np.pi/4)]])
    Hgate = np.array([[1+0j, 1+0j], [1+0j, -1+0j]])/np.sqrt(2)

    # State = np.dot(Hgate, np.array([1+0j, 0+0j]))
    # for i in range(T):
    #     State = np.dot(Tgate, State)
    #     State = np.dot(Hgate, State)
    # target = np.abs(State[0])**2

    # prepare circuit
    circ = "H\n"
    for i in range(T): circ += "T\nH\n"

    (tnum, tdenom) = probability(circ, {0:0}, config={"python": True, "quiet":True, "noapprox":True, "direct":True})
    tnum = L*tnum
    tdenom = L*tdenom

    if verbose: print("Target outputs:", tnum, tdenom)

    # multiplicative error
    enum = np.sqrt((2**T - 1)/(2**T + 1))*tnum/np.sqrt(L*pf)
    edenom = np.sqrt((2**T - 1)/(2**T + 1))*tdenom/np.sqrt(L*pf)
    data["num"]["targeterr"].append(enum)
    data["denom"]["targeterr"].append(edenom)

    if False:
        if verbose: print("L sampling")
        fidbound = 1e-3
        config = {"python":python, "fidbound":fidbound, "exact":False,
                  "rank":True, "quiet":True, "forceL": True, "direct":True}

        # update error bound, given | <H|L> |^2 > 1 - fidbound
        # equation (25) in paper gives fidbound = (e/5)^2
        # and then guarantees || P - P_out || < e/5
        enum += np.sqrt(fidbound)
        edenom += np.sqrt(fidbound)
    else:
        if verbose: print("H sampling")
        config = {"python":python, "exact":True, "direct":True}

    config["mpirun"] = "/usr/bin/mpirun --hostfile /home/prall/.mpi_hostfile"
    config["procs"] = 16

    numpassed = 0
    denompassed = 0
    numvalues = []
    denomvalues = []
    if verbose: print("Running", ntests,"tests...")
    for i in range(ntests):
        print("Test", i+1, "/", ntests, end="\r")
        (num, denom) = probability(circ, {0:0}, samples=L, config=config)
        numvalues.append(num)
        denomvalues.append(denom)
        if np.abs(tnum - num) < enum: numpassed += 1
        if np.abs(tdenom - denom) < edenom: denompassed += 1

    numprob = numpassed/ntests
    denomprob = denompassed/ntests

    print("Passed:", np.round(100*numprob,1), "%    ")

    z = 1.96  # 95 % confidence
    numconfshift = z*np.sqrt(numprob*(1-numprob)/ntests)
    denomconfshift = z*np.sqrt(denomprob*(1-denomprob)/ntests)

    data["num"]["success"].append(numprob)
    data["denom"]["success"].append(denomprob)
    data["num"]["confidence"].append(numconfshift)
    data["denom"]["confidence"].append(denomconfshift)

    data["num"]["error"].append(np.abs(np.array(numvalues) - tnum))
    data["denom"]["error"].append(np.abs(np.array(denomvalues) - tdenom))

    data["num"]["output"].append(np.array(numvalues/tnum))
    data["denom"]["output"].append(np.array(denomvalues/tdenom))
    print("")

axes = plt.subplots(3,2)[1].T

for i in range(2):
    titleappend = ["Numerator: ", "Denominator: "][i]
    dat = data[["num","denom"][i]]
    ax = axes[i]

    ax[0].errorbar(todo, dat["success"], yerr=dat["confidence"], label="Actual fraction")
    ax[0].plot(todo, [1-pf]*len(todo), label="Target fraction")
    ax[0].set_title(titleappend+"Fraction successful tests")
    ax[0].set_ylabel("Successful fraction")
    ax[0].grid(True)
    ax[0].set_xticks(todo)
    ax[0].set_xlim(todo[0]-1, todo[-1]+1)
    ax[0].set_ylim(0, 1.1)
    ax[0].legend(loc=0)

    ax[1].boxplot(dat["error"])
    ax[1].plot(np.array(range(len(todo)))+1, dat["targeterr"], color="g", label="Target error")
    ax[1].set_xticks(np.array(range(len(todo)))+1, todo)
    ax[1].set_title(titleappend + "Error sizes")
    ax[1].set_ylabel("Error")
    ax[1].grid(True)

    ax[2].boxplot(dat["output"])
    ax[2].plot(np.array(range(len(todo)))+1, len(todo)*[1], label="$|\psi|^2$")
    ax[2].set_xticks(np.array(range(len(todo)))+1, todo)
    ax[2].set_ylabel("Probability")
    ax[2].set_xlabel("T gates")
    ax[2].set_title(titleappend + "Output")
    ax[2].grid(True)
    ax[2].legend(loc=0)

plt.suptitle(title)
plt.show()
