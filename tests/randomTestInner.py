import numpy as np
import os
import sys
sys.path.append(os.path.abspath("."))

import matplotlib.pyplot as plt

from libcirc.stabilizer.stabilizer import StabilizerState
from libcirc.noapprox import prepH
from multiprocessing import Pool

# todo = [1,2]
# todo = [1,2,3,4,5,6]
todo = [1,2,3,4]

keys = ["success", "confidence", "actoutput", "error", "std", "targeterr"]
data = {}

for key in keys: data[key] = []

# title = "Numerical error suppression"
# title = "Complex addition"
title = "Inner product of |H^t>, without pool"
verbose = False

# number of samples
logL = 2
L = int(np.ceil(10**logL))

# probability of failure
logpf = -1
pf = 10**logpf

# number of tests
ntests = 300
# ntests = int(np.ceil(10**(-logpf+1.5)))  # autopick

title += "\n Samples: " + str(L) + ", Failure probability: " + str(np.round(pf*100,3)) + "%"

# python
# python = True
title += ", Python backend"
# else: title += ", C backend"

print(title)

for T in todo:
    # t gates
    print("T gates:", T)

    # theta = StabilizerState(T, T)
    # theta1 = prepH(0, T)
    # theta2 = prepH(1, T)
    norm = 2**int(np.floor(T/2))
    if T % 2 == 1: norm *= (2*np.cos(np.pi/8))**2

    # multiplicative error
    error = np.sqrt((2**T - 1)/(2**T + 1))/np.sqrt(L*pf)
    data["targeterr"].append(error)

    passed = 0
    values = []
    for i in range(ntests):
        print("Test", i, "/", ntests, end="\r")

        def sample(seed):
            np.random.seed(seed=seed)
            inner = 0
            phi = StabilizerState.randomStabilizerState(T)
            for j in range(2**int(np.ceil(T/2))):
                theta = prepH(j, T)
                inner += StabilizerState.innerProduct(phi, theta)
            return np.abs(inner)**2

        seeds = []
        for j in range(L):
            seeds.append(np.random.random_integers(0, 2**32 - 1))

        if False:
            pool = Pool()
            out = sum(pool.map(sample, seeds))
            pool.close()
            pool.join()
        else:
            out = 0
            for j in seeds:
                out += sample(j)

        out *= 2**T
        out /= norm
        out /= L

        values.append(out)
        if np.abs(out - 1) < error: passed += 1

    prob = passed/ntests

    z = 1.96  # 95 % confidence
    confshift = z*np.sqrt(prob*(1-prob)/ntests)

    data["success"].append(prob)
    data["confidence"].append(confshift)
    data["error"].append(np.abs(np.array(values) - 1))
    data["actoutput"].append(np.array(values))
    print("")

ax = plt.subplots(3,1)[1].T

dat = data

ax[0].errorbar(todo, dat["success"], yerr=dat["confidence"], label="Actual fraction")
ax[0].plot(todo, [1-pf]*len(todo), label="Target fraction")
ax[0].set_title("Fraction successful tests")
ax[0].set_ylabel("Successful fraction")
ax[0].grid(True)
ax[0].set_xticks(todo)
ax[0].set_xlim(todo[0]-1, todo[-1]+1)
ax[0].set_ylim(0, 1.1)
ax[0].legend(loc=0)

ax[1].boxplot(dat["error"])
ax[1].plot(np.array(range(len(todo)))+1, dat["targeterr"], color="g", label="Target error")
ax[1].set_xticks(np.array(range(len(todo)))+1, todo)
ax[1].set_title("Error sizes")
ax[1].set_ylabel("Error")
ax[1].grid(True)

ax[2].boxplot(dat["actoutput"])
ax[2].plot(np.array(range(len(todo)))+1, [1]*len(todo), label="$|\psi|^2$")
ax[2].set_xticks(np.array(range(len(todo)))+1, todo)
ax[2].set_ylabel("Probability")
ax[2].set_xlabel("T gates")
ax[2].set_title("Output")
ax[2].grid(True)
ax[2].legend(loc=0)

plt.suptitle(title)
plt.show()
