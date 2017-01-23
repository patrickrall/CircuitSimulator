import numpy as np
import os
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath("."))

from libcirc.stabilizer.stabilizer import StabilizerState
from libcirc.noapprox import prepH


fix, ax = plt.subplots()

todo = [1,2,3,4,5]
colors = "rgbcmy"

for n in todo:
    d = 2**n
    samples = int(1e4)

    # psi = StabilizerState.randomStabilizerState(n)
    psi = StabilizerState(n,n)

    if n%2 == 1:
        norm = 2*np.cos(np.pi/8)
    else:
        norm = 2
    # norm = 2**int(np.ceil(n/2))
    # if (n % 2 == 1): norm *= 2*np.cos(np.pi/8)

    kinds = {}
    outputs = []
    for i in range(samples):
        theta = StabilizerState.randomStabilizerState(n)

        out = 0
        # for j in range(2**int(np.ceil(n/2))):
        for j in [0,1]:
            psi = prepH(j, n)
            out += StabilizerState.innerProduct(theta, psi)

        out = 2**(n) * np.abs(out)**2/norm

        outputs.append(out)
        out = np.round(out, 5)
        if kinds.get(out) is None: kinds[out] = 0
        kinds[out] += 1

    mu = np.mean(outputs)
    sigma = np.std(outputs)
    print("\nn =", n)
    print("mean =", mu, ", error = ", np.abs(mu-1))
    print("std =", sigma, ", error = ", sigma - np.sqrt((2**n - 1)/(2**n + 1)))

    hist = []
    for out in sorted(kinds.keys()):
        # print(out, kinds[out])
        hist.append(kinds[out]/samples)

    ax.plot(range(len(hist)), hist, colors[todo.index(n)])
    ax.fill_between(range(len(hist)), 0, hist, facecolor=colors[todo.index(n)], alpha=0.5)


plt.show()
