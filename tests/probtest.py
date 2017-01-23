import numpy as np
import os
import sys
sys.path.append(os.path.abspath("."))

import matplotlib.pyplot as plt

from libcirc.probability import probability

# todo = [1,2,3,4,5,6,7]
# todo = [1,4,7]
todo = [1,2,3,4]
success = []
confidence = []

actoutput = []
expoutput = []

error = []
std = []
targeterr = []


for T in todo:
    # number of samples
    logL = 3
    L = int(np.ceil(10**logL))
    print("Samples:", L)

    # probability of failure
    logpf = -0.5
    pf = 10**logpf
    print("Expected success probability:", (1-pf)*100, "%")

    # t gates
    # T = 6
    print("T gates:", T)

    # python
    python = True
    if python: print("Python backend")
    else: print("C backend")

    # calculate expected output
    Tgate = np.array([[1+0j, 0+0j], [0+0j, np.exp(1j*np.pi/4)]])
    Hgate = np.array([[1+0j, 1+0j], [1+0j, -1+0j]])/np.sqrt(2)

    State = np.dot(Hgate, np.array([1+0j, 0+0j]))
    for i in range(T):
        State = np.dot(Tgate, State)
        State = np.dot(Hgate, State)

    target = np.abs(State[0])**2
    print("Target output:", target)
    expoutput.append(target)

    # prepare circuit
    circ = "H\n"
    for i in range(T): circ += "T\nH\n"

    # multiplicative error
    e = np.sqrt((2**T - 1)/(2**T + 1))*target/np.sqrt(L*pf)
    targeterr.append(e)

    if False:
        print("L sampling")
        fidbound = 1e-3
        config = {"python":python, "fidbound":fidbound, "exact":False,
                  "rank":True, "quiet":True, "forceL": True}

        # update error bound, given | <H|L> |^2 > 1 - fidbound
        # equation (25) in paper gives fidbound = (e/5)^2
        # and then guarantees || P - P_out || < e/5
        e += np.sqrt(fidbound)
    else:
        print("H sampling")
        config = {"python":python, "exact":True}

    print("Error cutoff:", e)

    config["procs"] = 8

    passed = 0
    values = []
    ntests = int(np.ceil(10**(-logpf+1.5)))
    print("Running", ntests,"tests...")
    for i in range(ntests):
        print("Test", i, "/", ntests, end="\r")
        p = probability(circ, {0:0}, samples=L, config=config)
        values.append(p)
        if np.abs(target - p) < e: passed += 1

    prob = passed/ntests
    print(passed, "passed. Success probability = ", prob*100, "%")

    z = 1.96  # 95 % confidence
    confshift = z*np.sqrt(prob*(1-prob)/ntests)
    print("95% confidence: [" + str((prob - confshift)*100) + "%, " + str((prob + confshift)*100) + "%]")

    if ((1-pf) < prob-confshift): print("significantly BETTER")
    elif ((1-pf) < prob): print("OK: Probability as expected (a little high)")
    elif ((1-pf) > prob+confshift): print("FAILED significantly")
    else: print("OK: Probability as expected (a little low)")

    success.append(prob)
    confidence.append(confshift)

    error.append(np.abs(np.array(values) - target))
    actoutput.append(np.array(values))
    print("Average error:", np.mean(np.abs(np.array(values) - target)))
    print("Drift: ", np.mean(values) - p)
    print("")

f, (ax1, ax2, ax3) = plt.subplots(3,1)

ax1.errorbar(todo, success, yerr=confidence, label="Estimate")
ax1.plot(todo, [1-pf]*len(todo), label="Target")
ax1.set_title("Fraction successful tests")
ax1.set_ylabel("Successful fraction")
ax1.grid(True)
ax1.set_xticks(todo)
ax1.set_xlim(todo[0]-1, todo[-1]+1)
ax1.set_ylim(0, 1)
ax1.legend()

ax2.boxplot(error)
ax2.plot(np.array(range(len(todo)))+1, targeterr, color="g")
ax2.set_xticks(np.array(range(len(todo)))+1, todo)
ax2.set_title("Error sizes")
ax2.set_ylabel("Error")
ax2.grid(True)


ax3.boxplot(actoutput)
ax3.plot(np.array(range(len(todo)))+1, expoutput, label="$|\psi|^2$")
ax3.set_xticks(np.array(range(len(todo)))+1, todo)
ax3.set_ylabel("Probability")
ax3.set_title("Output")
ax3.grid(True)
ax3.legend()

plt.show()
