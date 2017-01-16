import numpy as np
import os
import sys
sys.path.append(os.path.abspath("."))

from libcirc.probability import probability

# number of samples
logL = 2.5
L = int(np.ceil(10**logL))
print("Samples:", L)

# probability of failure
logpf = -0.5
pf = 10**logpf
print("Expected success probability:", (1-pf)*100, "%")

# t gates
T = 6
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

# prepare circuit
circ = "H\n"
for i in range(T): circ += "T\nH\n"

# multiplicative error
e = np.sqrt(1/(pf*L))*target

if True:
    print("L sampling")
    fidbound = 1e-3
    config = {"python":python, "fidbound":fidbound , "exact":False,
            "rank":True, "quiet":True, "forceL": True}

    # update error bound, given | <H|L> |^2 > 1 - fidbound
    # equation (25) in paper gives fidbound = (e/5)^2
    # and then guarantees || P - P_out || < e/5
    e += np.sqrt(fidbound)
else:
    print("H sampling")
    config = {"python":python, "exact":True}

print("Error cutoff:", e)

passed = 0
values = []
ntests = int(np.ceil(10**(-logpf+1.5)))
print("Running", ntests,"tests...")
for i in range(ntests):
    if (i%50 == 0): print("Test", i)
    p = probability(circ, {0:0}, samples=L, config=config)
    values.append(p)
    if np.abs(target - p) < e: passed += 1

prob = passed/ntests
print(passed, "passed. Success probability = ", prob*100, "%")

z = 1.96 # 95 % confidence
confshift = z*np.sqrt(prob*(1-prob)/ntests)
print("95% confidence: [" +  str((prob - confshift)*100) + "%, " + str((prob + confshift)*100) + "%]" )

if ((1-pf) < prob-confshift ): print("significantly BETTER")
elif ((1-pf) < prob ): print("OK: Probability as expected (a little high)")
elif ((1-pf) > prob+confshift ): print("FAILED significantly")
else: print("OK: Probability as expected (a little low)")

print("Average error:", np.mean(np.abs(np.array(values) - target )))
print("Drift: ", np.mean(values) - p)
