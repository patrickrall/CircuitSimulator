from stabilizer.stabilizer import StabilizerState
import numpy as np
import json
from copy import deepcopy

phases = [3.0,3.0,2.0]
xs = [np.array([1.0,0.0]), np.array([0.0,1.0]), np.array([1.0,1.0])]
zs = [np.array([1.0,1.0]), np.array([1.0,1.0]), np.array([0.0,0.0])]
# phases = [2.0]
# xs = [np.array([1.0, 1.0])]
# zs = [np.array([0.0, 0.0])]

# np.random.seed(23)
theta = StabilizerState.randomStabilizerState(2)
# theta.D = np.zeros(theta.k)
# theta.J = np.zeros((theta.k, theta.k))
theta2 = deepcopy(theta)

oldpacked = theta.unpack()
unpacked = [0.0, 0.0, 0.0, 0.0]

# import pdb; pdb.set_trace()
def printState(x):
    outstring = "["
    for comp in x:
        outstring += " %+0.3f + %+0.3fi " % (comp.real, comp.imag)
    outstring += "]"
    print(outstring)

printState(oldpacked)

count = 0
actions = []
# while not np.allclose(oldpacked, unpacked):
for i in range(3):
    # if (index == 0): print("Difference: ", (np.array(oldpacked) - np.array(unpacked)))
    oldpacked = unpacked
    # print("\nPrevious: ", oldpacked)
    res, status = theta.measurePauli(phases[i], zs[i], xs[i], give_status=True)
    unpacked = theta.unpack()



    printState(unpacked)

    # index = (index + 1) % 3
    count += 1
    if (len(actions) == 0 or actions[-1][0] != status):
        actions.append((status, 1))
    else:
        actions[-1] = (actions[-1][0], actions[-1][1] + 1)

    if count > 100:
        break

    # print("Index:", index)

if (count > 2):
    print("Path", actions)
    print("n", theta2.n)
    print("k", theta2.k)
    print("G", json.dumps(theta2.G.tolist()))
    print("Gbar", json.dumps(theta2.Gbar.tolist()))
    print("Q", theta2.Q)
    print("D", json.dumps(theta2.D.tolist()))
    print("J", json.dumps(theta2.J.tolist()))
