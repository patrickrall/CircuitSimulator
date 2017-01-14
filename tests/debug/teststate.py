from libcirc.stabilizer.stabilizer import StabilizerState
from libcirc.probability import printProjector
from libcirc.sample import evalHcomponent
import numpy as np

def testState(theta, P):
    printProjector(P)

    (phases, xs, zs) = P
    t = len(xs[0])

    # project random state to P
    projfactor = 1
    for g in range(len(phases)):
        res = theta.measurePauli(phases[g], zs[g], xs[g])
        projfactor *= res
        if res == 0: return 0  # theta annihilated by P

    func = evalHcomponent
    size = int(np.ceil(t/2))  # need one bit for each pair, plus one more if t is odd

    total = sum(map(func, [(i, None, theta, t) for i in range(0, 2**size)]))

    out = 2**t * np.abs(projfactor*total)**2
    return out


P = ([1], [np.array([ 1.,  1.])], [np.array([ 1.,  0.])])

state = StabilizerState(2, 1)
state.n = 2
state.k = 2
state.h = np.array([0,0])
state.G = np.array([[1,0],[0,1]])
state.Gbar = np.array([[1,0],[0,1]])
state.Q = 2
state.D = np.array([0,2])
state.J = np.array([[0,0],[0,4]])




testState(state, P)
