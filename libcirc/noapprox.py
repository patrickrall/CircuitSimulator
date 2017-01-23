#
# file: noapprox.py
# Calaculate || \Pi |H^(\otimes t)> exactly
#

import numpy as np
from libcirc.stabilizer.stabilizer import StabilizerState


def prepH(i, t):
    size = int(np.ceil(t/2))
    odd = t % 2 == 1

    bits = list(np.binary_repr(i, width=size))

    # initialize stabilizer state
    phi = StabilizerState(t, t)

    # set J matrix
    for idx in range(size):
        bit = int(bits[idx])
        if bit == 0 and not (odd and idx == size-1):
            # phi.J = np.array([[0, 4], [4, 0]])
            phi.J[idx*2+1, idx*2] = 4
            phi.J[idx*2, idx*2+1] = 4

    # truncate affine space using shrink
    for idx in range(size):
        bit = int(bits[idx])
        vec = np.zeros(t)

        if odd and idx == size-1:
            # bit = 0 is |+>
            # bit = 1 is |0>
            if bit == 1:
                vec[t-1] = 1
                phi.shrink(vec, 0)
            continue

        # bit = 1 corresponds to |00> + |11> state
        # bit = 0 corresponds to |00> + |01> + |10> - |11>

        if bit == 1:
            vec[idx*2] = 1
            vec[idx*2+1] = 1
            phi.shrink(vec, 0)  # only 00 and 11 have inner prod 0 with 11

    return phi


def calcExactProjector(P, Lnorm):
    (phases, xs, zs) = P

    # empty projector
    if len(phases) == 0:
        return np.abs(Lnorm)**2

    t = len(xs[0])

    # clifford circuit
    if t == 0:
        lookup = {0: 1, 2: -1}
        generators = [1]  # include identity
        for phase in phases: generators.append(lookup[phase])

        # calculate sum of all permutations of generators
        return sum(generators)/len(generators)

    size = int(np.ceil(t/2))
    total = 0
    for i in range(0, 2**size):
        theta = prepH(i, t)
        projfactor = 1
        for g in range(len(phases)):
            res = theta.measurePauli(phases[g], zs[g], xs[g])
            projfactor *= res

            if res == 0: break  # theta annihilated by P
        if projfactor == 0: continue

        for j in range(0, 2**size):
            phi = prepH(j, t)
            inner = StabilizerState.innerProduct(theta, phi)
            total += inner * projfactor

    return np.abs(total)
