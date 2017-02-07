#
# file: stateprep.py
# Prepare stabilizer components of states |H^t> and |L>
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


def prepL(i, t, L):
    # compute bitstring by adding rows of l
    Lbits = list(np.binary_repr(i, width=len(L)))
    bitstring = np.zeros(t)
    for idx in range(len(Lbits)):
        if Lbits[idx] == '1':
            bitstring += L[idx]
    bitstring = bitstring.astype(int) % 2

    # Stabilizer state is product of |0> and |+>
    # 1's in bitstring indicate positions of |+>

    # initialize stabilizer state
    phi = StabilizerState(t, t)

    # construct state using shrink
    for xtildeidx in range(t):
        if bitstring[xtildeidx] == 0:
            vec = np.zeros(t)
            vec[xtildeidx] = 1
            phi.shrink(vec, 0)
            # |0> at index, inner prod with 1 is 0
        # |+> at index -> do nothing

    return phi
