#
# file: sample.py
# construct a stabilizer decomposition of |H^(\otimes t)>
# Sample from || \Pi |H^(\otimes t)>
#

import numpy as np
from stabilizer.stabilizer import StabilizerState
from multiprocessing import Pool


def decompose(t, delta, exact=False):
    # trivial case
    if t == 0:
        return np.array([[]]), 1

    v = np.cos(np.pi/8)

    # find k such that  4 \geq 2^k v^(2t) delta \geq 2
    v2tdelta = v**(2*t) * delta
    k = 0
    k2v2tdelta = v2tdelta
    while k2v2tdelta < 2:
        k += 1
        k2v2tdelta *= 2

    if k2v2tdelta > 4:
        raise ValueError("No valid k found for t = %d, delta = %f" % (t, delta))

    # can achieve k = t/2 by pairs of stabilizer states
    if k > t/2 or exact:
        norm = 2**np.floor(t/2)
        if (t % 2 == 1): norm *= (2*v)
        print("norm", norm)
        return None, norm

    # this is redundant bc. previous statement, but
    # prevents infinite loops when debugging
    if (k > t): k = t

    innerProd = 0

    count = 0
    while innerProd < 1-delta:
        count += 1

        L = np.random.random_integers(0, 1, (k, t))

        # check rank
        if (np.linalg.matrix_rank(L) < k): continue

        # compute Z(L) = sum_x 2^{-|x|/2}
        Z_L = 0
        for i in range(2**k):
            z = np.array(list(np.binary_repr(i, width=k))).astype(int)[::-1]
            x = np.dot(z, L) % 2
            Z_L += 2**(-np.sum(x)/2)

        innerProd = 2**k * v**(2*t) / Z_L

    norm = np.sqrt(2**k * Z_L)

    return L, norm


# Inner product for some state in |L> ~= |H^t>
def evalLcomponent(args):
    (i, L, theta, t, seed) = args  # unpack arguments (easier for parallel code)

    # np.random.seed((seed * 2**len(L) + i) % 4294967296)

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

    # construct state by measuring paulis
    for xtildeidx in range(t):
        vec = np.zeros(t)
        vec[xtildeidx] = 1
        if bitstring[xtildeidx] == 1:
            # |+> at index, so measure X
            phi.measurePauli(0, np.zeros(t), vec)
        else:
            # |0> at index, so measure Z
            phi.measurePauli(0, vec, np.zeros(t))

    return StabilizerState.innerProduct(theta, phi)


# Inner product for some state in |H^t> using pairwise decomposition
def evalHcomponent(args):
    (i, L, theta, t, seed) = args  # unpack arguments (easier for parallel code)

    # import pdb; pdb.set_trace()
    size = int(np.ceil(t/2))
    odd = t % 2 == 1

    bits = list(np.binary_repr(i, width=size))

    # initialize stabilizer state
    phi = StabilizerState(t, t)

    phase = 0
    for idx in range(size):
        bit = int(bits[idx])
        vec = np.zeros(t)
        if odd and idx == size-1:
            vec[idx] = 1

            # last qubit: |H> = (1/2v)(|0> + |+>)
            if bit == 0:
                phi.measurePauli(0, vec, np.zeros(t))  # |0>, measure Z
            else:
                phi.measurePauli(0, np.zeros(t), vec)  # |+>, measure X

        else:
            vec[idx*2+1] = 1
            vec[idx*2] = 1

            # qubit pair: |A^2> = (1/2)(|00> + i|11>) + (e^(i\pi/4) / 2)(|01> + |10>)
            # Stabilized by ZZ and i(XZ)X. Act by conjugation with H*H, then S*S^\dagger
            # to obtain -(XZ)(XZ) and XZ
            if bit == 0:
                # phi.Q = 1
                # phi.J = np.array([[0, 4], [4, 0]])

                # |00> + i|11>
                phi.k = 1
                phi.G = np.array([[1, 1], [1, 0]])
                phi.Gbar = np.array([[0, 1], [1, 1]])
                phi.Q = 1
                phi.D = np.array([6])
                phi.J = np.array([[4]])
                # phi.measurePauli(2, vec, vec)  # measure -(XZ)(XZ)
                # vec1 = np.zeros(t)
                # vec2 = np.zeros(t)
                # vec1[idx*2] = 1
                # vec2[idx*2+1] = 1
                # phi.measurePauli(0, vec1, vec2)  # measure XZ
            else:
                phase += 1  # account for e^(i\pi/4)
                # |01> + |10>, stabilized by XX and -ZZ
                # Act by H then S: ZZ and XX
                phi.measurePauli(0, np.zeros(t), vec)  # measure XX
                phi.measurePauli(2, vec, np.zeros(t))  # measure ZZ

                # phi.measurePauli(0, vec, np.zeros(t))  # measure ZZ
                # phi.measurePauli(0, vec, vec)  # measure (XZ)(XZ)

    inner = StabilizerState.innerProduct(theta, phi)
    # print("\n", np.round(phi.unpack(), 3))
    # print(i, phase, inner)

    return np.exp(phase*1j*np.pi/4)*inner
    # return inner
    # return np.exp(phase*1j*np.pi/4)*StabilizerState.innerProduct(theta, phi)
    # return StabilizerState.innerProduct(theta, phi)


# Evaluate || <theta| P |H^t> ||^2 for a random stabilizer state theta.
def sampleProjector(args):
    (P, L, Lnorm, seed, parallel) = args  # unpack arguments (easier for parallel code)
    (phases, xs, zs) = P

    # empty projector
    if len(phases) == 0:
        return 1

    t = len(xs[0])

    # clifford circuit
    if t == 0:
        lookup = {0: 1, 2: -1}
        generators = [1]  # include identity
        for phase in phases: generators.append(lookup[phase])

        # calculate sum of all permutations of generators
        return sum(generators)/len(generators)

    # set unique seed for this calculation
    np.random.seed((seed) % 4294967296)

    # sample random theta
    theta = StabilizerState.randomStabilizerState(t)

    # project random state to P
    projfactor = 1
    for g in range(len(phases)):
        res = theta.measurePauli(phases[g], zs[g], xs[g])
        projfactor *= res

        if res == 0: return 0  # theta annihilated by P

    if L is None:  # use exact decomp into pairs of stabilizer states
        func = evalHcomponent
        size = int(np.ceil(t/2))  # need one bit for each pair, plus one more if t is odd
    else:
        func = evalLcomponent
        size = len(L)

    if parallel:  # parallelize for large enough L
        pool = Pool()
        total = sum(pool.map(func, [(i, L, theta, t, seed) for i in range(0, 2**size)]))
        pool.close()
        pool.join()
    else:
        total = sum(map(func, [(i, L, theta, t, seed) for i in range(0, 2**size)]))

    return 2**t * np.abs(projfactor*total/Lnorm)**2
