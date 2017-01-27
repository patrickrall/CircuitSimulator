#
# file: sample.py
# construct a stabilizer decomposition of |H^(\otimes t)>
# Sample from || \Pi |H^(\otimes t)>
#

import numpy as np
from multiprocessing import Pool
from libcirc.stabilizer.stabilizer import StabilizerState


# Needs from config dict: exact, k, fidbound, rank, fidelity, forceL, verbose, quiet
def decompose(t, config):
    quiet = config.get("quiet")
    verbose = config.get("verbose")

    # trivial case
    if t == 0:
        return np.array([[]]), 1

    v = np.cos(np.pi/8)

    # exact case
    norm = (2)**(np.floor(t/2)/2)
    if (t % 2 == 1): norm *= (2*v)
    if config.get("exact"): return None, norm

    k = config.get("k")
    forceK = (k is not None)  # Was k selected by the user

    if k is None:
        if config.get("fidbound") is None:
            raise ValueError("Need to specify either k or fidbound, or set exact=True to determine sampling method.")
        # pick unique k such that 1/(2^(k-2)) \geq v^(2t) \delta \geq 1/(2^(k-1))
        k = np.ceil(1 - 2*t*np.log2(v) - np.log2(config.get("fidbound")))
        if verbose: print("Autopicking k = %d." % k)
    k = int(k)

    # can achieve k = t/2 by pairs of stabilizer states
    # revert to exact norm
    if k > t/2 and not forceK and not config.get("forceL"):
        if verbose: print("k > t/2. Reverting to exact decomposition.")
        return None, norm

    # prevents infinite loops
    if (k > t):
        if forceK and not quiet: print("Can't have k > t. Setting k to %d." % t)
        k = t

    innerProd = 0
    Z_L = None

    while innerProd < 1-config.get("fidbound") or forceK:

        L = np.random.random_integers(0, 1, (k, t))

        if (config.get("rank")):
            # check rank
            if (np.linalg.matrix_rank(L) < k):
                if not quiet: print("L has insufficient rank. Sampling again...")
                continue

        if config.get("fidelity"):
            # compute Z(L) = sum_x 2^{-|x|/2}
            Z_L = 0
            for i in range(2**k):
                z = np.array(list(np.binary_repr(i, width=k))).astype(int)[::-1]
                x = np.dot(z, L) % 2
                Z_L += 2**(-np.sum(x)/2)

            innerProd = 2**k * v**(2*t) / Z_L
            if forceK:
                # quiet can't be set for this
                print("Inner product <H^t|L>: %f" % innerProd)
                break
            elif innerProd < 1-config.get("fidbound"):
                if not quiet: print("Inner product <H^t|L>: %f - Not good enough!" % innerProd)
            else:
                if not quiet: print("Inner product <H^t|L>: %f" % innerProd)
        else: break

    if config.get("fidelity"):
        norm = np.sqrt(2**k * Z_L)
        return L, norm
    else:
        return L, None


# Inner product for some state in |L> ~= |H^t>
def evalLcomponent(args):
    (i, L, theta, t, exact) = args  # unpack arguments (easier for parallel code)

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

    return StabilizerState.innerProduct(theta, phi, exact=exact)


# Inner product for some state in |H^t> using pairwise decomposition
def evalHcomponent(args):
    (i, _, theta, t, exact) = args  # unpack arguments (easier for parallel code)

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

    innerProd = StabilizerState.innerProduct(theta, phi, exact=exact)
    return innerProd


# Evaluate || <theta| P |H^t> ||^2 for a random stabilizer state theta.
def sampleProjector(args):
    (P, L, seed, parallel) = args  # unpack arguments (easier for parallel code)
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

    # init seed
    np.random.seed(seed)

    # sample random theta
    theta = StabilizerState.randomStabilizerState(t)

    # project random state to P
    projfactor = 1
    logprojfactor = 0
    for g in range(len(phases)):
        res = theta.measurePauli(phases[g], zs[g], xs[g])
        projfactor *= res
        if res != 0 and res != 1: logprojfactor += 1

        if res == 0: return 0  # theta annihilated by P

    if L is None:  # use exact decomp into pairs of stabilizer states
        func = evalHcomponent
        size = int(np.ceil(t/2))  # need one bit for each pair, plus one more if t is odd
    else:
        func = evalLcomponent
        size = len(L)

    # experimental feature: suppress numerical error
    suppress_numerical = False

    if not suppress_numerical:
        if parallel:  # parallelize for large enough L
            pool = Pool()
            total = sum(pool.map(func, [(i, L, theta, t, False) for i in range(0, 2**size)]))
            pool.close()
            pool.join()
        else:
            total = sum(map(func, [(i, L, theta, t, False) for i in range(0, 2**size)]))

        return 2**t * np.abs(projfactor*total)**2
    else:
        realpart = {}
        imagpart = {}

        def insert(p, dic, sign):
            if p not in dic.keys(): dic[p] = 0
            dic[p] += sign

        for i in range(0, 2**size):
            (eps, pp, m) = func((i, L, theta, t, True))
            if eps == 0: continue

            p = pp
            p -= logprojfactor
            p += t

            if m == 0: insert(p, realpart, +1)
            if m == 2: insert(p, imagpart, +1)
            if m == 4: insert(p, realpart, -1)
            if m == 6: insert(p, imagpart, -1)

            if m == 1:
                insert(p-1, realpart, +1)
                insert(p-1, imagpart, +1)

            if m == 3:
                insert(p-1, realpart, -1)
                insert(p-1, imagpart, +1)

            if m == 5:
                insert(p-1, realpart, -1)
                insert(p-1, imagpart, -1)

            if m == 7:
                insert(p-1, realpart, +1)
                insert(p-1, imagpart, -1)

        out = 0
        for p in realpart.keys():
            out += 2**(p/2) * realpart[p]
        for p in imagpart.keys():
            out += 2**(p/2)*1j * imagpart[p]

        return np.abs(out)**2
