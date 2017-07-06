#
# file: innerprod.py
# Methods for evaluating inner products of P|H^t> where
# P is a given projector.
#

import numpy as np
from multiprocessing import Pool, cpu_count
from libcirc.stabilizer.stabilizer import StabilizerState
from libcirc.stateprep import prepH, prepL


# Median of means calculation can be done via Chebychev and Chernoff bounds.
# http://www.cs.utexas.edu/~ecprice/courses/randomized/notes/lec5.pdf
# If sampledProjector has error worse than e with probability p,
# then multiSampledProjector has error worse than e with probability
# less than delta = exp(-2m(1/2 - p)^2) where m is the number of bins.
#
# Since L is proportional to 1/p, best total number of samples m*L is minimized
# at the minimum of m/p proportional to p^(-1) * (1/2 - p)^(-2) which is p = 1/6.
# Thus for best results pick L = 6 * (2^t - 1)/(2^t + 1) * (1/e^2)
# and m = 4.5 log(1 / delta). This is only better than mean-only sampling
# for failure probabilities smaller than 0.0076 which is the solution to
# the equation 1/p = 27 ln (1/p).
#
# In the large t limit, can achieve e = 0.1 with 99.9% chance with 600 samples,
# and 32 bins. This would need 100 000 samples for mean-only mode, over 5x more.
def multiSampledProjector(P, L, norm, bins=32, samples=600, procs=1):
    means = []
    for i in range(bins):
        means.append(sampledProjector(P, L, norm, samples=samples, procs=procs))
    return np.median(means)


# If number of samples is L, then this distribution has mean || P |H^t> ||^2
# and standard deviation sqrt((2^t-1)/(2^t+1)) * || P |H^t> ||^2 * (1/sqrt(L))
#
# On its own, this method guarantees output error e * || P |H^t> ||^2
# with probability (1 - p) if L = (2^t - 1)/(2^t + 1) * (1/p) * (1/e^2)
#
# In the large t limit, can achieve e = 0.1 with 95% chance with 2000 samples.
def sampledProjector(P, L, norm, samples=2000, procs=1):
    (phases, xs, zs) = P

    # empty projector
    if len(phases) == 0:
        return np.abs(norm)**2

    # clifford circuit
    if len(xs[0]) == 0:
        lookup = {0: 1, 2: -1}
        gens = [1]  # include identity
        for phase in phases: gens.append(lookup[phase])

        # calculate sum of all permutations of generators
        return sum(gens)/len(gens) * np.abs(norm)**2

    if procs is None:
        try:
            procs = cpu_count()
        except NotImplementedError:
            procs = 1

    seeds = np.random.random_integers(0, 2**32-1, samples)
    queries = [(P, L, seed) for seed in seeds]
    if procs > 1:
        pool = Pool(procs)
        out = sum(pool.map(singleProjectorSample, queries))/samples
        pool.close()
        return out
    else:
        return sum(map(singleProjectorSample, queries))/samples


# Evaluate 2^t * || <theta| P |H^t> ||^2 for a random theta.
# This distribution has mean || P |H^t> ||^2
# and standard deviation sqrt((2^t-1)/(2^t+1)) * || P |H^t> ||^2
def singleProjectorSample(args):
    (P, L, seed) = args  # unpack arguments (easier for parallel code)
    (phases, xs, zs) = P
    t = len(xs[0])

    # init seed
    np.random.seed(seed)

    # sample random theta
    theta = StabilizerState.randomStabilizerState(t)

    # project random state to P
    projfactor = 1
    for g in range(len(phases)):
        res = theta.measurePauli(phases[g], zs[g], xs[g])
        projfactor *= res

        if res == 0: return 0  # theta annihilated by P

    total = 0
    if L is None:  # exact decomposition
        size = int(np.ceil(t/2))
        for i in range(0, 2**size):
            phi = prepH(i, t)
            total += StabilizerState.innerProduct(theta, phi)

    else:  # approximate decomposition
        size = len(L)
        for i in range(0, 2**size):
            phi = prepL(i, t, L)
            total += StabilizerState.innerProduct(theta, phi)

    return 2**t * np.abs(projfactor*total)**2


# calculate projector exactly
def exactProjector(P, L, norm, procs=1):
    (phases, xs, zs) = P

    # empty projector
    if len(phases) == 0:
        return np.abs(norm)**2

    t = len(xs[0])

    # clifford circuit
    if t == 0:
        lookup = {0: 1, 2: -1}
        generators = [1]  # include identity
        for phase in phases: generators.append(lookup[phase])

        # calculate sum of all permutations of generators
        return sum(generators)/len(generators)

    if L is None: size = int(np.ceil(t/2))
    else: size = len(L)

    if procs is None:
        try:
            procs = cpu_count()
        except NotImplementedError:
            procs = 1

    queries = [(P, L, l) for l in range(0, 2**(size-1) * (2**size + 1))]
    if procs > 1:
        pool = Pool(procs)
        total = sum(pool.map(exactProjectorWork, queries))
        pool.close()
    else:
        total = sum(map(exactProjectorWork, queries))
    return np.abs(total)


def exactProjectorWork(args):
    (P, L, l) = args
    (phases, xs, zs) = P

    t = len(xs[0])
    if L is None: size = int(np.ceil(t/2))
    else: size = len(L)

    chi = 2**size

    i = 0
    while l >= chi - i:
        l -= chi - i
        i += 1
    j = l + i

    if L is None: theta = prepH(i, t)
    else: theta = prepL(i, t, L)

    projfactor = 1
    for g in range(len(phases)):
        res, status = theta.measurePauli(phases[g], zs[g], xs[g], give_status=True)

        projfactor *= res

        if res == 0: return 0  # theta annihilated by P

    if L is None: phi = prepH(j, t)
    else: phi = prepL(j, t, L)

    inner = StabilizerState.innerProduct(theta, phi)
    if i == j:
        return inner * projfactor
    else:
        return 2 * np.real(inner) * projfactor
