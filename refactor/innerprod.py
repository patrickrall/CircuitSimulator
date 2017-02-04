#
# file: innerprod.py
# Methods for evaluating inner products of P|H^t> where
# P is a given projector.
#

import numpy as np
from multiprocessing import Pool
from libcirc.stabilizer.stabilizer import StabilizerState
from libcirc.stateprep import prepH, prepL

# Median of means sampling achieves error e with failure prob p
# (medians) * (means) = 32 * e^(-1/2) * log(1/p)
# http://www.cs.utexas.edu/~ecprice/courses/randomized/notes/lec5.pdf

# Can only parallelize over means, so (means) must be larger
# than the number of cores.


def multiSampledProjector(P, L, Lnorm, medians=100, means=100):
    pass



# If number of samples is L, then this distribution has mean || P |H^t> ||^2
# and standard deviation sqrt((2^t-1)/(2^t+1)) * || P |H^t> ||^2 * (1/sqrt(L))

# On its own, this method guarantees output error e with probability (1 - p)
# if L =
def sampledProjector(P, L, Lnorm, samples=100, procs=1):
    (phases, xs, zs) = P

    # empty projector
    if len(phases) == 0:
        return np.abs(Lnorm)**2

    # clifford circuit
    if len(xs[0]) == 0:
        lookup = {0: 1, 2: -1}
        gens = [1]  # include identity
        for phase in phases: gens.append(lookup[phase])

        # calculate sum of all permutations of generators
        return sum(gens)/len(gens) * np.abs(Lnorm)**2

    seeds = np.random.random_integers(0, 2**32-1, samples)
    queries = [(P, L, seed) for seed in seeds]
    if procs > 1:
        pool = Pool(procs)
        return sum(pool.map(singleProjectorSample, queries))/samples
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
    logprojfactor = 0
    for g in range(len(phases)):
        res = theta.measurePauli(phases[g], zs[g], xs[g])
        projfactor *= res
        if res != 0 and res != 1: logprojfactor += 1

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
def exactProjector(P, Lnorm, procs=1):
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

    def thread(P, i):
        (phases, xs, zs) = P
        total = 0

        theta = prepH(i, t)
        projfactor = 1
        for g in range(len(phases)):
            res = theta.measurePauli(phases[g], zs[g], xs[g])
            projfactor *= res
            if res == 0: return 0  # theta annihilated by P

        for j in range(0, 2**size):
            phi = prepH(j, t)
            inner = StabilizerState.innerProduct(theta, phi)

            total += inner * projfactor
        return total

    queries = [(P, i) for i in range(0, 2**size)]
    if procs > 1:
        pool = Pool(procs)
        total = sum(pool.map(thread, queries))
    else:
        total = sum(map(thread, queries))
    return np.abs(total)
