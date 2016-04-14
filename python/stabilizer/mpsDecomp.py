#########################################################################
# Mini research project: How well do stabilizer states fit into an MPS? #
#########################################################################

#
# File: mpsDecomp.py
# Computes MPS schmidt rank of random stabilizer states
# Generates some statistics
#

import numpy as np
from stabilizer import StabilizerState
import matplotlib.pyplot as plt
from tests import decToBitstring

# ----------------- Main -----------------------

n = 8
nsamples = int(1e3)

samples = []

dsamples = {}
for d in range(1, n+1):
    dsamples[d] = []

for i in range(nsamples):
    # state, d = StabilizerState.randomStabilizerState(n, provide_d=True)
    # samples.append(state)

    # dsamples[d].append(n-state.k)
    pass

raise NotImplementedError
