import numpy as np
import os
import sys
sys.path.append(os.path.abspath("."))

# import matplotlib.pyplot as plt
from libcirc.stabilizer.stabilizer import StabilizerState

n = 4
L = 10  # per state

print("n =", n, "and L =", L)


def numstates(k):
    if k < 1: return False
    if k == 1: return 6
    return 2*(2**k + 1)*numstates(k-1)


N = numstates(n)

output = {}

for i in range(L*N):
    print(i, "/", L*N, end="\r")
    state, d = StabilizerState.randomStabilizerState(n, provide_d=True)
    key = ""

    # key = "\n" + str(d) + "\n"
    # key += str(X) + "\n"
    # key += str(state.G) + "\n"
    # key += str(state.G[:state.k, :]) + "\n"

    if False:
        for zi in range(2**state.k):
            if state.k == 0:
                # x = (np.zeros(state.n) + state.h)
                x = (np.zeros(state.n))
            else:
                z = np.array(list(np.binary_repr(zi, width=state.k))).astype(int)[::-1]
                # x = ((np.dot(z, state.G[:state.k, :]) % 2) + state.h) % 2
                try:
                    x = ((np.dot(z, state.G[:state.k, :]) % 2)) % 2
                except:
                    import pdb; pdb.set_trace()
            key += str(x.astype(int).astype(str)) + ", "

    if True:
        state = state.unpack()*2**(state.k/2)
        state = np.round(state, 3)
        i = 0
        while state[i] == 0: i += 1
        phase = state[i]/np.abs(state[i])
        state /= phase

        state = np.round(state, 3)
        key = ""
        for i in state:
            key += str(int(i.real)) if i.real != 0 else "0"
            key += ","
            key += str(int(i.imag)) if i.imag != 0 else "0"
            key += ";\t"

    if key not in output.keys(): output[key] = 0
    output[key] += 1

print(len(output.keys()), "/", N, "states supported")

purity = 0
for key in output.keys():
    # print(key, output[key])
    purity += (output[key] / (L*N))**2

print("N*purity:", np.round(N*purity, 3), "/", N)
