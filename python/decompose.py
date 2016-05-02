#
# file: decompose.py
# construct a stabilizer decomposition of |H^(\otimes t)>
#

import numpy as np


def decompose(t, delta):
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

    if (k > t): k = t  # prevent infinite loops. Needed only for small t.

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

    print("Found in %d steps, k=%d!" % (count, k))

    norm = np.sqrt(2**k * Z_L)
    print("Z(L) ", Z_L)
    # norm = np.sqrt(2**k)
    # norm = innerProd

    return norm, L

if __name__ == "__main__":
    print(decompose(9, 0.0001))
