#################################################################################
# Goal: Implement some of the algorithms in the appendix for learning purposes  #
# ###############################################################################

#
# File: stabilizer.py
# Defines a class for a stabilizer state
# Implements all algorithms in the appendix
#

import numpy as np


class StabilizerState:

    # Create an empty stabilizer state as used by
    # the RandomStabilizerState function. It has
    # K =\mathbb{F}^n_2 and has q(x) = 0 for all x.
    def __init__(self, n, k):
        if n < 2:
            raise ValueError("Vector space must have positive nonzero nontrivial dimension.")
        if k > n:
            raise ValueError("Stabilizer state dimension cannot be greater than vector space dimension.")

        # define K from RandomStabilizerState algorithm (page 16)
        self.n = n
        self.k = k
        self.h = np.zeros(n)        # in \mathbb{F}^n_2
        self.G = np.identity(n)     # in \mathbb{F}^{n\times n}_2
        self.Gbar = np.identity(n)  # = (G^-1)^T

        # define q to be zero for all x
        self.Q = 0                  # in \mathbb{Z}_8
        self.D = np.zeros(k)        # in {0,2,4,6}^k
        self.J = np.zeros((k, k))   # in {0,4}^{k\times k}, symmetric

    # -------------- State Vector ------------------

    # Does the affine space contain vector x?
    def contains(self, x):
        if len(x) != self.n:
            raise ValueError("Input vector is not the right length for the vector space.")

        # If k = n then K spans all of \mathbb{F}^n_2
        if (self.k == self.n):
            return True

        # try to write x as a linear combination of basis vectors of K
        B = self.G[:self.k].T
        vecx = np.linalg.lstsq(B, x + self.h)[0].astype(int) % 2

        # check result: should succeed if x in K
        return np.allclose(np.dot(B, vecx) % 2, (x + self.h) % 2)

    # Evaluate q(x) for a string in K (page 10, equation 42)
    def q(self, x):
        if len(x) != self.n:
            raise ValueError("Input vector is not the right length for the vector space.")

        # If affine space has dimension zero then phase does not matter
        if (self.k == 0):
            return 0

        # x is a length n vector in basis of \mathbb{F}^n_2
        # vecx is a length k vector in basis of L(K)

        # Determine vecx from x. Method: https://gist.github.com/mikofski/11192605
        # B is n*k 'basis matrix' with each row a length n basis vector
        # Let vecx and x be row vectors. Then solve equation B vecx = x+h
        B = self.G[:self.k].T
        vecx = np.linalg.lstsq(B, x + self.h)[0].astype(int) % 2

        # check result: should succeed if x in K
        if not np.allclose(np.dot(B, vecx) % 2, (x + self.h) % 2):
            raise LookupError("Input vector is not the affine space.")

        # Evaluate equation 42
        qx = self.Q
        qx += np.inner(self.D, vecx)

        for a in range(self.k):
            for b in range(a):
                qx += self.J[a, b]*vecx[a]*vecx[b]

        return qx % 8

    # Coefficient for x in the superposition
    def coeff(self, x):
        if len(x) != self.n:
            raise ValueError("Input vector is not the right length for the vector space.")

        # compute coefficient according to page 10, equation 46
        try:
            return np.power(2, -0.5*self.k) * np.exp(self.q(x) * 1j * np.pi/4)
        except LookupError:
            return 0  # if vector is not in affine space

    # ------------------ Shrink --------------------

    # helper to update D, J using equations 48, 49 on page 10
    def updateDJ(self, R):
        # equation 48
        self.D = np.dot(R, self.D)
        for b in range(self.k):
            for c in range(b):
                self.D += self.J[b, c]*R[b]*R[:, c]
        self.D = self.D % 8

        # equation 49
        self.J = np.dot(np.dot(R, self.J), R.T) % 8

    # helper to update Q, D using equations 51, 52 on page 10
    def updateQD(self, y):
        # equation 51
        self.Q += np.dot(self.D, y)
        for a in range(self.k):
            for b in range(a):
                self.Q += self.J[a, b]*y[a]*y[b]
        self.Q = self.Q % 8

        # equation 52
        self.D += np.dot(self.J, y)
        self.D = self.D % 8

    # attempt to shrink the stabilizer state by eliminating a part of
    # the basis that has inner product alpha with vector xi
    def shrink(self, xi, alpha, lazy=False):
        if len(xi) != self.n:
            raise ValueError("Input vector xi is not the right length for the vector space.")

        # S <- { a in [k] : (xi, g) = 1 }
        # Note that a is zero-indexed.
        S = [a for a in range(self.k) if np.inner(self.G[a], xi) % 2 == alpha]

        beta = (alpha + np.inner(xi, self.h)) % 2

        if len(S) == 0 and beta == 1:
            return "EMPTY"

        if len(S) == 0 and beta == 0:
            return "SAME"

        i = S[0]  # pick any i in S
        S.remove(i)

        for a in S:
            # g^a <- g^a \oplus g^i
            # compute shift matrix for G
            shift = np.concatenate((np.zeros((a, self.n)), [self.G[i]],
                                    np.zeros((self.n - a - 1, self.n))))
            self.G = (self.G + shift) % 2

            # update D, J using equations 48, 49 on page 10
            # compute k*k basis change matrix R (equation 47)
            if not lazy:
                R = np.identity(self.k)
                R[a, i] = 1
                self.updateDJ(R)

            # gbar^i <- gbar^i + \sum_a gbar^a
            self.Gbar[i] += self.Gbar[a]
        self.Gbar = self.Gbar % 2

        # swap g^i and g^k, gbar^i and gbar^k
        # remember elements are zero-indexed, so we use k-1
        self.G[[i, self.k-1]] = self.G[[self.k-1, i]]
        self.Gbar[[i, self.k-1]] = self.Gbar[[self.k-1, i]]

        # update D, J using equations 48, 49 on page 10
        if not lazy:
            R = np.identity(self.k)
            R[[i, self.k-1]] = R[[self.k-1, i]]
            self.updateDJ(R)

        # h <- h \oplus beta*g^k
        self.h = (self.h + beta*self.G[self.k-1]) % 2

        if not lazy:
            # update Q, D using equations 51, 52 on page 10
            y = np.zeros(self.k)
            y[self.k-1] = beta
            self.updateQD(y)

            # remove last row and column from J
            self.J = self.J[1:, 1:]

            # remove last element from D
            self.D = self.D[1:]

        self.k -= 1

        return "SUCCESS"

    # ---------------- InnerProduct ----------------

    def innerProduct(state1, state2):
        raise NotImplementedError

    # ------------ RandomStabilizerState -----------

    # cache probability distributions for stabilizer state dimension k
    # in a dictionary, with a key for each n
    dDists = {}

    @classmethod
    def randomStabilizerState(cls, n, provide_d=False):
        if n < 2:
            raise ValueError("Vector space must have positive nonzero nontrivial dimension.")

        # ensure probability distribution is available for this n
        if n not in cls.dDists:
            # compute distribution given by equation 79 on page 15
            def eta(d):
                if d == 0:
                    return 0

                product = 1
                for a in range(1, d+1):
                    product *= (1 - 2**(d - n - a))
                    product /= (1 - 2**(-a))

                return 2**(-d*(d+1)/2) * product

            # collect numerators
            dist = np.array([])
            for d in range(n):
                dist = np.append(dist, [eta(d)], 0)

            # normalize
            norm = sum(dist)
            dist /= norm

            # cache result
            cls.dDists[n] = dist

        # sample d from distribution
        sample = 1-np.random.random()  # sample from (0.0, 1.0]
        d = 0
        cumulative = 0
        while cumulative < sample:
            cumulative += cls.dDists[n][d]
            d += 1

        k = n - d

        # pick random X in \mathbb{F}^{d,n}_2 with rank d
        while True:
            X = np.random.random_integers(0, 1, (d, n))

            if np.linalg.matrix_rank(X) == d:
                break

        # create the state object. __init__ gives the correct properties
        state = StabilizerState(n, k)

        for a in range(d):
            # lazy shrink with a'th row of X
            state.shrink(X[a], 0, lazy=True)
        # now K = ker(X) and is in standard form

        state.h = np.random.random_integers(0, 1, n)
        state.Q = np.random.random_integers(0, 7)
        state.D = 2*np.random.random_integers(0, 3, state.k)

        state.J = np.zeros((state.k, state.k))
        for a in range(state.k):
            state.J[a, a] = 2*state.D[a] % 8
            for b in range(a):
                state.J[a, b] = 4*np.random.random_integers(0, 1)
                state.J[b, a] = state.J[b, a]

        if not provide_d:
            return state
        else:
            return state, d

    # ---------------- MeasurePauli ----------------

    def measurePauli(self, pauli):
        raise NotImplementedError
