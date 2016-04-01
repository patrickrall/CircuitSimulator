#################################################################################
# Goal: Implement some of the algorithms in the appendix for learning purposes  #
# ###############################################################################

#
# File: stabilizer.py
# Defines a class for a stabilizer state
# Implements all algorithms in the appendix
#

import numpy as np
from copy import deepcopy


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
        if (self.k == self.n): return True

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
        if (self.k == 0): return 0

        # x is a length n vector in basis of \mathbb{F}^n_2
        # vecx is a length k vector in basis of L(K)

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
        try: return np.power(2, -0.5*self.k) * np.exp(self.q(x) * 1j * np.pi/4)
        except LookupError: return 0  # if vector is not in affine space

    # ---------- Common Helper Functions -----------

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

    # -------------- Exponential Sum ---------------

    # evaluates the expression in the comment on page 12
    def evalW(p, m, eps):
        return eps * 2**(p/2) * np.exp(1j*np.pi*m/4)

    # Helpers for evaluating equations like 63, 68. For even A,B only!

    # Evaluates 1 + e^{A*i*pi/4} + e^{A*i*pi/4} - e^{(A + B)*i*pi/4}
    def Gamma(A, B):
        if not (A % 2 == 0 and B % 2 == 0):
            raise ValueError("A and B must be even!")

        lookup = {0: 1, 2: 1j, 4: -1, 6: -1j}
        gamma = 1 + 0j + lookup[A % 8] + lookup[B % 8] - lookup[(A + B) % 8]

        if gamma == 0: return(0, 0, 0)  # eps, p, m
        lookup = {1: 0, 1+1j: 1, 1j: 2, -1: 4, -1j: 6, 1-1j: 7}
        return(1, 2, lookup[gamma/2])

    # Evaluates 1 + e^{A*i*pi/4}
    def partialGamma(A):
        if not (A % 2 == 0): raise ValueError("A must be even!")
        lookup = {0: (1, 2, 0), 2: (1, 1, 1), 4: (0, 0, 0), 6: (1, 2, 7)}
        return lookup[A % 8]

    # Helper required for InnerProduct and MeasurePauli.
    # Depends only on Q, D, J. Manipulates integers p, m, eps
    # to avoid rounding error then evaluates to a real number.
    def exponentialSum(self):
        S = [a for a in range(self.k) if self.D[a] in [2, 6]]
        if len(S) != 0:
            a = S[0]

            # Construct R as in comment on page 12
            R = np.identity(self.k)
            for b in S[1:]:
                R[b, a] += 1
            R = R % 2

            self.updateDJ(R)
            S = [a]
        # Now J[a, a] = 0 for all a not in S

        E = [k for k in range(self.k) if k not in S]
        M = []
        Dimers = []  # maintain list of dimers rather than r

        while len(E) > 0:
            a = E[0]
            K = [b for b in E[1:] if self.J[a, b] == 4]

            if len(K) == 0:  # found a new monomer {a}
                M.append(a)
                E = E[1:]
            else:
                b = K[0]

                # Construct R for basis change
                R = np.identity(self.k)
                for c in [x for x in E if x != a and x != b]:
                    if self.J[a, c] == 4: R[c, a] += 1
                    if self.J[b, c] == 4: R[c, b] += 1
                R = R % 2

                self.updateDJ(R)

                # {a, b} form a new dimer
                Dimers.append([a, b])
                E = [x for x in E if x != a and x != b]

        if len(S) != 0:
            # Compute W(K,q) from Eq. 63
            W = (1, 0, self.Q)
            for c in M:
                (eps, p, m) = self.partialGamma(self.D[c])
                if eps == 0: return 0
                W = (1, W[1] + p, W[2] + m)
            for dim in Dimers:
                (eps, p, m) = self.Gamma(self.D[dim[0]], self.D[dim[1]])
                if eps == 0: return 0
                W = (1, W[1] + p, W[2] + m)
            return self.evalW(W[0], W[1], W[2])
        else:
            s = S[0]

            # Compute W_0, W_1 from Eq. 68
            def Wsigma(sigma):
                W = (1, 0, self.Q + sigma*self.D[s])
                for c in M:
                    (eps, p, m) = self.partialGamma(self.D[c] + sigma*self.J[c, s])
                    if eps == 0: return 0
                    W = (1, W[1] + p, W[2] + m)
                for dim in Dimers:
                    (eps, p, m) = self.Gamma(self.D[dim[0]] + sigma*self.J[dim[0], s],
                                             self.D[dim[1]] + sigma*self.J[dim[1], s])
                    if eps == 0: return 0
                    W = (1, W[1] + p, W[2] + m)
                return self.evalW(W[0], W[1], W[2])
            return Wsigma(0) + Wsigma(1)

    # ------------------ Shrink --------------------

    # attempt to shrink the stabilizer state by eliminating a part of
    # the basis that has inner product alpha with vector xi
    def shrink(self, xi, alpha, lazy=False):
        if len(xi) != self.n:
            raise ValueError("Input vector xi is not the right length for the vector space.")

        # S <- { a in [k] : (xi, g) = 1 }
        # Note that a is zero-indexed.
        S = [a for a in range(self.k) if np.inner(self.G[a], xi) % 2 == alpha]

        beta = (alpha + np.inner(xi, self.h)) % 2

        if len(S) == 0 and beta == 1: return "EMPTY"

        if len(S) == 0 and beta == 0: return "SAME"

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

    @classmethod
    def innerProduct(cls, state1, state2):
        if not (state1.n == state2.n):
            raise ValueError("States do not have same dimension.")

        # K <- K_1, (also copy q_1)
        state = deepcopy(state1)
        for b in range(state2.k, state2.n):  # not k_2+1 because of zero-indexing
            alpha = np.dot(state2.k, state2.Gbar[b]) % 2
            eps = state.shrink(state2.Gbar[b], alpha)  # Lazy?
            if eps == "EMPTY": return 0

        # Now K = K_1 \cap K_2

        y = np.zeros(state2.k)
        R = np.identity(np.max(state2.k, state.k))  # Is this correct?
        for a in range(state2.k):
            y[a] = np.dot((state.h + state2.h) % 2, state2.Gbar[a]) % 2
            for b in range(state.k):
                R[b, a] = np.dot(state.G[b], state2.Gbar[a]) % 2

        # need a copy of state2 that I can mutate
        state2 = deepcopy(state2)

        state2.h += np.dot(y, state2.G[:state2.k])
        state2.h = state2.h % 2
        state2.updateQD(y)
        state2.updateDJ(R)

        # now q1, q2 are defined in the same basis
        state.Q = state1.Q - state2.Q
        state.D = state1.D - state2.D
        state.J = state1.J - state2.J

        return (2**(-(state1.k + state2.k)/2)) * state.exponentialSum()

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
                if d == 0: return 0

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

            if np.linalg.matrix_rank(X) == d: break

        # create the state object. __init__ gives the correct properties
        state = StabilizerState(n, k)

        for a in range(d):
            # lazy shrink with a'th row of X
            state.shrink(X[a], 0, lazy=True)

            # reset state's k after shrinking
            state.k = k

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

        if not provide_d: return state
        else: return state, d

    # ---------------- MeasurePauli ----------------

    # Helper: if xi not in K, extend it to an affine space that does
    # Doesn't return anything, instead modifies state
    def extend(self, xi):
        S = [a for a in range(self.n) if np.dot(xi, self.Gbar[a]) % 2 == 1]
        T = [a for a in S if self.k < a and self.k < self.n]  # zero indexing

        if len(T) == 0: return  # xi in L(K)

        i = T[0]
        S = [a for a in S if a != i]

        for a in S:
            self.Gbar[a] += self.Gbar[i]
        self.Gbar = self.Gbar % 2

        for a in S:
            self.G[i] += self.G[a]
        self.G[i] = self.G[i] % 2
        # Now g^i = xi

        # Swap g^i and g^k (not g^{k+1} because zero indexing)
        self.G[[i, self.k]] = self.G[[self.k, i]]
        self.Gbar[[i, self.k]] = self.Gbar[[self.k, i]]

    # Write a pauli as P = i^m * Z(zeta) * X(xi), m in Z_4
    # Returns the norm of the projected state Gamma = ||P_+ |K,q>||
    # If Gamma nonzero, projects the state to P_+|K,q>
    def measurePauli(self, m, zeta, xi):
        if not (self.n == len(zeta) and self.n == len(xi)):
            raise ValueError("States and Pauli do not have same dimension.")

        # write zeta, xi in basis of K
        vecZeta = np.zeros(self.k)
        vecXi = np.zeros(self.k)
        for a in range(self.k):
            vecZeta[a] = np.dot(self.Gbar[a], zeta) % 2
            vecXi[a] = np.dot(self.Gbar[a], xi) % 2

        xiPrime = np.dot(vecXi, self.G[:self.k])

        # compute w in {0, 2, 4, 6} using eq. 88
        w = 2*m
        w += 4*(np.dot(zeta, self.h) % 2)
        w += np.dot(self.D, vecXi)
        for b in range(self.k):
            for a in range(b):
                w += self.J[a, b]*vecXi[a]*vecXi[b]
        w = w % 8

        # Compute eta_0, ..., eta_{k-1} using eq. 94
        eta = np.zeros(self.k)
        for a in range(self.k):
            eta[a] = vecZeta[a]
            for b in range(self.k):
                eta[a] += self.J[a, b]*vecXi[b]
        eta = eta % 2

        if xiPrime == xi and w in [0, 4]:
            gamma = np.dot(eta, self.G[:self.k]) % 2
            omegaPrime = w/4
            alpha = (omegaPrime + np.dot(eta, self.h)) % 2  # different bases?

            eps = self.shrink(gamma, alpha)

            if eps == "EMPTY": return 0
            if eps == "SAME": return 1
            if eps == "SUCCESS": return 2**(-1/2)

        if xiPrime == xi and w in [2, 6]:
            sigma = 2 - w/2

            # Update Q, D, J using equations 100, 101
            self.Q = (self.Q + sigma) % 8
            for a in range(self.k):
                self.D[a] = (self.D[a] - 2*sigma*eta[a]) % 8
            for a in range(self.k):
                for b in range(self.k):
                    if (a == b): continue
                    self.J[a, b] = (self.J[a, b] + 4*eta[a]*eta[b]) % 8

            return 2**(-1/2)

        # remaining case: xiPrime != xi
        self.extend(xi)
        self.D = np.concatenate((self.D, [2*m + 4*(np.dot(zeta, self.h + xi) % 2)]))

        # update J using equation 104
        self.J = np.concatenate((self.J, [4*vecZeta]))
        self.J = np.concatenate((self.J, np.transpose([np.concatenate((4*vecZeta, 4*m))])), 1)

        return 2**(-1/2)
