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
        if n < 1:
            raise ValueError("Vector space must have positive nonzero dimension.")
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
        self.J = np.zeros((k, k))   # in {0,4}^{k\times k}, symmetric, diagonal is 2*self.D

    # print some python code that initializes the state. Handy for debugging.
    def display(self):
        print("state.n = " + str(self.n))
        print("state.k = " + str(self.k))

        print("state.h = np.array([" + ",".join([str(int(x)) for x in np.array(self.h)]) + "])")

        print("state.G = np.array([[" + "],[".join([",".join([str(int(x)) for x in y]) for y in np.array(self.G)]) + "]])")
        print("state.Gbar = np.array([[" + "],[".join([",".join([str(int(x)) for x in y]) for y in np.array(self.Gbar)]) + "]])")

        print("state.Q = "+str(int(self.Q)))
        print("state.D = np.array([" + ",".join([str(int(x)) for x in np.array(self.D)]) + "])")

        if (self.k > 0):
            print("state.J = np.array([[" + "],[".join([",".join([str(int(x)) for x in y]) for y in np.array(self.J)]) + "]])")
        else: print("J = np.array((0,0))")

    # -------------- State Vector ------------------

    # bring into state vector format
    def unpack(self):
        omega = np.exp(1j*np.pi/4)
        self.J = np.array(self.J)

        psi = np.zeros(2**self.n).astype(complex)
        if self.k == 0:
            idx = int("".join(self.h.astype(int).astype(str)), 2)
            psi[idx] = omega**self.Q
        else:
            J = deepcopy(self.J)
            np.fill_diagonal(J, 0)
            J = (J/4).astype(int)

            for zi in range(2**self.k):
                z = np.array(list(np.binary_repr(zi, width=self.k))).astype(int)[::-1]
                phase = omega**(self.Q + np.dot(self.D, z) + 2*np.dot(np.dot(z, J), z))

                x = ((np.dot(z, self.G[:self.k, :]) % 2) + self.h) % 2
                idx = int(("".join(x.astype(int).astype(str))), 2)
                psi[idx] = phase

        return psi/2**(self.k/2)

    # ---------- Common Helper Functions -----------

    # helper to update D, J using equations 49, 50 on page 10
    def updateDJ(self, R):
        # equation 49
        self.D = np.dot(R, self.D)

        for b in range(self.k):
            for c in range(b):
                self.D += self.J[b, c]*R[:, b]*R[:, c]
        self.D = self.D % 8

        # equation 50
        self.J = np.dot(np.dot(R, self.J), R.T) % 8

    # helper to update Q, D using equations 51, 52 on page 10
    def updateQD(self, y):
        # type stuff
        y = y.astype(int)
        self.D = self.D.astype(int)
        self.J = np.array(self.J.astype(int))

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
    def evalW(self, eps, p, m):
        return eps * 2**(p/2) * np.exp(1j*np.pi*m/4)

    # Helpers for evaluating equations like 63, 68. For even A,B only!

    # Evaluates 1 + e^{A*i*pi/4} + e^{A*i*pi/4} - e^{(A + B)*i*pi/4}
    def Gamma(self, A, B):
        if not (A % 2 == 0 and B % 2 == 0):
            raise ValueError("A and B must be even!")

        lookup = {0: 1, 2: 1j, 4: -1, 6: -1j}
        gamma = 1 + 0j + lookup[A % 8] + lookup[B % 8] - lookup[(A + B) % 8]

        if gamma == 0: return(0, 0, 0)  # eps, p, m
        lookup = {1: 0, 1+1j: 1, 1j: 2, -1: 4, -1j: 6, 1-1j: 7}
        return(1, 2, lookup[gamma/2])

    # Evaluates 1 + e^{A*i*pi/4}
    def partialGamma(self, A):
        if not (A % 2 == 0): raise ValueError("A must be even!")
        lookup = {0: (1, 2, 0), 2: (1, 1, 1), 4: (0, 0, 0), 6: (1, 1, 7)}
        return lookup[A % 8]

    # Helper required for InnerProduct and MeasurePauli.
    # Depends only on Q, D, J. Manipulates integers p, m, eps
    # to avoid rounding error then evaluates to a real number.
    def exponentialSum(self, exact=False):

        S = [a for a in range(self.k) if self.D[a] in [2, 6]]

        if len(S) != 0:
            a = S[0]

            # Construct R as in comment on page 12
            R = np.identity(self.k)
            for b in S[1:]:
                R[b, a] += 1
            R = R % 2

            self.updateDJ(R)

            # swap a and k, such that we only need to focus
            # on k-1)*(k-1) submatrix of J later
            R = np.identity(self.k)
            R[:, [a, self.k-1]] = R[:, [self.k-1, a]]

            self.updateDJ(R)

            S = [self.k-1]

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
                R = np.identity(self.k)
                for c in [x for x in E if x not in [a, b]]:
                    if self.J[a, c] == 4: R[c, b] += 1
                    if self.J[b, c] == 4: R[c, a] += 1

                R = R % 2
                self.updateDJ(R)

                # {a, b} form a new dimer
                Dimers.append([a, b])
                E = [x for x in E if x not in [a, b]]

        # helper to distinguish exact and non-exact cases
        def zero():
            if exact: return (0, 0, 0)
            else: return 0

        def Wsigma(sigma, s=0):
            if (self.k == 0):
                if exact: return (1, 0, self.Q)
                else: return self.evalW(1, 0, self.Q)

            W = (1, 0, self.Q + sigma*self.D[s])
            for c in M:
                (eps, p, m) = self.partialGamma(self.D[c] + sigma*self.J[c, s])
                if eps == 0:
                    return zero()
                W = (1, W[1] + p, (W[2] + m) % 8)
            for dim in Dimers:
                (eps, p, m) = self.Gamma(self.D[dim[0]] + sigma*self.J[dim[0], s],
                                         self.D[dim[1]] + sigma*self.J[dim[1], s])
                if eps == 0: return zero()
                W = (1, W[1] + p, (W[2] + m) % 8)
            if exact: return W
            else: return self.evalW(W[0], W[1], W[2])

        if len(S) == 0:
            # Compute W(K,q) from Eq. 63
            return Wsigma(0)
        else:
            s = S[0]

            # Compute W_0, W_1 from Eq. 68

            if not exact: return Wsigma(0, s) + Wsigma(1, s)
            else:
                (eps0, p0, m0) = Wsigma(0, s)
                (eps1, p1, m1) = Wsigma(1, s)

                if eps0 == 0: return (eps1, p1, m1)
                if eps1 == 0: return (eps0, p0, m0)
                # Now eps1 == eps0 == 1

                if p0 != p1: raise ValueError("p0, p1 must be equal.")
                if (m1-m0) % 2 == 1: raise ValueError("m1-m0 must be even.")

                # Rearrange 2^{p0/2} e^{i pi m0/4} + 2^{p1/2} e^{i pi m1/4}
                # To 2^(p0/2) ( 1 + e^(i pi (m1-m0)/4)) and use partialGamma

                (eps, p, m) = self.partialGamma(m1-m0)
                if eps == 0: return (0, 0, 0)
                return (1, p+p0, (m0 + m) % 8)

    # ------------------ Shrink --------------------

    # attempt to shrink the stabilizer state by eliminating a part of
    # the basis that does not have inner product alpha with vector xi
    def shrink(self, xi, alpha, lazy=False):
        if len(xi) != self.n:
            raise ValueError("Input vector xi is not the right length for the vector space.")

        # S <- { a in [k] : (xi, g) = 1 }
        # Note that a is zero-indexed.
        S = [a for a in range(self.k) if np.inner(self.G[a], xi) % 2 == 1]

        beta = (alpha + np.inner(xi, self.h)) % 2

        if len(S) == 0 and beta == 1: return "EMPTY"

        if len(S) == 0 and beta == 0: return "SAME"

        i = S[-1]  # pick last
        S = S[:-1]

        for a in S:
            # g^a <- g^a \oplus g^i
            self.G[a] = (self.G[a] + self.G[i]) % 2

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
            self.J = self.J[:-1, :-1]

            # remove last element from D
            self.D = self.D[:-1]

        self.k -= 1

        return "SUCCESS"

    # ---------------- InnerProduct ----------------

    @classmethod
    def innerProduct(cls, state1, state2, exact=False):

        if not (state1.n == state2.n):
            raise ValueError("States do not have same dimension.")

        # K <- K_1, (also copy q_1)
        state = deepcopy(state1)

        for b in range(state2.k, state2.n):  # not k_2+1 because of zero-indexing
            alpha = np.dot(state2.h, state2.Gbar[b]) % 2
            eps = state.shrink(state2.Gbar[b], alpha)
            if eps == "EMPTY":
                if exact: return (0, 0, 0)
                else: return 0

        # Now K = K_1 \cap K_2

        y = np.dot(state2.Gbar[:state2.k, :], (state.h + state2.h)) % 2

        R = np.dot(state.G[:state.k:, :], state2.Gbar[:state2.k, :].T) % 2

        # R is now k by k_2. Since k <= k_2,
        # pad R with zeros to make it square
        while R.shape[0] < R.shape[1]:
            R = np.vstack((R, np.zeros(state2.k)))

        # need a copy of state2 that I can mutate
        state2 = deepcopy(state2)

        state2.updateQD(y)
        state2.updateDJ(R)

        # now q, q2 are defined in the same basis
        state.Q = (state.Q - state2.Q) % 8
        state.D = ((state.D - state2.D[:state.k]) % 8).astype(int)
        state.J = (state.J - state2.J[:state.k, :state.k]).astype(int) % 8

        if not exact: return (2**(-(state1.k + state2.k)/2)) * state.exponentialSum()
        else:
            (eps, p, m) = state.exponentialSum(exact=True)
            return (eps, p-(state1.k + state2.k), m)

    # ------------ RandomStabilizerState -----------

    # cache probability distributions for stabilizer state dimension k
    # in a dictionary, with a key for each n
    dDists = {}

    @classmethod
    def randomStabilizerState(cls, n, provide_d=False):
        if n < 1:
            raise ValueError("Vector space must have positive nonzero dimension.")

        # ensure probability distribution is available for this n
        if n not in cls.dDists:
            # compute distribution given by equation 79 on page 15
            def logeta(d):
                if d == 0: return 0

                product = 0
                for a in range(1, d+1):
                    product += np.log2((1 - 2**(d - n - a)))
                    product -= np.log2((1 - 2**(-a)))

                return (-d*(d+1)/2) + product

            # collect numerators
            dist = np.array([])
            for d in range(n+1):
                dist = np.append(dist, [2**logeta(d)], 0)

            # normalize
            norm = sum(dist)
            dist /= norm

            # cache result
            cls.dDists[n] = dist

        # sample d from distribution
        cumulative = np.cumsum(cls.dDists[n])
        sample = 1-np.random.random()  # sample from (0.0, 1.0]

        for d in range(n+1):
            if sample <= cumulative[d]:
                break
        k = n - d

        # create the state object. __init__ gives the correct properties
        state = StabilizerState(n, n)

        # Alternate algorithm that does not require rank verification
        # of a separate matrix X.
        # randomly shrink until it has the right dimension
        while state.k > k:
            xi = np.random.random_integers(0, 1, n)
            state.shrink(xi, 0, lazy=True)

        # now K = ker(X) and is in standard form

        state.h = np.random.random_integers(0, 1, n)
        state.Q = np.random.random_integers(0, 7)
        state.D = 2*np.random.random_integers(0, 3, state.k)

        state.J = np.zeros((state.k, state.k))
        for a in range(state.k):
            state.J[a, a] = 2*state.D[a] % 8
            for b in range(a):
                state.J[a, b] = 4*np.random.random_integers(0, 1)
                state.J[b, a] = state.J[a, b]

        if not provide_d: return state
        else: return state, d

    # ---------------- MeasurePauli ----------------

    # Helper: if xi not in K, extend it to an affine space that does
    # Doesn't return anything, instead modifies state
    def extend(self, xi):
        S = [a for a in range(self.n) if np.dot(xi, self.Gbar[a]) % 2 == 1]
        T = [a for a in S if self.k <= a and self.k < self.n]  # zero indexing

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

        self.k += 1

    # Write a pauli as P = i^m * Z(zeta) * X(xi), m in Z_4
    # Returns the norm of the projected state Gamma = ||P_+ |K,q>||
    # If Gamma nonzero, projects the state to P_+|K,q>
    def measurePauli(self, m, zeta, xi, give_status=False):
        if not (self.n == len(zeta) and self.n == len(xi)):
            raise ValueError("States and Pauli do not have same dimension.")

        # write zeta, xi in basis of K
        vecZeta = np.zeros(self.k)
        vecXi = np.zeros(self.k)
        for a in range(self.k):
            vecZeta[a] = np.dot(self.G[a], zeta) % 2
            vecXi[a] = np.dot(self.Gbar[a], xi) % 2

        xiPrime = 0
        for a in range(self.k):
            xiPrime += vecXi[a]*self.G[a]
        xiPrime = xiPrime % 2

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
                eta[a] += self.J[a, b]*vecXi[b]/4
        eta = eta % 2

        if np.allclose(xiPrime, xi) and (w in [0, 4]):
            # gamma = np.dot(eta, self.G[:self.k]) % 2  # matches PAPER
            gamma = np.dot(eta, self.Gbar[:self.k]) % 2  # matches MATLAB
            omegaPrime = w/4
            alpha = (omegaPrime + np.dot(gamma, self.h)) % 2

            eps = self.shrink(gamma, alpha)

            if eps == "EMPTY": return (0, "EMPTY") if give_status else 0
            if eps == "SAME": return (1, "SAME") if give_status else 1
            if eps == "SUCCESS": return (2**(-1/2), "SHRINK") if give_status else 2**(-1/2)

        if np.allclose(xiPrime, xi) and w in [2, 6]:
            sigma = 2 - w/2

            # Update Q, D, J using equations 100, 101
            self.Q = (self.Q + sigma) % 8
            for a in range(self.k):
                self.D[a] = (self.D[a] - 2*sigma*eta[a]) % 8

            # ignore a != b for some reason, MATLAB code does it too
            # still satisfies J[a,a] = 2 D[a] mod 8
            self.J = (self.J + 4*np.outer(eta, eta)) % 8  # original

            # code that upholds a != b
            # for a in range(self.k):
            #     for b in range(self.k):
            #         # if (a != b):
            #         self.J[a, b] += 4*np.outer(eta, eta)[a, b]
            # self.J = self.J % 8

            return (2**(-1/2), "PERMUTE") if give_status else 2**(-1/2)

        # remaining case: xiPrime != xi
        self.extend(xi)
        newDval = (2*m + 4*(np.dot(zeta, self.h + xi) % 2)) % 8
        self.D = np.concatenate((self.D, [newDval]))

        self.J = np.bmat([[self.J, np.array([4*vecZeta]).T],
                         [np.array([4*vecZeta]), [[(4*m) % 8]]]])

        return (2**(-1/2), "EXTEND") if give_status else 2**(-1/2)
