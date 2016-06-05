from stabilizer.stabilizer import StabilizerState
import numpy as np
from copy import deepcopy

phi = StabilizerState(2, 2)
phi.Q = 1
phi.J = np.array([[0, 4], [4, 0]])
print(phi.unpack())
print(phi.k)
print(phi.unpack()*2**(phi.k/2))
import pdb; pdb.set_trace()

theta = StabilizerState(2, 2)
theta.measurePauli(0, np.zeros(2), np.array([1, 1]))  # measure XX
theta.measurePauli(0, np.array([1, 1]), np.zeros(2))
print(theta.unpack())
print("sum", np.exp(-1j*np.pi/4)*(phi.unpack() + np.exp(-1j*np.pi/4)*theta.unpack())/2)


phi2 = deepcopy(phi)
theta2 = deepcopy(theta)

# phifactor = 1
# phifactor *= phi2.measurePauli(0, np.array([1, 1]), np.array([0, 0]))  # +ZZ
# phifactor *= phi2.measurePauli(1, np.array([1, 0]), np.array([1, 1]))  # +YX = i(XZ)(X)

# thetafactor = 1
# thetafactor *= theta2.measurePauli(0, np.array([1, 1]), np.array([0, 0]))  # +ZZ
# thetafactor *= theta2.measurePauli(1, np.array([1, 0]), np.array([1, 1]))  # +YX = i(XZ)(X)

p = np.exp(1j*np.pi/4)
s = 0

res = 1
for x in [True, False]:
    n = 1000
    meanproj = 0
    for i in range(n):
        xi = StabilizerState.randomStabilizerState(2)
        proj = 1
        if x: proj *= xi.measurePauli(0, np.array([1, 1]), np.array([0, 0]))  # +ZZ
        proj *= xi.measurePauli(1, np.array([1, 0]), np.array([1, 1]))  # +YX = i(XZ)(X)
        meanproj += proj

        s += proj * StabilizerState.innerProduct(xi, phi)
        s += proj * p * StabilizerState.innerProduct(xi, theta)

    s = s/n
    r = 4 * np.abs(s/2)**2
    print(r)
    if x: res *= r
    else: res /= r

print(res)
