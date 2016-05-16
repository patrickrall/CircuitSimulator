from stabilizer.stabilizer import StabilizerState
import numpy as np


# np.random.seed(2)

n = 2**np.random.choice(5)
k = np.random.choice(n)

vecs = []
poss = []
for i in range(k):
    while True:
        v = np.zeros(n)
        pos = np.random.choice(n)
        v[pos] = 1

        found = False
        for w in vecs:
            if np.allclose(w,v):
                found = True
                break
        if found: continue

        vecs.append(v)
        poss.append(pos)
        break

vecs = np.array(vecs)
print(vecs)
# raise ValueError("test")

# create using matrix inversion
phi = StabilizerState(n, k)

for i in range(k):
    phi.G[i] = vecs[i]

count = 0
while np.abs(np.linalg.det(phi.G)) != 1:
    count +=1
    if count > 1e3:
        print(phi.G)
        raise ValueError("Can't Make Singular")

    # randomly sample remaining rows
    for i in range(k, n):
        phi.G[i] = np.random.random_integers(0, 1, (n))

phi.Gbar = np.linalg.inv(phi.G).T % 2
if not np.allclose(np.dot(phi.G, phi.Gbar.T) % 2, np.eye(n)):
    print(phi.G)
    print(phi.Gbar.T)
    raise ValueError("bad inverse")


# create using shrink
theta = StabilizerState(n, n)

for i in range(n):
    vec = np.zeros(n)
    vec[i] = 1
    if i in poss:
        # project onto X
        _, status = theta.measurePauli(0, np.zeros(n), vec, give_status=True)
    else:
        # project onto Z
        _, status = theta.measurePauli(0, vec, np.zeros(n), give_status=True)
    print(status)

# compare
# import pdb; pdb.set_trace()
def printState(x):
    outstring = "["
    for comp in x:
        outstring += " %+0.3f + %+0.3fi, " % (comp.real, comp.imag)
    outstring += "]"
    print(outstring)


# printState(phi.unpack())
# printState(theta.unpack())
print("n, k:", n, k)
print("theta k:", theta.k)
print("Inner prod:", np.abs(StabilizerState.innerProduct(theta, phi))**2)



