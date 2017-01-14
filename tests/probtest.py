import numpy as np
from libcirc.probability import probability

# number of samples
logL = 3
L = 10**logL
print("Samples:", L)

# probability of failure
logpf = -0.4
pf = 10**logpf
print("Expected success probability:", (1-pf)*100, "%")

e = np.sqrt(1/(pf*L))
print("Error cutoff:", e)

# t gates
T = 4
print("T gates:", T)

# python
python = False
if python: print("Python backend")
else: print("C backend")

# calculate expected output
Tgate = np.array([[1+0j, 0+0j], [0+0j, np.exp(1j*np.pi/4)]])
Hgate = np.array([[1+0j, 1+0j], [1+0j, -1+0j]])/np.sqrt(2)

State = np.dot(Hgate, np.array([1+0j, 0+0j]))
for i in range(T):
    State = np.dot(Tgate, State)
    State = np.dot(Hgate, State)

target = np.abs(State[0])**2
print("Target output:", target)

# prepare circuit
circ = "H\n"
for i in range(T): circ += "T\nH\n"

passed = 0
errors = []
ntests = int(np.ceil(10**(-logpf+1.5)))
print("Running", ntests,"tests...")
for i in range(ntests):
    p = probability(circ, {0:0}, samples=L, config={"python":python})
    errors.append(np.abs(target - p))
    if np.abs(target - p) < e: passed += 1

print(passed, "passed. Success probability = ", (passed/ntests)*100, "%")
if ((1-pf) < passed/ntests): print("PASSED")
else: print("FAILED")
print("Average error:", np.mean(errors))
