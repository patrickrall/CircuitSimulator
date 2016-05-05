########################################################################
# Tests the python stabilizer state implmentation using Matlab outputs #
########################################################################

#
# File: unittests.py
#

import numpy as np
from stabilizer import StabilizerState
import json

directory = "exampleFunctionCalls/"

# ------------------ Helper Funcs ---------------


def load(psi):
    state = StabilizerState(psi["n"], psi["k"])
    state.G = np.array(psi["G"])
    state.Gbar = np.array(psi["Gbar"])
    state.h = np.array(psi["h"])

    if "D" in psi:
        state.D = np.array(psi["D"])[:state.k]
        state.J = np.array(psi["J"])[:state.k, :state.k]*4
        state.Q = psi["Q"]

    return state


def compare(state, psi):
    if not (state.n == psi["n"]): return False, "n"
    if not (state.k == psi["k"]): return False, "k: %d != %d" % (state.k, psi["k"])
    if not np.allclose(state.G, np.array(psi["G"])): return False, "G"
    if not np.allclose(state.Gbar, np.array(psi["Gbar"])): return False, "Gbar"
    if not np.allclose(state.h, np.array(psi["h"])): return False, "h"
    if "D" in psi:
        if not state.Q == np.array(psi["Q"]): return False, "Q"
        if not np.allclose(state.D, np.array(psi["D"])[:state.k]): return False, "D"
        if not np.allclose(state.J, np.array(psi["J"])[:state.k, :state.k]*4): return False, "J"
    return True, ""

# ---------------- ExponentialSum ---------------

f = open(directory + "ExponentialSum.txt")
tests = json.loads(f.read())
tests = []

# indexcut = 267
indexcut = 0
failed = 0
index = 0
for test in tests:
    index += 1
    if index < indexcut: continue
    # don't actually depend on n, but need for initialization
    k = len(test["D"]["D"])
    # k = 4
    state = StabilizerState(k, k)
    # state.D = np.array(test["D"]["D"])[:4]
    state.D = np.array(test["D"]["D"])
    # state.J = np.array(test["J"]["J"])[:4, :4]*4
    state.J = np.array(test["J"]["J"])*4
    state.Q = test["Q"]["Q"]

    (eps, p, m) = state.exponentialSum(exact=True)
    if eps != test["eps_out"]\
            or m != test["m_out"] % 8\
            or p != test["p_out"]:
        if failed == 0: print("ExponentialSum errors:")
        print("ExponentialSum %d (k=%d) failed: (%d,%d,%d) should be (%d,%d,%d)" %
              (index, state.k, eps, p, m, test["eps_out"], test["p_out"], test["m_out"]))
        failed += 1
    # break

if len(tests) == 0: print("[\033[93mSkipped\033[0m] ExponentialSum")
else:
    if failed == 0: print("[\033[92mPassed\033[0m] ExponentialSum")
    else: print("[\033[91mFailed\033[0m] ExponentialSum: %d/%d tests failed" % (failed, len(tests)))

# -------------------- Shrink -------------------

f = open(directory + "ShrinkSmall.txt")
tests = json.loads(f.read())
tests = []

indexcut = 0
failed = 0
index = 0
for test in tests:
    index += 1
    if index < indexcut: continue

    # try:
    state = load(test["psi"]["psi"])
    # except KeyError:
    #    import pdb; pdb.set_trace()
    xi = np.array(test['xi']['xi'])
    alpha = test['alpha']

    lazy = False
    lazyString = ""
    if test['isLAZY']:
        lazy = True
        lazyString = " (lazy)"

    result = state.shrink(xi, alpha, lazy=lazy)

    statuscheck = True
    if result == test["status_out"]:
        status = result
    else:
        status = result + " != " + test["status_out"]

    good, problem = compare(state, test["psi_out"]["psi_out"])
    if not good or not statuscheck:
        if failed == 0: print("Shrink errors:")
        if problem == "": problem = status

        print("Shrink %d failed%s, %s wrong -> %s" % (index, lazyString, problem, status))
        failed += 1

    # break

if len(tests) == 0: print("[\033[93mSkipped\033[0m] Shrink")
else:
    if failed == 0: print("[\033[92mPassed\033[0m] Shrink")
    else: print("[\033[91mFailed\033[0m] Shrink: %d/%d tests failed" % (failed, len(tests)))

# ----------------- InnerProduct ----------------

f = open(directory + "InnerProduct.txt")
tests = json.loads(f.read())
tests = tests[:100]
tests = []

indexcut = 2
failed = 0
index = 0
for test in tests:
    index += 1
    if index < indexcut: continue

    state1 = load(test["state1"]["state1"])
    state2 = load(test["state2"]["state2"])

    (eps, p, m) = StabilizerState.innerProduct(state1, state2, exact=True)
    if eps != test["eps_out"]\
            or m != test["m_out"] % 8\
            or p != test["p_out"]:
        if failed == 0: print("InnerProduct errors:")
        print("InnerProduct %d failed: (%d,%d,%d) should be (%d,%d,%d)" %
              (index, eps, p, m, test["eps_out"], test["p_out"], test["m_out"]))
        failed += 1

if len(tests) == 0: print("[\033[93mSkipped\033[0m] InnerProduct")
else:
    if failed == 0: print("[\033[92mPassed\033[0m] InnerProduct")
    else: print("[\033[91mFailed\033[0m] InnerProduct: %d/%d tests failed" % (failed, len(tests)))


# -------------------- Extend -------------------    ktrue = test['psi']['psi']['k']
f = open(directory + "Extend.txt")
tests = json.loads(f.read())
tests = []

indexcut = 0
failed = 0
index = 0
for test in tests:
    index += 1
    if index < indexcut: continue

    state = load(test["K"]["K"])
    xi = np.array(test['xi']['xi'])

    state.extend(xi)
    good, problem = compare(state, test["K_out"]["K_out"])
    if not good:
        if failed == 0: print("Extend errors:")
        print("Extend %d failed, %s wrong" % (index, problem))
        failed += 1

if len(tests) == 0: print("[\033[93mSkipped\033[0m] Extend")
else:
    if failed == 0: print("[\033[92mPassed\033[0m] Extend")
    else: print("[\033[91mFailed\033[0m] Extend: %d/%d tests failed" % (failed, len(tests)))

# ----------------- MeasurePauli ----------------

f = open(directory + "MeasurePauli.txt")
tests = json.loads(f.read())
# tests = []

indexcut = 0
failed = 0
index = 0
for test in tests:
    index += 1
    if index < indexcut: continue

    state = load(test["psi"]["psi"])
    xi = np.array(test['Pauli']['Pauli']['X'])
    zeta = np.array(test['Pauli']['Pauli']['Z'])
    m = test['Pauli']['Pauli']['m']

    norm, case = state.measurePauli(m, zeta, xi, give_status=True)

    failednow = False
    good, problem = compare(state, test["psi_out"]["psi_out"])
    if not good:
        if failed == 0: print("MeasurePauli errors:")
        print("MeasurePauli %d failed (case %s), %s wrong" % (index, case, problem))
        failed += 1
        failednow = True
    if not (norm - test["norm_out"]) < 1e-5:
        if not failednow:
            if failed == 0: print("MeasurePauli errors:")
            failed += 1
            print("MeasurePauli %d failed: norm wrong" % index)

if len(tests) == 0: print("[\033[93mSkipped\033[0m] MeasurePauli")
else:
    if failed == 0: print("[\033[92mPassed\033[0m] MeasurePauli")
    else: print("[\033[91mFailed\033[0m] MeasurePauli: %d/%d tests failed" % (failed, len(tests)))

# ------------------------ Unpack ------------------------

f = open(directory + "UnpackRevised.txt")
tests = json.loads(f.read())
# tests = []

indexcut = 0
failed = 0
index = 0
for test in tests:
    index += 1
    if index < indexcut: continue

    state = load(test["output"]["psi_out"])
    psi = state.unpack()

    correct = np.array(test["unpackReals"]["root"]) + 1j*np.array(test["unpackImaginaries"]["root"])

    if len([(psi[x] - correct[x]) for x in range(len(psi)) if np.abs(psi[x] - correct[x]) > 1e-3]) > 0:
        if failed == 0: print("Unpack errors:")
        failed += 1
        print("Unpack %d failed" % index)
        for x in [x for x in range(len(psi)) if np.abs(psi[x] - correct[x]) > 1e-3]:
            print(x, np.binary_repr(x, width=state.k), np.round(psi[x], 6), correct[x])

if len(tests) == 0: print("[\033[93mSkipped\033[0m] Unpack")
else:
    if failed == 0: print("[\033[92mPassed\033[0m] Unpack")
    else: print("[\033[91mFailed\033[0m] Unpack: %d/%d tests failed" % (failed, len(tests)))
