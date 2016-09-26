#
# file: compilecirc.py
# input a file with a list of gates, not necessarily in standard set
# output compiled circuit with all gates in standard set
#


# everything after a '#' is removed. All whitespace is removed too.
def removeComments(data):
    output = ""
    for line in data.splitlines():
        loc = line.find("#")
        if loc == -1:
            outLine = line.replace(" ", "")
        else:
            outLine = line[:loc].replace(" ", "")

        if len(outLine) > 0:
            output += outLine + "\n"

    return output


# gates that do not require references
gateSet = ["H", "X", "S", "T", "CX"]


def compileRawCircuit(raw, reference):
    # collect dependencies, verify number of qubits
    dependencies = []
    n = 0
    lineNr = 0
    for line in raw.splitlines():
        lineNr += 1

        # skip empty lines
        if line == "": continue

        # check correct number of qubits
        noLowerLen = len([x for x in line if not x.islower()])
        if n == 0: n = noLowerLen
        elif n != noLowerLen:
            raise SyntaxError("Line %d: Bad number of qubits (expected %d)" % (lineNr, n))

        # append dependency
        gate = standardGate(line)
        if gate not in gateSet:
            dependencies.append(gate)
            if gate not in reference:
                raise SyntaxError("Line %d: Undefined gate %s" % (lineNr, gate))

    # complete dependency list with dependencies of dependencies
    def subDepends(dependencies):
        if len(dependencies) == 0: return []
        newDepends = []
        for depend in dependencies:
            for subdepend in reference[depend]["dependencies"]:
                newDepends.append(subdepend)
        newDepends += subDepends(newDepends)
        return newDepends

    dependencies += subDepends(dependencies)

    # count dependency types
    dependencyCounts = {}
    for depend in dependencies:
        if depend not in dependencyCounts:
            dependencyCounts[depend] = 1
        else: dependencyCounts[depend] += 1

    # count ancillas
    ancillas = 0
    for depend in dependencyCounts.keys():
        reused = len(reference[depend]["reused"])
        needed = reference[depend]["n"] - len(standardGate(depend, listMode=True)) - reused
        ancillas += needed*dependencyCounts[depend] + reused

    totalqubits = n + ancillas

    unA = range(n, n+ancillas)  # unused ancillas
    dU = {}  # what ancillas belong to what dependencies
    for depend in dependencyCounts.keys():
        dU[depend] = []

    # recursive circuit builder
    def buildCircuit(line, qubits, unA, dU, total):
        lineList = standardGate(line, lineMode=True)
        gate = standardGate(line)

        # skip reuse markers
        if (gate == "R"): return "", unA, dU

        # standard gates
        if gate in gateSet:
            output = list(total * "_" + "\n")
            for idx in range(len(qubits)):
                output[qubits[idx]] = lineList[idx]
            return "".join(output), unA, dU

        # need reference
        # assign target qubits
        targets = []
        for char in standardGate(line, listMode=True):
            loc = lineList.index(char)
            targets.append(qubits[loc])
            lineList[loc] = " "

        # allocate qubits for reused
        reused = reference[gate]["reused"]
        while len(reused) > len(dU[gate]):
            dU[gate].append(unA[0])
            unA = unA[1:]

        # find locations for ancillas
        for bit in range(len(gate), reference[gate]["n"]):
            if bit in reused:
                reusedLoc = reused.index(bit)
                # reused qubits are first ones
                targets.append(dU[gate][reusedLoc])
                continue

            targets.append(unA[0])
            dU[gate].append(unA[0])
            unA = unA[1:]

        # build output
        output = ""
        for step in reference[gate]["steps"]:
            out, unA, dU = buildCircuit(step, targets, unA, dU, total)

            # Measurement case
            if "M" in lineList:
                Mloc = lineList.index("M")
                outold = out
                out = ""
                for line in outold.splitlines():
                    line = list(line)
                    line[qubits[Mloc]] = "M"
                    out += "".join(line) + "\n"

            output += out

        return output, unA, dU

    # build circuit
    output = ""
    lineNr = 0
    for line in raw.splitlines():
        lineNr += 1

        # skip empty lines
        if line == "": continue

        out, unA, dU = buildCircuit(line, range(n), unA, dU, totalqubits)

        output += out

    return output


# bring a line into standard form
# Normal mode: e.g. 'MTi_C' to 'CTi'
# List mode: 'MTi_C' to ['C','Ti']
# Line mode: 'MTi_C' to ['_', 'Ti', '_', 'C']
def standardGate(line, listMode=False, lineMode=False):
    if not lineMode:
        toRemove = ["_", "M"]
        for r in toRemove:
            line = line.replace(r, "")

    elems = []
    for letter in line:
        if not letter.islower(): elems.append(letter)
        else: elems = elems[:-1] + [elems[-1] + letter]

    if lineMode: return elems
    if listMode: return sorted(elems)
    return "".join(sorted(elems))


def parseReference(data):
    output = {}

    current = ""
    lineNr = 0
    n = 0
    for line in data.splitlines():
        lineNr += 1

        # heading
        if ":" in line:
            current = line[:-1]

            # already defined
            if current in output:
                raise SyntaxError("Line %d: Header %s already defined" % (lineNr, current))

            # prepare state
            n = 0
            output[current] = {"steps": [], "dependencies": [], "reused": [], "n": 0}
            continue

        # skip empty lines
        if line == "": continue

        # no header
        if current == "":
            raise SyntaxError("Line %d: No header" % lineNr)

        # number of non-lowercase characters
        noLowerLen = len([x for x in line if not x.islower()])
        if n == 0:
            n = noLowerLen
            output[current]["n"] = n

        elif n != noLowerLen:
            raise SyntaxError("Line %d: Bad number of qubits (expected %d)" % (lineNr, n))

        # identify dependency
        gate = standardGate(line)
        if gate not in (gateSet + ["R"]):
            # note: dependencies appear in list as many times are they are used
            output[current]["dependencies"].append(gate)

        # append
        output[current]["steps"].append(line)

    # verify dependencies
    okDepends = []
    for key in output.keys():
        for depend in output[key]["dependencies"]:
            if depend in okDepends: continue

            def checkDepend(history):
                if history[-1] not in output:
                    raise SyntaxError("Undefined gate: %s" % history[-1])

                for nextDepend in output[history[-1]]["dependencies"]:
                    if nextDepend in history:
                        raise SyntaxError("Dependency loop: %s -> %s" % (str(history), nextDepend))
                    checkDepend(history + [nextDepend])

            checkDepend([depend])
            okDepends.append(depend)

    # verify measurements
    for key in output.keys():
        measured = []
        for line in output[key]["steps"]:
            for m in measured:
                if line[m] not in ["_", "M"]:
                    raise SyntaxError("In %s: cannot perform operation on measured qubit %d on line %s" % (key, m, line))

            idx = line.find("M")
            if idx == -1: continue
            if idx < len(key):
                raise SyntaxError("In %s: cannot measure non-ancilla qubit on line %s" % (key, line))
            measured.append(idx)

    # verify reuse
    for key in output.keys():
        reused = []
        for line in output[key]["steps"]:
            for r in reused:
                if line[r] != "_":
                    raise SyntaxError("In %s: cannot perform operation on reused qubit %d on line %s" % (key, m, line))

            idx = line.find("R")
            if idx == -1: continue
            if idx < len(key):
                raise SyntaxError("In %s: cannot reuse non-ancilla qubit on line %s" % (key, line))
            reused.append(idx)
        output[key]["reused"] = reused

    return output  # done
