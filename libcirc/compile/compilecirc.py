#
# file: compilecirc.py
# input a file with a list of gates, not necessarily in standard set
# output compiled circuit with all gates in standard set
#

import os.path
import sys


# convenience handle
def compileCircuit(raw=None, fname=None, name="main"):
    if raw is None and fname is None:
        raise ValueError("Must specify either a file or raw circuit data.")

    try:
        if raw is None:
            f = open(fname)
            data = f.read()
            f.close()

            path = os.path.dirname(fname)
            symbols = circuitSymbols(data, path, os.path.basename(fname), [])
        if fname is None:
            symbols = circuitSymbols(raw, "", "<raw input>", [])

        return compile(symbols, name=name)
    except SyntaxError as e:
        sys.exit(e)


# COMMENT PARSING:
# Everything after a '//' is removed.
# Everything between "/*" and  "*/"
# All whitespace is removed too.
def commentParse(line, blockcomment, eRef):

    loc = line.find("//")
    if loc is not -1:
        line = line[:loc]

    loc = line.find("*/")
    if loc is not -1:
        if blockcomment:
            line = line[loc+2:]
            blockcomment = False
        else:
            raise SyntaxError(eRef + "Unexpected '*/'.")

    loc = line.find("/*")

    if blockcomment:
        if loc is not -1:
            raise SyntaxError(eRef + "Unexpected '/*' inside block comment.")
        return "", blockcomment

    if loc is not -1:
        line = line[:loc]
        blockcomment = True

    loc = line.find("import")
    if loc is -1:
        line = line.replace(" ", "")

    return line, blockcomment


# gates that do not require references
gateSet = ["H", "X", "S", "T", "CX"]


# extracts dictionary of symbols
# recursively extracts data from imported files
def circuitSymbols(data, path, fname, past):
    path = os.path.normpath(path)

    if os.path.join(path, fname) in past:
        pastpath = past + [os.path.join(path, fname)]
        raise SyntaxError("Dependency loop detected: " + " -> ".join(pastpath))

    output = {}

    blockcomment = False
    current = ""
    lineNr = 0
    n = 0
    for line in data.splitlines():
        lineNr += 1
        eRef = "Error at " + os.path.join(path, fname) + ":" + str(lineNr) + " - "

        line, blockcomment = commentParse(line, blockcomment, eRef)
        if len(line) == 0: continue

        # import statements
        if "import" in line:
            statement = line.split()
            if statement[0] == "import":
                toimport = []
            elif statement[0] == "from" and statement[2] == "import":
                toimport = statement[3:]
            else: raise SyntaxError(eRef + "Invalid import statement.")

            importfile = os.path.join(path, statement[1])

            try:
                f = open(importfile)
                data = f.read()
                f.close()
            except FileNotFoundError:
                raise SyntaxError(eRef + "File %s not found." % importfile)

            importpath = os.path.dirname(importfile)
            importpast = past + [os.path.join(path, fname)]
            symbols = circuitSymbols(data, importpath, os.path.basename(importfile), importpast)

            imported = []
            depimport = []
            for symbol in symbols.keys():
                if len(toimport) > 0 and symbol not in toimport: continue

                if symbol in output and symbol not in depimport:
                    raise SyntaxError(eRef + "Can't import %s because it is already defined!" % symbol)

                output[symbol] = symbols[symbol]
                imported.append(symbol)

                # grab dependencies of imported symbol
                for depend in symbols[symbol]["dependencies"]:
                    if depend not in output:
                        output[depend] = symbols[depend]
                        depimport.append(depend)
                        if depend in toimport:
                            imported.append(depend)

            if len(toimport) > 0 and len(toimport) != len(imported):
                failed = ["'"+x+"'" for x in toimport if x not in imported]
                raise SyntaxError(eRef + "Failed to import symbol(s): " + " ".join(failed))

            continue

        # heading
        if ":" in line:
            current = line[:-1]

            if current != standardGate(current):
                raise SyntaxError(eRef + "Symbol %s is not in standard form (expected %s)." % (current, standardGate(current)))

            if current in gateSet:
                raise SyntaxError(eRef + "Symbol %s is in gate set." % current)

            # already defined
            if current in output:
                raise SyntaxError(eRef + "Symbol %s already defined." % current)

            # prepare state
            n = 0
            output[current] = {"steps": [], "dependencies": [], "reused": [], "n": 0}
            continue

        # no header
        if current == "":
            raise SyntaxError(eRef + "Expected function declaration.")

        # number of non-lowercase characters
        noLowerLen = len([x for x in line if not x.islower() and not x.isdigit()])
        if n == 0:
            n = noLowerLen
            output[current]["n"] = n

        elif n != noLowerLen:
            raise SyntaxError(eRef + "Bad number of qubits (expected %d)." % n)

        # identify dependency
        gate = standardGate(line)
        if gate not in (gateSet + ["R"]):
            # note: dependencies appear in list as many times are they are used
            output[current]["dependencies"].append(gate)

        # append
        output[current]["steps"].append(line)

    if blockcomment:
        raise SyntaxError(eRef + "Expected '*/' before end of file.")

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


# bring a line into standard form
# Normal mode: e.g. 'MTi_C' to 'CTi'
# List mode: 'MTi_C' to ['C','Ti']
# Line mode: 'MTi_C' to ['_', 'Ti', '_', 'C']
def standardGate(line, listMode=False, lineMode=False):
    if line == "main":
        if listMode or lineMode:
            raise ValueError("Can't bring 'main' into list form.")
        return "main"

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


def compile(symbols, name="main"):
    if name not in symbols:
        raise SyntaxError("Could not find symbol %s to compile" % name)

    # collect dependencies, verify number of qubits
    n = symbols[name]["n"]
    dependencies = symbols[name]["dependencies"]
    steps = symbols[name]["steps"]

    # complete dependency list with dependencies of dependencies
    def subDepends(dependencies):
        if len(dependencies) == 0: return []
        newDepends = []
        for depend in dependencies:
            for subdepend in symbols[depend]["dependencies"]:
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
        reused = len(symbols[depend]["reused"])
        needed = symbols[depend]["n"] - len(standardGate(depend, listMode=True)) - reused
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

        # need symbols
        # assign target qubits
        targets = []
        for char in standardGate(line, listMode=True):
            loc = lineList.index(char)
            targets.append(qubits[loc])
            lineList[loc] = " "

        # allocate qubits for reused
        reused = symbols[gate]["reused"]
        while len(reused) > len(dU[gate]):
            dU[gate].append(unA[0])
            unA = unA[1:]

        # find locations for ancillas
        for bit in range(len(gate), symbols[gate]["n"]):
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
        for step in symbols[gate]["steps"]:
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
    for line in steps:
        lineNr += 1

        out, unA, dU = buildCircuit(line, range(n), unA, dU, totalqubits)

        output += out

    return output
