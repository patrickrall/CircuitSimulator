from subprocess import PIPE, Popen


def printProjector(projector):
    phases, xs, zs = projector
    for g in range(len(phases)):
        phase, x, z = phases[g], xs[g], zs[g]
        tmpphase = phase
        genstring = ""
        for i in range(len(x)):
            if (x[i] == 1 and z[i] == 1):
                tmpphase -= 1  # divide phase by i
                # genstring += "(XZ)"
                genstring += "Y"
                continue
            if (x[i] == 1):
                genstring += "X"
                continue
            if (z[i] == 1):
                genstring += "Z"
                continue
            genstring += "_"
        tmpphase = tmpphase % 4
        lookup = {0: " +", 1: " i", 2: " -", 3: "-i"}
        print(lookup[tmpphase] + genstring)
    if len(phases) == 0:
        print("Projector is empty.")


def send(s):
    return (str(s) + "\n").encode()


def writeProjector(projector):
    ph, xs, zs = projector

    dat = b""

    dat += send(len(ph))  # Nstabs
    dat += send(len(xs[0]))  # Nqubits
    for i in range(len(ph)):
        dat += send(ph[i])
        for j in range(len(xs[i])):
            dat += send(xs[i][j])
            dat += send(zs[i][j])

    return dat


Gph = [1, 2, 1]
Gxs = [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
Gzs = [[1, 1, 1], [1, 0, 1], [0, 1, 0]]
G = (Gph, Gxs, Gzs)

Hph = [1, 1]
Hxs = [[1, 0, 0], [0, 1, 1]]
Hzs = [[1, 1, 1], [0, 1, 0]]
H = (Hph, Hxs, Hzs)

p = Popen("libcirc/sample", stdin=PIPE, stdout=PIPE, bufsize=1)

indat = b""

indat += send(len(Gxs[0]))  # t
indat += send(0)  # k
indat += send(0.001)  # fidbound
indat += send(0)  # exact
indat += send(1)  # rank
indat += send(0)  # fidelity

indat += writeProjector(G)
indat += writeProjector(H)

indat += send(40000)  # Nsamples
indat += send(1)  # numcores


# out = p.communicate(input=indat)
# print(out[0].decode())

p.stdin.write(indat)
p.stdin.flush()


while True:
    line = p.stdout.readline().decode()[:-1]
    if not line:
        break

    if ("Numerator:" in line or "Denominator:" in line):
        print(line, end="\r")
    else: print(line + " "*50)
