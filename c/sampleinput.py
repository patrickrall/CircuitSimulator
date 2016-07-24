from subprocess import PIPE, Popen
import os
import sys
from threading import Thread
from queue import Queue, Empty
import time


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


def send(p, s):
    print(s)
    p.stdin.write((str(s) + "\n").encode())


def writeProjector(p, projector):
    ph, xs, zs = projector

    send(p, len(ph))  # Nstabs
    send(p, len(xs[0]))  # Nqubits
    for i in range(len(ph)):
        send(p, ph[i])
        for j in range(len(xs[i])):
            send(p, xs[i][j])
            send(p, zs[i][j])


Gph = [1, 2, 1]
Gxs = [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
Gzs = [[1, 1, 1], [1, 0, 1], [0, 1, 0]]
G = (Gph, Gxs, Gzs)

Hph = [1, 1]
Hxs = [[1, 0, 0], [0, 1, 1]]
Hzs = [[1, 1, 1], [0, 1, 0]]
H = (Hph, Hxs, Hzs)


def enqueue_output(p, queue):
    data = p.stdout.read(1).decode()
    while data != "":
        queue.put(data)
        p.stdout.flush()
        data = p.stdout.readline(1).decode()
    p.communicate()

# p = Popen(os.getcwd()+"/sample", stdin=PIPE, stdout=PIPE, bufsize=1)
p = Popen("gdb "+os.getcwd()+"/sample", shell=True, stdin=PIPE, stdout=PIPE, bufsize=1)
q = Queue()
t = Thread(target=enqueue_output, args=(p, q))
t.daemon = True
t.start()

time.sleep(0.1)

p.stdout.flush()
while True:
    try: line = q.get_nowait()
    except Empty: break
    else:
        sys.stdout.write(line)

while True:
    sys.stdout.write("py: ")
    line = input()

    if line == "make":
        sys.stdout.write("Python: Don't do that\n")
        send(p, "")  # t
    elif line == "run":
        time.sleep(0.1)
        send(p, "run")  # t
        send(p, len(Gxs[0]))  # t
        send(p, 0)  # k
        send(p, 0.001)  # fidbound
        send(p, 0)  # exact
        send(p, 1)  # rank
        send(p, 0)  # fidelity

        writeProjector(p, G)
        writeProjector(p, H)

        send(p, 1000)  # Nsamples
        send(p, 0)  # parallel
    else:
        send(p, line)

    p.stdin.flush()
    time.sleep(0.1)

    p.stdout.flush()
    while True:
        try: line = q.get_nowait()
        except Empty: break
        else:
            sys.stdout.write(line)

print(p.communicate()[0].decode())
