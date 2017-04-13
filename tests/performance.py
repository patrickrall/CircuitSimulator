import numpy as np
import os
import sys
import json
sys.path.append(os.path.abspath("."))
from datetime import datetime

from libcirc.probability import probability

# todoK  = [2,3,4,5,6,7,8,9]
# todoK  = [2,3,4,5]
todoK  = [1,2,3,4,5]
# todoK  = [9]
# todoK = [1]
ntests = 300

# parameters
# mode = "noapprox"
# mode = "exactDecomp"
# mode = "Lsample"
modes = ["Lsample"]
# modes = ["noapprox"]

L = 1000

fout = open("performance.txt", "a")

for mode in modes:
    print("mode", mode)
    for k in todoK:
        print("k =", k, "\n")
        # choose t
        if mode == "noapprox":
            # decomposition size: 2^(k/2) * 2^(k/2)
            # = 2^(t/2) * 2^(t/2)
            t = k
            if (t % 2 != 0):
                print("Can't run with exactly k=%d on noapprox" % t)
                continue

            config = {"noapprox": True}
        else:
            t = 2*k

        if mode == "exactDecomp":
            config = {"samples": L}
        if mode == "Lsample":
            config = {"exact": False, "k":k, "samples": L, "rank":False}

        config["mpirun"] = "/usr/bin/mpirun --hostfile /home/prall/.mpi_hostfile"
        config["procs"] = 16

        # build circuit:
        circ = "H\n"
        for i in range(t): circ += "T\nH\n"

        times = []
        outs = []
        for i in range(ntests):
            if (i > 0):
                print("Test: ", i+1, "/", ntests, "(remaining", np.round((ntests-i)*np.mean(times)/60,2), " min)", end="\r")
            starttime = datetime.now()
            out = probability(circ, {0:0}, config=config)
            dt = (datetime.now() - starttime)
            times.append(dt.seconds + dt.microseconds/1e6)
            outs.append(out)

        data = {"k":k, "t":t, "mode": mode, "L": L, "ntests": ntests,
                "mutime": np.mean(times), "sigmatime": np.std(times),
                "muout": np.mean(outs), "sigmaout": np.std(outs)}
        fout.write(json.dumps(data) + "\n")
        fout.flush()

fout.close()
