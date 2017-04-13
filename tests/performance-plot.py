import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

noapproxX = []
exactDecompX = []
LsampleX = []

noapproxY = []
exactDecompY = []
LsampleY = []

noapproxYerr = []
exactDecompYerr = []
LsampleYerr = []

L = 1000

f = open("performance.txt")
for line in f.readlines():
    try:
        data = json.loads(line)
    except:
        continue

    if data["mode"] == "noapprox":
        noapproxX.append(data["k"])
        noapproxY.append(data["mutime"])
        noapproxYerr.append(data["sigmatime"]/(data["ntests"]*data["ntests"]))
    if data["mode"] == "exactDecomp":
        exactDecompX.append(data["k"])
        exactDecompY.append(L*data["mutime"]/data["L"])
        # exactDecompYerr.append(data["sigmatime"]/(data["L"]*data["ntests"]*data["ntests"]))
        exactDecompYerr.append(L*data["sigmatime"]/(data["L"]))
    if data["mode"] == "Lsample":
        LsampleX.append(data["k"])
        LsampleY.append(L*data["mutime"]/data["L"])
        # LsampleYerr.append(data["sigmatime"]/(data["L"]*data["ntests"]*data["ntests"]))
        LsampleYerr.append(L*data["sigmatime"]/(data["L"]))

f.close()

def func(x,a,b,c):
    return a * 2**(b*x) + c


# plt.errorbar(noapproxX, noapproxY, yerr=noapproxYerr)



popt, _ = curve_fit(func, exactDecompX, exactDecompY)
plt.plot(exactDecompX, func(np.array(exactDecompX), *popt), label="exactDecomp fit")
plt.errorbar(exactDecompX, exactDecompY, yerr=exactDecompYerr, fmt="o", label="exactDecomp")

print("exactDecomp time = ", popt[0],"* 2^(", popt[1], "* k) +", popt[2])


plt.errorbar(LsampleX, LsampleY, yerr=LsampleYerr, fmt="o", label="Lsample")

plt.xlabel("k = 2t")
plt.ylabel("Time per %d samples in seconds" % L)
plt.title("Simulation time for 16 cores over two nodes")
plt.yscale("log")

plt.legend()

plt.show()
