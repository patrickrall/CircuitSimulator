import json
import numpy as np

fin = open("../python/stabilizer/exampleFunctionCalls/MeasurePauli.txt")
data = json.loads(fin.read())

fout = open("measurePauliTests.txt", "w")


#for test in data:
#print(len(data))
for test in data:
    def mul(val):
        return (np.array(val)*4).tolist()

    psi = test['psi']['psi']
    outpsi = test['psi_out']['psi_out']
    pauli = test['Pauli']['Pauli']
    n = psi['n']
    k = psi['k']

    def parse(val, pad):
        # val = np.array(val)
        # shape = list(val.shape)
        # for idx, _ in enumerate(shape): shape[idx] = pad
        # padval = np.zeros(tuple(shape))
        # for i, v in np.ndenumerate(val):
        #     if i in padval: padval[i] = v
        # print(padval.shape)
        # padval = padval.astype(int).tolist()

        return str(val).replace(" ", "").replace("[", "").replace("]", "")

    fout.write(str(psi['n']) + "\n")  # n
    fout.write(str(psi['k']) + "\n")  # k
    fout.write(str(psi['Q']) + "\n")  # Q
    fout.write(str(pauli['m']) + "\n")  # m
    fout.write(str(parse(psi['h'],n)) + "\n")  # h
    fout.write(str(parse(psi['D'],k)) + "\n")  # D
    fout.write(str(parse(pauli['Z'],n)) + "\n")  # zeta
    fout.write(str(parse(pauli['X'],n)) + "\n")  # xi
    fout.write(str(parse(psi['G'],n)) + "\n")  # G
    fout.write(str(parse(psi['Gbar'],n)) + "\n")  # Gbar
    fout.write(str(parse(mul(psi['J']),k)) + "\n")  # J

    fout.write(str(test['norm_out']) + "\n")  # out

    n = outpsi['n']
    k = outpsi['k']

    # import pdb; pdb.set_trace()
    fout.write(str(outpsi['k']) + "\n")  # outk
    fout.write(str(outpsi['Q']) + "\n")  # outQ
    fout.write(str(parse(outpsi['h'],n)) + "\n")  # outh
    fout.write(str(parse(outpsi['D'],k)) + "\n")  # outD
    fout.write(str(parse(outpsi['G'],n)) + "\n")  # outG
    fout.write(str(parse(outpsi['Gbar'],n)) + "\n")  # outGbar
    fout.write(str(parse(mul(outpsi['J']),k)) + "\n")  # outJ

    # fout.write(str(test['eps_out']) + "\n")  # outEps
    # fout.write(str(test['p_out']) + "\n")  # outP
    # fout.write(str(test['m_out']) + "\n")  # outM

fout.close()
