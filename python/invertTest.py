import numpy as np

n = 8

for i in range(10):
    A = np.random.random_integers(0,1, (n, n))
    while (np.abs(np.linalg.det(A)) != 1):
        A = np.random.random_integers(0,1, (n, n))

    Borig = np.linalg.inv(A)
    B = Borig % 2
    if not np.allclose(np.dot(A,B) % 2, np.eye(n)):
        print("A:", A.astype(int))
        print("B:", B.astype(int))
        print("I?:", np.dot(A,B).astype(int) % 2)
        print("Borig", Borig.astype(int))
        print("I", np.dot(A,Borig).astype(int))
        import pdb; pdb.set_trace()

        raise ValueError("Error!")

