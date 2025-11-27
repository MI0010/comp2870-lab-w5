import numpy as np
import gram_schmidt as gs # contains qr factor. & qr eigenvals.
import pandas as pd

######################################

# s1: formation of matrix A_e
i = -16
e = 10**i
A = np.array([[1,(1+e)],
              [(1+e),1]])
Q,R = np.zeros_like(A)

# S2: initialise the errors
id_2 = np.identity(2) # identity matrix of 2x2
Q_T = Q.T # transpose of Q
error1 = []
error2 = []
error3 = []

# s3: iterate through between this being -16 <= e <=-6
for j in range(0,11):
    # store the values of Q and R
    Q,R = gs.gram_schmidt_qr(A)
    e1 = np.linalg.norm(A-Q*R)
    error1.append(e1)
    e2 = np.linalg.norm(Q_T*Q-id_2)
    error1.append(e2)
    e3 = np.linalg.norm(R-np.triu(R))
    error1.append(e3)
    i += 1

df = pd.DataFrame(data)