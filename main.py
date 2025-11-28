import numpy as np
import gram_schmidt as gs # contains qr factor. & qr eigenvals.
import pandas as pd

######################################
e = 10**-16
i = -16

e_vals = []
error1 = []
error2 = []
error3 = []

# s3: iterate through between this being -16 <= e <=-6
for j in range(0,11):
    # s1: formation of matrix A_e
    A = np.array([[1,(1+e)],
                [(1+e),1]])
    # store the values of Q and R
    Q,R = gs.gram_schmidt_qr(A)
    id_2 = np.identity(2) # identity matrix of 2x2
    Q_T = Q.T # transpose of Q
    # put e values into list and store as a string
    e_vals.append(f"10e**{str(i)}")
    i += 1
    # error 1
    e1_inp = A - Q*R
    e1 = np.linalg.norm(e1_inp)
    error1.append(e1)
    # error 2
    e2_inp = (Q_T*Q) - id_2
    e2 = np.linalg.norm(e2_inp)
    error2.append(e2)
    # error 3
    e3_inp = R - (np.triu(A))
    e3 = np.linalg.norm(e3_inp)
    error3.append(e3)
    e = e * 10**1

# plot the data
data = {
        "e": e_vals,
        "error1": error1,
        "error2": error2,
        "error3": error3
    }
df = pd.DataFrame(data)
print(df)