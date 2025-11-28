import numpy as np
import pandas as pd
from tabulate import tabulate

def gram_schmidt_qr(A):
    """
    Compute the QR factorisation of a square matrix using the classical
    Gram-Schmidt process.

    Parameters
    ----------
    A : numpy.ndarray
        A square 2D NumPy array of shape ``(n, n)`` representing the input
        matrix.

    Returns
    -------
    Q : numpy.ndarray
        Orthonormal matrix of shape ``(n, n)`` where the columns form an
        orthonormal basis for the column space of A.
    R : numpy.ndarray
        Upper triangular matrix of shape ``(n, n)``.
    """
    n, m = A.shape
    if n != m:
        raise ValueError(f"the matrix A is not square, {A.shape=}")

    Q = np.empty_like(A)
    R = np.zeros_like(A)

    for j in range(n):
        # Start with the j-th column of A
        u = A[:, j].copy()

        # Orthogonalize against previous q vectors
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], A[:, j])  # projection coefficient
            u -= R[i, j] * Q[:, i]  # subtract the projection

        # Normalize u to get q_j
        R[j, j] = np.linalg.norm(u)
        Q[:, j] = u / R[j, j]

    return Q, R

######################################
e = 10**-16
i = -16

e_vals = []
error1 = []
error2 = []
error3 = []

# s3: iterate through between this being -16 <= e <=-6
for j in range(0,11):
    # formation of matrix A_e
    A = np.array([[1,(1+e)],
                [(1+e),1]])
    # store the values of Q and R
    Q,R = gram_schmidt_qr(A)
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
# UNCOMMENT LINE BELOW IF TABULATE DOESN'T WORK
#print(df)

# TABULATE (requires tabulate to be installed) for data visualisation
# simple table - some values are shortened (in the terminal)
#print(tabulate(df,headers="keys",tablefmt="simple_grid"))
# shows the accurate numbers (in the terminal)
print(tabulate(df,headers="keys",tablefmt="pretty"))