import numpy as np
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

def gram_schmidt_eigen(A, maxiter=100, verbose=False):
    """
    Compute the eigenvalues and eigenvectors of a square matrix using the QR
    algorithm with classical Gram-Schmidt QR factorisation.

    This function implements the basic QR algorithm:

    1. Factorise the matrix `A` into `Q` and `R` using Gram-Schmidt QR
       factorisation.
    2. Update the matrix as:

       .. math::
           A_{k+1} = R_k Q_k

    3. Accumulate the orthonormal transformations in `V` to compute the
       eigenvectors.
    4. Iterate until `A` becomes approximately upper triangular or until the
       maximum number of iterations is reached.

    Once the iteration converges, the diagonal of `A` contains the eigenvalues,
    and the columns of `V` contain the corresponding eigenvectors.

    Parameters
    ----------
    A : numpy.ndarray
        A square 2D NumPy array of shape ``(n, n)`` representing the input
        matrix. This matrix will be **modified in place** during the
        computation.
    maxiter : int, optional
        Maximum number of QR iterations to perform. Default is 100.
    verbose : bool, optional
        If ``True``, prints intermediate matrices (`A`, `Q`, `R`, and `V`) at
        each iteration. Useful for debugging and understanding convergence.
        Default is ``False``.

    Returns
    -------
    eigenvalues : numpy.ndarray
        A 1D NumPy array of length ``n`` containing the eigenvalues of `A`.
        These are the diagonal elements of the final upper triangular matrix.
    V : numpy.ndarray
        A 2D NumPy array of shape ``(n, n)`` whose columns are the normalized
        eigenvectors corresponding to the eigenvalues.
    it : int
        The number of iterations taken by the algorithm.
    """
    # identity matrix to store eigenvectors
    V = np.eye(A.shape[0])

    if verbose:
        np.print_array(A)

    it = -1
    for it in range(maxiter):
        if verbose:
            print(f"\n\n{it=}")

        # perform factorisation
        Q, R = gram_schmidt_qr(A)
        if verbose:
            np.print_array(Q)
            np.print_array(R)

        # update A and V in place
        A = R @ Q
        V = V @ Q

        if verbose:
            np.print_array(A)
            np.print_array(V)

        # test for convergence: is A upper triangular up to tolerance 1.0e-8?
        if np.allclose(A, np.triu(A), atol=1.0e-8):
            break

    eigenvalues = np.diag(A)
    return eigenvalues, V, it

######
e = 10**-6
print(e)
# matrix = [[1,(1+e)],[(1+e),1]]

# compute the errors
# error1 = np.linalg.norm()

# loop from 10^-6 to 10^-16 
#for i in range(10**-16)

# table = tabulate(
#     # data =
#     headers=["e","error 1","error 2","error 3"]
#     tablefmt="grid"
# )