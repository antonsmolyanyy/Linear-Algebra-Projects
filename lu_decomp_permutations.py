#LU Decomposition with Permutations
import numpy as np
import copy
import scipy

matrixA = np.array([[2,4,2], [4,-10,2], [1,2,4]])
vectorB = np.array([[13,-10,22]], dtype=float)

matrixC = np.array([[0,1,1], [1,3,7], [2,4,8]])
vectorD = np.array([[-9,5,7,11]])

matrixE = np.array([[2, 5, 8, 7], [5, 2, 2, 8], [7, 5, 6, 6], [5, 4, 4, 8]])
matrixP = np.array([[0,0,0,1], [0,0,1,0], [1,0,0,0], [0,1,0,0]])
vectorP = np.array([[4,3,1,2]])

def lu_factorization_pivot_anywhere(matrix):

    """
    Finds the LU factorization of a square array that may or may not require permutations.

    Parameters
    ----------
    matrix : np.ndarray
        Array to factor with shape (n,n).

    Returns
    -------
    tuple of np.ndarray
        A tuple containing:
        - total_perm_vector (nd.ndarray): vector that holds the permutation information as row indices (shape 1,n)
        - l (np.ndarray): Lower triangular array with 1's on diagonal (shape nxn).
        - u (np.ndarray): Upper triangular array (shape nxn).

    Raises
    ------
    ValueError
        Exception thrown when the array is not square.
    """

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Cannot find LU factorization for nonsquare matrix.")

    size = int(matrix.shape[0])
    currMatrix = copy.deepcopy(matrix)

    l = np.identity(size)
    u = np.zeros((size, size))
    total_perm_vector = np.arange(0, size).reshape(1,size)

    for step in range(size):

        #finds index of largest pivot in current column
        index_largest = np.argmax(abs(currMatrix[step:, step])) + step

        #if largest pivot is not in the current row, switch that row with step row
        if (index_largest != step):
            total_perm_vector[0,[step, index_largest]] = total_perm_vector[0,[index_largest, step]] 
            currMatrix[[index_largest, step], :] = currMatrix[[step, index_largest], :]

        l[step+1:,step] = currMatrix[step+1:,step] / currMatrix[step, step]
        u[step] = currMatrix[step]
        currMatrix = np.subtract(currMatrix, l[:,step].reshape(size,1) @ u[step].reshape(1,size))

    return (total_perm_vector,l,u)

print(np.allclose(scipy.linalg.lu(matrixC, p_indices=True)[0], lu_factorization_pivot_anywhere(matrixC)[0]))
print(np.allclose(scipy.linalg.lu(matrixC, p_indices=True)[1], lu_factorization_pivot_anywhere(matrixC)[1]))
print(np.allclose(scipy.linalg.lu(matrixC, p_indices=True)[2], lu_factorization_pivot_anywhere(matrixC)[2]))