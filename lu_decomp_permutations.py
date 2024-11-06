#LU Decomposition with Permutations
import numpy as np
import copy

matrixA = np.array([[-3,-3,-2.], [3,0,2], [-6,6,-8]], dtype=float)
vectorB = np.array([[13,-10,22]], dtype=float)

matrixC = np.array([[3,-7,-2,2], [-3,5,1,0], [6,-4,0,-5], [-9,5,-5,12]])
vectorD = np.array([[-9,5,7,11]])

matrixE = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16]])
matrixP = np.array([[0,0,0,1], [0,0,1,0], [1,0,0,0], [0,1,0,0]])
vectorP = np.array([[4], [3], [1], [2]])

def lu_factorization_pivot_anywhere(matrix):

    """
    Finds the LU decomposition of a square array that does not require permutation (the pivot element is at [0,0]).

    Parameters
    ----------
    matrix : np.ndarray
        Array to decompose with shape (n,n).

    Returns
    -------
    tuple of np.ndarray
        A tuple containing:
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

    for step in range(size):

        l[step+1:,step] = currMatrix[step+1:,step] / currMatrix[step, step]
        u[step] = currMatrix[step]
        currMatrix = np.subtract(currMatrix, l[:,step].reshape(size,1) @ u[step].reshape(1,size))

    return (l, u)

def solve_x_by_lu(matrix, given_vector):

    """
    Solves for vector x in Ax=b equation, where A is matrix with shape (m,n) and b is vector with shape (n,1).

    Parameters
    ----------
    matrix : np.ndarray
        Array with shape (m,n).
    given_vector : np.ndarray
        Array with shape (n,1).

    Returns
    -------
    np.ndarray
        The unique vector with shape (m,1) that makes the equation A @ x = b true.
    Raises
    ------
    ValueError
        Exception thrown when the dimensions of the matrix and vector are incompatible.
    """

    if matrix.shape[1] != given_vector.shape[0]:
        raise ValueError("Matrix and vector are not of proper dimensions to solve equation.")

    try: 
        matrixL, matrixU = lu_factorization_pivot_anywhere(matrix)
    except:
        print("Matrix must be square to find a unique solution for the equation.")

    #forward substitution
    vectorY = np.zeros((matrixL.shape[0], 1))
    for position in range(vectorY.shape[0]):

        vectorY[position,0] = given_vector[position,0] - matrixL[position, :position] @ vectorY[:position, 0]
    
    #backwards substitution
    vectorX = np.zeros((matrixL.shape[0], 1))
    for position in range(vectorY.shape[0]-1,-1,-1):
        
        vectorX[position,0] = (vectorY[position,0] - (matrixU[position, position:] @ vectorX[position:,0])) / matrixU[position,position]
        
    return vectorX

def permute_matrix(perm_matrix, matrix):

    if (perm_matrix.shape[1] > 1):
        return perm_matrix @ matrix
    
    for row in range(perm_matrix.shape[0]):
        oldRow = matrix[row]
        matrix[row] = matrix[perm_matrix[row,0]]
        matrix[perm_matrix[row,0]] = oldRow

    return matrix
    
def generate_perm_matrix(vector):

    perm_matrix = np.zeros((vector.shape[0], vector.shape[0]))

    for row in range(vector.shape[0]):
        perm_matrix[row,vector[row,0]-1] = 1

    return perm_matrix

#print(np.allclose(solve_x_by_lu(matrixA, vectorB.T), np.linalg.solve(matrixA, vectorB.T)))
#print(np.allclose(solve_x_by_lu(matrixC, vectorD.T), np.linalg.solve(matrixC, vectorD.T)))

#print(permute_matrix(matrixP, matrixE))
print(generate_perm_matrix(vectorP))