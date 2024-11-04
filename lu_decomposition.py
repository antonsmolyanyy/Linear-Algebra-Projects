#LU Decomposition
import numpy as np
import copy
import scipy

matrixA = np.array([[3,-7,-2,2], [-3,5,1,0], [6,-4,0,-5], [-9,5,-5,12]], dtype=float)
vectorB = np.array([[-9,5,7,11]])

vector1 = np.array([[1,2,3]])
vector2 = np.array([[4,5,6]])

matrixC = np.array([[3,-1,0], [2,5,1], [-7,1,3]])
matrixD = np.array([[6,-1,0], [0,1,-2], [3,-8,1]])
matrixF = np.array([[3,-6,-3], [2,0,6], [-4,7,4]])

matrixE = np.array([[1,2,3], [2,5,7], [2,7,8]])

matrixTallSkinny = np.array([[3,-7], [5,1], [6,-4]])
matrixShortFat = np.array([[3,6,2], [9,1,0]])

matrixIdentity = np.array([[1,0,0], [0,1,0], [0,0,1]])
matrixP = np.array([[0,1,0], [1,0,0], [0,0,1]])

def lu_factorization_pivot_at_00(matrix):

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
        currMatrix -= l[:,step].reshape(size,1) @ u[step].reshape(1,size)

    return (l, u)

def solve_x_by_lu(matrix, vector):

    matrixL, matrixU = lu_factorization_pivot_at_00(matrix)
    #matrixU = lu_factorization_pivot_at_00(matrix)[1]

    #vectorC = np.array([[1] + [0] * (int(matrixL.shape[0]) - 1)])

    #forward substitution
    vectorC = np.zeros((1, matrixL.shape[0]))

    for vectorPos in range(0, int(vectorC.shape[0])):

        vectorC[vectorPos] = vector[vectorPos]
        for i in range(vectorPos,0):
            vectorC[vectorPos] -= matrixL[vectorPos, i] * vector[vectorPos - i]

    #backwards substitution
    vectorX = np.zeros((1, matrixL.shape[0]))
    vectorX[0, len(vectorX) - 1] = vectorC[0, len(vectorC) - 1] / matrixU[len(matrixU), len(matrixU)] 

    for vectorPos in range(int(vectorC.shape[0]) - 1, 0):

        vectorX[vectorPos] = vectorC[vectorPos] 
        for i in range(0,vectorPos):
            vectorC[vectorPos] -= matrixL[vectorPos, i] * vector[vectorPos - i]


print(lu_factorization_pivot_at_00(matrixA)[0])
print(lu_factorization_pivot_at_00(matrixA)[1])

#print(scipy.linalg.lu(matrixA))
#print(lu_factorization(matrixNonSquare1))

#print(matrixC)
#print(matrix_Multiplication(matrixC, matrixIdentity))
#print(matrix_Multiplication(matrixC, matrixP))


