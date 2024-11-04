#Matrix Operations -- Addition, Subtraction, Multiplication
import numpy as np

matrixA = np.array([[3,-7,-2,2], [-3,5,1,0], [6,-4,0,-5], [-9,5,-5,12]])
vectorB = np.array([[-9,5,7,11]])

vector1 = np.array([[1,2,3]])
vector2 = np.array([[4,5,6]])

matrixC = np.array([[3,-1,0], [2,5,1], [-7,1,3]])
matrixD = np.array([[6,-1,0], [0,1,-2], [3,-8,1]])

matrixTallSkinny = np.array([[3,-7], [5,1], [6,-4]])
matrixShortFat = np.array([[3,6,2], [9,1,0]])

matrixIdentity = np.array([[1,0,0], [0,1,0], [0,0,1]])

def matrix_Multiplication(matrix1, matrix2):

    """
    Multiplies two matrices (2D numpy arrays) with compatible dimensions (number columns of matrix1 must equal number of rows of matrix2).

    Parameters
    ----------
    matrix1 : np.ndarray
        First matrix to be multiplied, with shape (m,n).
    matrix2 : np.ndarray
        Second matrix to be multiplied, with shape (n,p).

    Returns
    -------
    np.ndarray
        The product of matrix1 and matrix2, with shape (m,p)

    Raises
    ------
    ValueError
        Exception thrown when the dimensions of the given arrays are not compatible.
    """

    if matrix1.shape[1] != matrix2.shape[0]:
        raise Exception("Matrices are not of proper dimensions to perform multiplication.")

    resultantMatrix = np.zeros((matrix1.shape[0], matrix2.shape[1]))

    for col in range(int(matrix1.shape[1])):

        resultantMatrix += matrix1[:, col].reshape(matrix1.shape[0],1) @ matrix2[col, :].reshape(1,matrix2.shape[1])

    return resultantMatrix

def matrix_Addition(matrix1, matrix2): 

    """
    Performs element-wise addition of two matrices (2D numpy arrays) with compatible dimensions (both matrices of same shape).

    Parameters
    ----------
    matrix1 : np.ndarray
        First matrix, with shape (m,n).
    matrix2 : np.ndarray
        Second matrix, with shape (m,n).

    Returns
    -------
    np.ndarray
        The element-wise sum of matrix1 and matrix2, with shape (m,n)

    Raises
    ------
    ValueError
        Exception thrown when the dimensions of the given arrays are not compatible.
    """

    if (matrix1.shape != matrix2.shape):
        raise ValueError("Matrices are not of proper dimensions to perform addition.")

    return matrix1 + matrix2

def matrix_Subtraction(matrix1, matrix2): 

    """
    Performs element-wise subtraction of two matrices (2D numpy arrays) with compatible dimensions (both matrices of same shape).

    Parameters
    ----------
    matrix1 : np.ndarray
        First matrix, with shape (m,n).
    matrix2 : np.ndarray
        Second matrix, with shape (m,n).

    Returns
    -------
    np.ndarray
        The element-wise difference of matrix1 and matrix2, with shape (m,n)

    Raises
    ------
    ValueError
        Exception thrown when the dimensions of the given arrays are not compatible.
    """

    if (matrix1.shape != matrix2.shape):
        raise Exception("Matrices are not of proper dimensions to perform subtraction.")

    return matrix1 - matrix2


print(matrix_Multiplication(matrixA, vectorB.T) == (matrixA @ vectorB.T))

print(matrix_Multiplication(matrixC, matrixD) == (matrixC @ matrixD))

print(matrix_Multiplication(matrixTallSkinny, matrixShortFat) == (matrixTallSkinny @ matrixShortFat))

print(matrix_Multiplication(vector1.T, vector2) == (vector1.T @ vector2))

print(matrix_Addition(matrixC, matrixD) == (matrixC + matrixD))
print(matrix_Subtraction(matrixC, matrixD) == (matrixC - matrixD))

print(matrix_Multiplication(matrixC, matrixIdentity) == matrixC)


