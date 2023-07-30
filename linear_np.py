from typing import Callable, List, Tuple
import numpy as np

Vector = np.array
Matrix = np.array


def add(v: Vector, w: Vector) -> Vector:
    """Adds corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"
    return v + w

def subtract(v: Vector, w: Vector) -> Vector:
    """Subtracts corresponding elements"""
    assert len(v) == len(w), "vectors must be the same length"
    return v - w

def vector_sum(vectors: List[Vector]) -> Vector:
    """Sums all corresponding elements"""
    # Checks that vectors are not empty
    assert vectors, "no vectors provided!"

    # Checks the vectors are all the same size
    num_elements = len(vectors[0])
    assert all(len(v) == num_elements for v in vectors)

    return np.sum(vectors, axis=0)

def scalar_multiply(c: float, v: Vector) -> Vector:
    """Mulitples every element by c"""
    return c * v

def vector_mean(vectors: List[Vector]) -> Vector:
    """Computes the element-wise average"""
    n = len(vectors)
    return np.mean(vectors, axis=0)

def dot(v: Vector, w: Vector) -> float:
    """Computes v_i * w_i + ... + v_n * w_n"""
    assert len(v) == len(w), "vectors must be the same length"
    return np.dot(v, w)

def sum_of_squares(v: Vector) -> float:
    """Computes v_i * v_i + ... + v_n * v_n"""
    return np.sum(v * v)

def magnitude(v: Vector) -> float:
    """Returns the magnitude (or length) of v"""
    return np.sqrt(sum_of_squares(v))

def distance(v: Vector, w: Vector) -> float:
    """Computes the distance between v and w"""
    return magnitude(v - w)

def shape(A: Matrix) -> Tuple[int, int]:
    """Returns (# of rows of A, # of Columns of A"""
    return A.shape[0], A.shape[1]

def get_row(A: Matrix, i: int) -> Vector:
    """Returns the i-th row of A (as a Vector)"""
    return A[i]

def get_column(A: Matrix, j: int) -> Vector:
    """Returns the jth column of A (as a Vector)"""
    return A[:, j]

def make_matrix(num_rows: int, num_cols: int, entry_fn: Callable[[int, int], float]) -> Matrix:
    """
    Returns a num_rows x num_cols matrix
    whose (i,j)-th entry is entry_fn(i, j)
    """
    return np.array([[entry_fn(i, j) for j in range(num_cols)] for i in range(num_rows)])

def identity_matrix(n: int) -> Matrix:
    """return the n x n identity matrix"""
    return np.eye(n)
