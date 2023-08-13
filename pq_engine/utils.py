import cupy as cp

from .exceptions import (
    NonOneTraceError,
    NotHermiteError,
    NotOneDimensionalError,
    NotSquareError,
    NotUnitaryError,
)


def check_1darray(array: cp.ndarray) -> None:
    """
    check given array is 1-dimensional

    args:
        array: cupy.ndarray
    """
    if len(array.shape) != 1:
        raise NotOneDimensionalError


def check_square(matrix: cp.ndarray) -> None:
    """
    check matrix is square

    args:
        matrix: cupy.ndarray
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise NotSquareError


def check_hermite(matrix: cp.ndarray) -> None:
    check_square(matrix)
    if not cp.allclose(matrix, cp.conj(cp.transpose(matrix)), atol=1.0e-5):
        raise NotHermiteError


def check_unitary(matrix: cp.ndarray) -> None:
    check_square(matrix)
    expected_dimension = matrix.shape[0]
    if not cp.allclose(
        cp.dot(
            matrix,
            cp.conj(cp.transpose(matrix)),
        ),
        cp.identity(expected_dimension, dtype=cp.complex64),
        atol=1.0e-5,
    ):
        raise NotUnitaryError


def check_one_trace(matrix: cp.ndarray) -> None:
    check_square(matrix)
    if not cp.allclose(cp.trace(matrix), 1, atol=1.0e-5):
        raise NonOneTraceError


def check_density(matrix: cp.ndarray) -> None:
    check_hermite(matrix)
    check_one_trace(matrix)
