from .exceptions import (
    NonOneTraceError,
    NotHermiteError,
    NotSquareError,
    NotUnitaryError,
)
from .settings import atol, xp


def check_square(matrix: xp.ndarray) -> None:
    """
    check matrix is square

    args:
        matrix:  xp.ndarray
            target matrix
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise NotSquareError


def check_hermite(matrix: xp.ndarray) -> None:
    """
    check matrix is hermite

    args:
        matrix:  xp.ndarray
            target matrix
    """
    check_square(matrix)
    if not xp.allclose(matrix, xp.conj(xp.transpose(matrix)), atol=atol):
        raise NotHermiteError


def check_unitary(matrix: xp.ndarray) -> None:
    """
    check matrix is unitary

    args:
        matrix:  xp.ndarray
            target matrix
    """
    check_square(matrix)
    expected_dimension = matrix.shape[0]
    if not xp.allclose(
        xp.dot(
            matrix,
            xp.conj(xp.transpose(matrix)),
        ),
        xp.identity(expected_dimension, dtype=xp.complex64),
        atol=1.0e-5,
    ):
        raise NotUnitaryError


def check_one_trace(matrix: xp.ndarray) -> None:
    """
    check trace of matrix is one

    args:
        matrix:  xp.ndarray
            target matrix
    """
    check_square(matrix)
    if not xp.allclose(xp.trace(matrix), 1, atol=1.0e-5):
        raise NonOneTraceError


def check_density(matrix: xp.ndarray) -> None:
    """
    check matrix is density

    args:
        matrix:  xp.ndarray
            target matrix
    """
    check_hermite(matrix)
    check_one_trace(matrix)
