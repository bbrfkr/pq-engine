from .exceptions import (
    NonOneTraceError,
    NotHermiteError,
    NotSquareError,
    NotUnitaryError,
    SizeNotMatchError,
    NotMeasurementError,
)
from typing import Any, List
from .settings import atol, array_engine


def check_square(matrix: Any) -> None:
    """
    check matrix is square

    Args:
        matrix (array_engine.ndarray): target matrix
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise NotSquareError


def check_hermite(matrix: Any) -> None:
    """
    check matrix is hermite

    Args:
        matrix (array_engine.ndarray): target matrix
    """
    check_square(matrix)
    if not array_engine.allclose(
        matrix, array_engine.conj(matrix).T, atol=atol
    ):
        raise NotHermiteError


def check_unitary(matrix: Any) -> None:
    """
    check matrix is unitary

    Args:
        matrix (array_engine.ndarray): target matrix
    """
    check_square(matrix)
    dimension = matrix.shape[0]
    if not array_engine.allclose(
        array_engine.dot(
            matrix,
            array_engine.conj(matrix).T,
        ),
        array_engine.identity(
            dimension, dtype=array_engine.complex64
        ),
        atol=atol,
    ):
        raise NotUnitaryError


def check_one_trace(matrix: Any) -> None:
    """
    check trace of matrix is one

    Args:
        matrix (array_engine.ndarray): target matrix
    """
    check_square(matrix)
    if not array_engine.allclose(array_engine.trace(matrix), 1, atol=atol):
        raise NonOneTraceError


def check_density(matrix: Any) -> None:
    """
    check matrix is density

    Args:
        matrix (array_engine.ndarray): target matrix
    """
    check_hermite(matrix)
    check_one_trace(matrix)
