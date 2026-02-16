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
from .measurement import MeasurementUnit


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
        matrix, array_engine.conj(array_engine.transpose(matrix)), atol=atol
    ):
        raise NotHermiteError


def check_unitary(matrix: Any) -> None:
    """
    check matrix is unitary

    Args:
        matrix (array_engine.ndarray): target matrix
    """
    check_square(matrix)
    expected_dimension = matrix.shape[0]
    if not array_engine.allclose(
        array_engine.dot(
            matrix,
            array_engine.conj(array_engine.transpose(matrix)),
        ),
        array_engine.identity(
            expected_dimension, dtype=array_engine.complex64
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


def check_measurement(units: List[MeasurementUnit]) -> None:
    # check all units are the same dimension
    dimensions = set([
        unit.shape[0]
        for unit in units
    ])
    if not len(dims) == 1:
        raise SizeNotMatchError

    expected_dimension = dimensions[0]
    # check the sum over generate matricies of units is identity
    matricies = [
        array_engine.dot(
            array_engine.conj(
                unit.matrix
            ),
            unit.matrix
        )
        for unit in units
    ]
    if not array_engine.allclose(
        array_engine.sum(
            matricies
        ),
        array_engine.identity(
            expected_dimension, dtype=array_engine.complex64
        ),
        atol=atol,
    ):
        raise NotMeasurementError
