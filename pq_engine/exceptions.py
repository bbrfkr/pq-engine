class PQEngineBaseError(Exception):
    """base error class"""


class SizeNotMatchError(PQEngineBaseError):
    """error that dimension is not matching"""

    message = "size of matrix is not match."


class NotSquareError(PQEngineBaseError):
    """error that matrix is not square."""

    message = "matrix is not square."


class NotHermiteError(PQEngineBaseError):
    """error that matrix is not hermite"""

    message = "matrix is not hermite."


class NotUnitaryError(PQEngineBaseError):
    """error that matrix is not unitary"""

    message = "matrix is not unitary."


class NonOneTraceError(PQEngineBaseError):
    """error that trace of matrix is not 1."""

    message = "trace of matrix is not 1."


class TargetNotFoundError(PQEngineBaseError):
    """error that target is not found"""

    message = "target is not found."
