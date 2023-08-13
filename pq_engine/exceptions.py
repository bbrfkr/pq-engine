class PQEngineBaseError(Exception):
    """base error class"""


class DimensionNotMatchError(PQEngineBaseError):
    """error that dimension is not matching"""

    message = "dimension of matrix is not 2."


class NotOneDimensionalError(PQEngineBaseError):
    """error that array is not 1 dimeisinoal."""

    message = "array is not 1 dimeisinoal."


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


class InconsistentStructureError(PQEngineBaseError):
    """error for inconsistence between matrix dimension and structure"""

    message = "structure is inconsist with matrix dimension."


class StructuresNotMatchError(PQEngineBaseError):
    """error that structures not match"""

    message = "structures not match."
