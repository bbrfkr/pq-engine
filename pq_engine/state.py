import cupy as cp

from .exceptions import InconsistentStructureError
from .utils import check_1darray, check_density


class State:
    matrix: cp.ndarray
    structure: cp.ndarray

    def __init__(self, matrix: cp.ndarray, structure: cp.ndarray):
        check_1darray(structure)
        check_density(matrix)
        expected_dimension = cp.prod(structure)
        if cp.array([matrix.shape[0]], dtype=cp.int16) != expected_dimension:
            raise InconsistentStructureError
        self.structure = structure
        self.matrix = matrix
