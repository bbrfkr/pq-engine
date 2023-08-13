import cupy as cp

from .exceptions import InconsistentStructureError, StructuresNotMatchError
from .state import State
from .utils import check_1darray, check_unitary


class TimeEvolution:
    matrix: cp.ndarray
    structure: cp.ndarray

    def __init__(self, matrix: cp.ndarray, structure: cp.ndarray):
        check_1darray(structure)
        check_unitary(matrix)
        expected_dimension = cp.prod(structure)
        if cp.array([matrix.shape[0]], cp.int16) != expected_dimension:
            raise InconsistentStructureError
        self.structure = structure
        self.matrix = matrix

    def time_evolve(self, state: State):
        if not cp.array_equal(self.structure, state.structure):
            raise StructuresNotMatchError
        state.matrix = cp.dot(
            self.matrix,
            cp.dot(state.matrix, cp.conj(cp.transpose(self.matrix))),
        )
