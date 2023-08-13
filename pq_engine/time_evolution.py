from .exceptions import InconsistentStructureError, StructuresNotMatchError
from .settings import xp
from .state import State
from .utils import check_1darray, check_unitary


class TimeEvolution:
    """
    time evolution

    params:
        matrix:  xp.ndarray
            representation matrix
        structure:  xp.ndarray
            structure of time evolution w.r.t. compound system
    """

    matrix: xp.ndarray
    structure: xp.ndarray

    def __init__(self, matrix: xp.ndarray, structure: xp.ndarray):
        check_1darray(structure)
        check_unitary(matrix)
        expected_dimension = xp.prod(structure)
        if xp.array([matrix.shape[0]], xp.int16) != expected_dimension:
            raise InconsistentStructureError
        self.structure = structure
        self.matrix = matrix

    def time_evolve(self, state: State) -> None:
        """
        time evolve target state

        args:
            state: State
                target state
        """
        if not xp.array_equal(self.structure, state.structure):
            raise StructuresNotMatchError
        state.matrix = xp.dot(
            self.matrix,
            xp.dot(state.matrix, xp.conj(xp.transpose(self.matrix))),
        )
