from .exceptions import InconsistentStructureError
from .settings import xp
from .utils import check_1darray, check_density


class State:
    """
    state

    params:
        matrix:  xp.ndarray
            representation matrix
        structure:  xp.ndarray
            structure of state w.r.t. compound system
    """

    matrix: xp.ndarray
    structure: xp.ndarray

    def __init__(self, matrix: xp.ndarray, structure: xp.ndarray):
        check_1darray(structure)
        check_density(matrix)
        expected_dimension = xp.prod(structure)
        if xp.array([matrix.shape[0]], dtype=xp.int16) != expected_dimension:
            raise InconsistentStructureError
        self.structure = structure
        self.matrix = matrix
