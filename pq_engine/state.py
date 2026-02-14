from math import sqrt
from typing import Any

from .exceptions import SizeNotMatchError, TargetNotFoundError
from .settings import array_engine
from .utils import check_density


class State:
    """
    state

    Attributes:
        matrix (array_engine.ndarray): representation matrix
    """

    def __init__(self, matrix: Any):
        check_density(matrix)
        self.matrix = matrix

    def reduce(self, target: int, structure: list[int]):
        """
        reduce state by partial trace

        Args:
            target (int): reduction target index
            sttucture (list[int]): dimensions with partial systems
        """
        if self.matrix.shape[0] != int(
            array_engine.prod(array_engine.array(structure))
        ):
            raise SizeNotMatchError
        if target not in range(len(structure)):
            raise TargetNotFoundError
        shape = []
        for dim in structure:
            for i in range(2):
                shape.append(dim)
        self.matrix = self.matrix.reshape(shape)
        axis1 = target
        axis2 = target + len(structure)
        self.matrix = array_engine.trace(self.matrix, axis1=axis1, axis2=axis2)
        self.matrix = self.matrix.reshape(
            int(sqrt(self.matrix.size)), int(sqrt(self.matrix.size))
        )
