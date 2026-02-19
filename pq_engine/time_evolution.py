from typing import Any
from .settings import array_engine
from .state import State
from .utils import check_unitary


class TimeEvolution:
    """
    time evolution

    Attributes:
        matrix (Any): representation matrix
    """

    def __init__(self, matrix: Any):
        check_unitary(matrix)
        self.matrix = matrix

    def time_evolve(self, state: State) -> None:
        """
        time evolve target state

        Args:
            state (State): target state
        """
        state.matrix = array_engine.dot(
            self.matrix,
            array_engine.dot(
                state.matrix,
                array_engine.conj(self.matrix).T,
            ),
        )

    def compose(self, other: "TimeEvolution") -> "TimeEvolution":
        """
        composite time evolution (tensor product U1 ⊗ U2)

        Args:
            other (TimeEvolution): composition target

        Returns:
            TimeEvolution: composed time evolution
        """
        composed_matrix = array_engine.kron(self.matrix, other.matrix)
        return TimeEvolution(composed_matrix)
