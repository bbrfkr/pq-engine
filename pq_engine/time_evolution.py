from .settings import xp
from .state import State
from .utils import check_unitary


class TimeEvolution:
    """
    time evolution

    Attributes:
        matrix (xp.ndarray): representation matrix
    """

    def __init__(self, matrix: xp.ndarray):
        check_unitary(matrix)
        self.matrix = matrix

    def time_evolve(self, state: State) -> None:
        """
        time evolve target state

        Args:
            state (State): target state
        """
        state.matrix = xp.dot(
            self.matrix,
            xp.dot(state.matrix, xp.conj(xp.transpose(self.matrix))),
        )
