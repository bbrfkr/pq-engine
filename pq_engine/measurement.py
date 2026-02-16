from typing import Any
from .settings import array_engine, rounded_decimal
from .state import State
from .utils import check_square, check_measurement
from .exceptions import InvalidValueError


class MeasurementUnit:
    def __init__(
        self,
        value: Any,
        matrix: Any,
    ):
        if not array_engine.isreal(value):
            raise InvalidValueError
        check_square(matrix)
        self.value = value
        self.matrix = matrix


class Measurement:
    """
    measurement

    Attributes:
        units (List[MeasurementUnit]): list of measurement units
    """

    def __init__(self, units: Any):
        check_measurement(units)
        self.units = units

    def measure(self, state: State) -> float:
        """
        observe target system with given state

        Args:
            state (State): state of target system
        Returns:
            float: observed value
        """
        pass

    def _analyze_observable(self) -> tuple[Any, list[Any]]:
        """
        derivate eigen values and eigen vectors of observable

        Returns:
            tuple[Any, list[Any]]:
                devivated eigen values and eigen vectors groups
        """
        pass

    def _converge(
        self,
        state: State,
        eigen_values: Any,
        eigen_vectors_groups: list[Any],
    ) -> Any:
        """
        converge state

        Args:
            state (State): target state for converged
        Returns:
            Any: observed value
        """
        pass
