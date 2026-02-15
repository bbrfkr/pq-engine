from typing import Any
from .settings import array_engine, rounded_decimal
from .state import State
from .utils import check_hermite


class Observable:
    """
    observable

    Attributes:
        matrix (Any): representation matrix
    """

    def __init__(self, matrix: Any):
        check_hermite(matrix)
        self.matrix = matrix

    def observe(self, state: State) -> float:
        """
        observe target system with given state

        Args:
            state (State): state of target system
        Returns:
            float: observed value
        """
        eigen_values, eigen_vectors_groups = self._analyze_observable()
        return float(self._converge(state, eigen_values, eigen_vectors_groups))

    def _analyze_observable(self) -> tuple[Any, list[Any]]:
        """
        derivate eigen values and eigen vectors of observable

        Returns:
            tuple[Any, list[Any]]:
                devivated eigen values and eigen vectors groups
        """
        eigen_values, eigen_vectors = array_engine.linalg.eigh(self.matrix)
        eigen_vectors = array_engine.transpose(eigen_vectors)
        sorted_indices = array_engine.argsort(eigen_values)
        eigen_values.sort()
        eigen_vectors = eigen_vectors[sorted_indices]
        eigen_values, indices = array_engine.unique(
            array_engine.round(eigen_values, decimals=rounded_decimal),
            return_index=True,
        )
        indices = list(indices)
        indices.append(eigen_vectors.shape[0])
        eigen_vectors_groups = [
            eigen_vectors[indices[i] : indices[i + 1]]
            for i in range(len(indices) - 1)
        ]
        return (eigen_values, eigen_vectors_groups)

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
        probabilities = array_engine.array(
            [
                array_engine.round(
                    array_engine.sum(
                        array_engine.array(
                            [
                                array_engine.inner(
                                    eigen_vectors[i],
                                    array_engine.dot(
                                        state.matrix, eigen_vectors[i]
                                    ),
                                )
                                for i in range(eigen_vectors.shape[0])
                            ]
                        )
                    ).real,
                    decimals=rounded_decimal,
                )
                for eigen_vectors in eigen_vectors_groups
            ],
            dtype=array_engine.float32,
        )
        indices = array_engine.arange(probabilities.size)
        observed_index = int(
            array_engine.random.choice(indices, size=None, p=probabilities)
        )
        observed_probability = probabilities[observed_index]
        observed_vectors = eigen_vectors_groups[observed_index]
        observed_projection = array_engine.zeros(state.matrix.shape)
        for i in range(observed_vectors.shape[0]):
            observed_projection = array_engine.add(
                observed_projection,
                array_engine.dot(
                    array_engine.transpose(
                        array_engine.array([observed_vectors[i]])
                    ),
                    array_engine.conj(
                        array_engine.array([observed_vectors[i]])
                    ),
                ),
            )
        state.matrix = array_engine.divide(
            array_engine.dot(
                observed_projection,
                array_engine.dot(state.matrix, observed_projection),
            ),
            observed_probability,
        )
        observed_value = eigen_values[observed_index].real
        return observed_value
