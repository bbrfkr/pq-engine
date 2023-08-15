from .settings import rounded_decimal, xp
from .state import State
from .utils import check_hermite


class Observable:
    """
    observable

    Attributes:
        matrix (xp.ndarray): representation matrix
    """

    def __init__(self, matrix: xp.ndarray):
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

    def _analyze_observable(self) -> tuple[xp.array, list[xp.array]]:
        """
        derivate eigen values and eigen vectors of observable

        Returns:
            tuple[xp.array, list[xp.array]]:
                devivated eigen values and eigen vectors groups
        """
        eigen_values, eigen_vectors = xp.linalg.eigh(self.matrix)
        eigen_vectors = xp.transpose(eigen_vectors)
        sorted_indices = xp.argsort(eigen_values)
        eigen_values.sort()
        eigen_vectors = eigen_vectors[sorted_indices]
        eigen_values, indices = xp.unique(
            xp.round(eigen_values, decimals=rounded_decimal), return_index=True
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
        eigen_values: xp.array,
        eigen_vectors_groups: list[xp.array],
    ) -> xp.float32:
        """
        converge state

        Args:
            state (State): target state for converged
            observable_values (list[xp.array]):
                array of observable values
            observable_projections (list[ xp.array]):
                array of observable projections

        Returns:
            xp.float32: observed value
        """
        probabilities = xp.array(
            [
                xp.round(
                    xp.sum(
                        xp.array(
                            [
                                xp.inner(
                                    eigen_vectors[i],
                                    xp.dot(state.matrix, eigen_vectors[i]),
                                )
                                for i in range(eigen_vectors.shape[0])
                            ]
                        )
                    ).real,
                    decimals=rounded_decimal,
                )
                for eigen_vectors in eigen_vectors_groups
            ],
            dtype=xp.float32,
        )
        indices = xp.arange(probabilities.size)
        observed_index = int(
            xp.random.choice(indices, size=1, p=probabilities)
        )
        observed_probability = probabilities[observed_index]
        observed_vectors = eigen_vectors_groups[observed_index]
        observed_projection = xp.zeros(state.matrix.shape)
        for i in range(observed_vectors.shape[0]):
            observed_projection = xp.add(
                observed_projection,
                xp.dot(
                    xp.transpose(xp.array([observed_vectors[i]])),
                    xp.conj(xp.array([observed_vectors[i]])),
                ),
            )
        state.matrix = xp.divide(
            xp.dot(
                observed_projection, xp.dot(state.matrix, observed_projection)
            ),
            observed_probability,
        )
        observed_value = eigen_values[observed_index].real
        return observed_value
