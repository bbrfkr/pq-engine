import cupy as cp

from .exceptions import InconsistentStructureError, StructuresNotMatchError
from .state import State
from .utils import check_1darray, check_hermite


class Observable:
    """
    observable class

    params:
        matrix: cupy.ndarray
            representation matrix
        structure: cupy.ndarray
            structure of observable w.r.t. compound system
    """

    matrix: cp.ndarray
    structure: cp.ndarray

    def __init__(self, matrix: cp.ndarray, structure: cp.ndarray):
        check_1darray(structure)
        check_hermite(matrix)
        expected_dimension = cp.prod(structure)
        if cp.array([matrix.shape[0]], dtype=cp.int16) != expected_dimension:
            raise InconsistentStructureError
        self.structure = structure
        self.matrix = matrix

    def observe(self, state: State) -> float:
        """
        observe target system with given state

        args:
            state: State
                state of target system
        return value:
            float
                observed value
        """
        if not cp.array_equal(self.structure, state.structure):
            raise StructuresNotMatchError

        eigen_values, eigen_vectors = cp.linalg.eigh(self.matrix)
        sorted_indices = cp.argsort(eigen_values)
        eigen_values.sort()
        eigen_vectors = eigen_vectors[sorted_indices]
        eigen_values, indices = cp.unique(
            cp.round(eigen_values, decimals=5), return_index=True
        )
        indices = list(indices)
        indices.append(eigen_vectors.shape[0])
        eigen_vectors_groups = [
            eigen_vectors[indices[i] : indices[i + 1]]
            for i in range(len(indices) - 1)
        ]
        return float(self._converge(state, eigen_values, eigen_vectors_groups))

    def _converge(
        self,
        state: State,
        eigen_values: cp.array,
        eigen_vectors_groups: list[cp.array],
    ) -> cp.float32:
        """
        converge state

        params:
            state: State
                target state for converged
            observable_values: list[cupy.array]
                array of observable values
            observable_projections: list[cupy.array]
                array of observable projections

        return value:
            cupy.float32
                observed value
        """
        probabilities = cp.array(
            [
                cp.sum(
                    cp.array(
                        [
                            cp.inner(
                                eigen_vectors[i],
                                cp.dot(state.matrix, eigen_vectors[i]),
                            )
                            for i in range(eigen_vectors.shape[0])
                        ]
                    )
                ).real
                for eigen_vectors in eigen_vectors_groups
            ],
            dtype=cp.float32,
        )
        indices = cp.arange(probabilities.size)
        observed_index = int(
            cp.random.choice(indices, size=1, p=probabilities)
        )
        observed_probability = probabilities[observed_index]
        observed_vectors = eigen_vectors_groups[observed_index]
        observed_projection = cp.sum(
            cp.array(
                [
                    cp.dot(
                        cp.transpose(observed_vectors[i]),
                        cp.conj(observed_vectors[i]),
                    )
                    for i in range(observed_vectors.shape[0])
                ],
                dtype=cp.complex64,
            )
        )
        state.matrix = cp.divide(
            cp.dot(
                observed_projection, cp.dot(state.matrix, observed_projection)
            ),
            observed_probability,
        )
        observed_value = eigen_values[observed_index].real
        return observed_value
