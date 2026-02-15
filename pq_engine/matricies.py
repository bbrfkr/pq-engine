from math import sqrt
from typing import Any

from .settings import array_engine


def create_matrix_from_vector(vector: Any) -> Any:
    """
    create matrix from vector

    Args:
        vector (array_engine.ndarray): source vector

    Returns:
        array_engine.ndarray: matrix converted by source vector
    """
    vector = vector.reshape(vector.size, 1)
    return array_engine.dot(
        vector, array_engine.conj(array_engine.transpose((vector)))
    )


# binary state vectors (column vectors)
#: |0>
zero_state_vector = array_engine.array([1, 0], dtype=array_engine.complex64)
#: |1>
one_state_vector = array_engine.array([0, 1], dtype=array_engine.complex64)
#: |+>
plus_state_vector = array_engine.array(
    [1 / sqrt(2), 1 / sqrt(2)], dtype=array_engine.complex64
)
#: |->
minus_state_vector = array_engine.array(
    [1 / sqrt(2), -1 / sqrt(2)], dtype=array_engine.complex64
)

# binary state matricies
#: |0><0|
zero_state_matrix = create_matrix_from_vector(zero_state_vector)
#: |1><1|
one_state_matrix = create_matrix_from_vector(one_state_vector)
#: |+><+|
plus_state_matrix = create_matrix_from_vector(plus_state_vector)
#: |-><-|
minus_state_matrix = create_matrix_from_vector(minus_state_vector)

#: hadamard matrix
hadamard_matrix = array_engine.array(
    [[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)]],
    dtype=array_engine.complex64,
)

# pauli matricies
#: σ_x
pauli_x_matrix = array_engine.array(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=array_engine.complex64,
)
#: σ_y
pauli_y_matrix = array_engine.array(
    [
        [0, -1j],
        [1j, 0],
    ],
    dtype=array_engine.complex64,
)
#: σ_z
pauli_z_matrix = array_engine.array(
    [
        [1, 0],
        [0, -1],
    ],
    dtype=array_engine.complex64,
)

#: not matrix
not_matrix = pauli_x_matrix

#: controled-not matrix
controlled_not = array_engine.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=array_engine.complex64,
)

#: epr pairs state vectors
epr_pair_vectors = [
    array_engine.dot(
        controlled_not,
        (
            array_engine.kron(
                array_engine.dot(hadamard_matrix, zero_state_vector),
                zero_state_vector,
            )
        ),
    ),
    array_engine.dot(
        controlled_not,
        (
            array_engine.kron(
                array_engine.dot(hadamard_matrix, zero_state_vector),
                one_state_vector,
            )
        ),
    ),
    array_engine.dot(
        controlled_not,
        (
            array_engine.kron(
                array_engine.dot(hadamard_matrix, one_state_vector),
                one_state_vector,
            )
        ),
    ),
    array_engine.dot(
        controlled_not,
        (
            array_engine.kron(
                array_engine.dot(hadamard_matrix, one_state_vector),
                zero_state_vector,
            )
        ),
    ),
]
#: epr pairs state matricies
epr_pair_matricies = [
    create_matrix_from_vector(epr_pair_vector)
    for epr_pair_vector in epr_pair_vectors
]
