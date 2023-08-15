from math import sqrt

from .settings import xp


def create_matrix_from_vector(vector: xp.ndarray) -> xp.ndarray:
    """
    create matrix from vector

    Args:
        vector (xp.ndarray): source vector

    Returns:
        xp.ndarray: matrix converted by source vector
    """
    vector = vector.reshape(vector.size, 1)
    return xp.dot(vector, xp.conj(xp.transpose((vector))))


# binary state vectors (column vectors)
zero_state_vector = xp.array([1, 0], dtype=xp.complex64)
one_state_vector = xp.array([0, 1], dtype=xp.complex64)
plus_state_vector = xp.array([1 / sqrt(2), 1 / sqrt(2)], dtype=xp.complex64)
minus_state_vector = xp.array([1 / sqrt(2), -1 / sqrt(2)], dtype=xp.complex64)

# binary state matricies
zero_state_matrix = create_matrix_from_vector(zero_state_vector)
one_state_matrix = create_matrix_from_vector(one_state_vector)
plus_state_matrix = create_matrix_from_vector(plus_state_vector)
minus_state_matrix = create_matrix_from_vector(minus_state_vector)

# hadamard matrix
hadamard_matrix = xp.array(
    [[1 / sqrt(2), 1 / sqrt(2)], [1 / sqrt(2), -1 / sqrt(2)]],
    dtype=xp.complex64,
)

# pauli matricies
pauli_x_matrix = xp.array(
    [
        [0, 1],
        [1, 0],
    ],
    dtype=xp.complex64,
)
pauli_y_matrix = xp.array(
    [
        [0, -1j],
        [1j, 0],
    ],
    dtype=xp.complex64,
)
pauli_z_matrix = xp.array(
    [
        [1, 0],
        [0, -1],
    ],
    dtype=xp.complex64,
)

# not matrix
not_matrix = pauli_x_matrix

# controled-not matrix
controlled_not = xp.array(
    [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1],
        [0, 0, 1, 0],
    ],
    dtype=xp.complex64,
)

# epr pairs
epr_pair_vectors = [
    xp.dot(
        controlled_not,
        (
            xp.kron(
                xp.dot(hadamard_matrix, zero_state_vector), zero_state_vector
            )
        ),
    ),
    xp.dot(
        controlled_not,
        (
            xp.kron(
                xp.dot(hadamard_matrix, zero_state_vector), one_state_vector
            )
        ),
    ),
    xp.dot(
        controlled_not,
        (xp.kron(xp.dot(hadamard_matrix, one_state_vector), one_state_vector)),
    ),
    xp.dot(
        controlled_not,
        (
            xp.kron(
                xp.dot(hadamard_matrix, one_state_vector), zero_state_vector
            )
        ),
    ),
]
epr_pair_matricies = [
    create_matrix_from_vector(epr_pair_vector)
    for epr_pair_vector in epr_pair_vectors
]
