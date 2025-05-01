import random

from pq_engine.matricies import (
    create_matrix_from_vector,
    hadamard_matrix,
    identity_matrix,
    one_state_vector,
    zero_state_vector,
)
from pq_engine.observable import Observable
from pq_engine.settings import xp
from pq_engine.state import State


def multiple_kron(n, array):
    result = array
    for _ in range(n - 1):
        result = xp.kron(result, array)
    return result


from random import randint

bits_count = 10


def zero_function(x):
    return 0


def one_function(x):
    return 1


def modulo_balanced_function(x):
    return x % 2


def random_balanced_function(x):
    balanced_list = random.choices(
        range(2**bits_count), k=2 ** (bits_count - 1)
    )
    return 0 if x in balanced_list else 1


def bob_function_selector():
    random_value = randint(0, 4)
    if random_value == 0:
        return zero_function
    elif random_value == 1:
        return one_function
    elif random_value == 2:
        return modulo_balanced_function
    else:
        return random_balanced_function


bob_function = bob_function_selector()

# Create the initial state vector  |0>^n|1>
initial_state_vector = xp.kron(
    multiple_kron(bits_count, zero_state_vector),
    one_state_vector,
)

# Create the Hadamard transformation matrix
multiple_hadamard_matrix = multiple_kron(bits_count + 1, hadamard_matrix)

# Apply the Hadamard transformation to the initial state vector
superposition_state_vector = xp.dot(
    multiple_hadamard_matrix,
    initial_state_vector,
)

# Create the time evolution matrix for the function f
controlled_function_matrix = xp.zeros(
    (2 ** (bits_count + 1), 2 ** (bits_count + 1))
)
for x in range(2 ** (bits_count)):
    f = bob_function(x)
    x_zero = x * 2
    x_one = x * 2 + 1
    x_f_zero = x_zero + f
    x_f_one = x_one - f
    controlled_function_matrix[x_f_zero, x_zero] = 1
    controlled_function_matrix[x_f_one, x_one] = 1

# Apply the controlled function transformation to the superposition state vector
entangled_state_vector = xp.dot(
    controlled_function_matrix,
    superposition_state_vector,
)

# Create the Hadamard transformation matrix for the first n qubits
finalize_multiple_hadamard_matrix = xp.kron(
    multiple_kron(bits_count, hadamard_matrix), identity_matrix
)

# Apply the Hadamard transformation to the first n qubits of the entangled state vector
final_state_vector = xp.dot(
    finalize_multiple_hadamard_matrix,
    entangled_state_vector,
)

# Create the state
state = State(create_matrix_from_vector(final_state_vector))

# Create the observable matrix for the final state
observable_matrix = xp.kron(
    xp.diag(xp.array([i for i in range(2**bits_count)])),
    identity_matrix,
)

# Create the observable
observable = Observable(observable_matrix)

# Measure the observable to obtain the result
observed_value = observable.observe(state)

# Print the observed value and its interpretation
print("Observed value:", int(observed_value))
if int(observed_value) == 0:
    print("It implies that the function is constant")
else:
    print("It implies that the function is balanced")

# Print the function type
if bob_function == zero_function:
    print("Bob's function is constant: f(x) = 0")
elif bob_function == one_function:
    print("Bob's function is constant: f(x) = 1")
elif bob_function == modulo_balanced_function:
    print("Bob's function is balanced: f(x) = x mod 2")
elif bob_function == random_balanced_function:
    print("Bob's function is balanced: random balanced function")
