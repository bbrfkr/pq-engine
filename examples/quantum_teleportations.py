from math import sqrt
from random import uniform

from pq_engine.matricies import (
    create_matrix_from_vector,
    epr_pair_matricies,
    one_state_vector,
    pauli_x_matrix,
    pauli_z_matrix,
    zero_state_vector,
)
from pq_engine.observable import Observable
from pq_engine.settings import array_engine
from pq_engine.state import State
from pq_engine.time_evolution import TimeEvolution

zero_probability = uniform(0, 1)
print("=== initial prob. of |0> ===")
print(zero_probability)

zero_amplitude = sqrt(zero_probability)
one_amplitude = sqrt(1 - zero_probability)
input_state_vector = array_engine.add(
    array_engine.multiply(zero_amplitude, zero_state_vector),
    array_engine.multiply(one_amplitude, one_state_vector),
)
input_state_matrix = create_matrix_from_vector(input_state_vector)
print()
print("=== input state ===")
print(input_state_matrix)

input_epr_pair_matrix = epr_pair_matricies[0]
print()
print("=== channel epr pair ===")
print(input_epr_pair_matrix)

initial_state_matrix = array_engine.kron(
    input_state_matrix,
    input_epr_pair_matrix,
)
print()
print("=== initial compound state ===")
print(initial_state_matrix)

projection_matricies = [
    array_engine.kron(epr_pair_matrix, array_engine.identity(2))
    for epr_pair_matrix in epr_pair_matricies
]
observable_matrix = array_engine.zeros((8, 8))
for index, projection_matrix in enumerate(projection_matricies):
    observable_matrix = array_engine.add(
        observable_matrix, array_engine.multiply(index, projection_matrix)
    )
print()
print("=== epr pairs observation matrix ===")
print(initial_state_matrix)

state = State(initial_state_matrix)
observable = Observable(observable_matrix)
observed_value = observable.observe(state)

print()
print("=== observed value ===")
print(observed_value)
print()
print("=== converged state ===")
print(state.matrix)

if int(observed_value) == 0:
    time_evolution_matrix = array_engine.kron(
        array_engine.identity(4), array_engine.identity(2)
    )
elif int(observed_value) == 1:
    time_evolution_matrix = array_engine.kron(
        array_engine.identity(4), pauli_x_matrix
    )
elif int(observed_value) == 2:
    time_evolution_matrix = array_engine.kron(
        array_engine.identity(4),
        array_engine.dot(pauli_x_matrix, pauli_z_matrix),
    )
elif int(observed_value) == 3:
    time_evolution_matrix = array_engine.kron(
        array_engine.identity(4), pauli_z_matrix
    )
else:
    raise ValueError(f"Unexpected observed value: {observed_value}")
time_evolution = TimeEvolution(time_evolution_matrix)
print()
print("=== time evolution matrix ===")
print(time_evolution.matrix)

time_evolution.time_evolve(state)
print()
print("=== time evolved compound state ===")
print(state.matrix)

state.reduce(0, [2, 2, 2])
state.reduce(0, [2, 2])
print()
print("=== final state matrix ===")
print(state.matrix)
