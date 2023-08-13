from math import sqrt

import cupy as cp

from pq_engine.observable import Observable
from pq_engine.state import State
from pq_engine.time_evolution import TimeEvolution

qubits_count = 13
structure = cp.array([2 for i in range(qubits_count)], dtype=cp.int16)
zero_state = cp.array(
    [
        [1, 0],
        [0, 0],
    ],
    dtype=cp.complex64,
)
one_state = cp.array(
    [
        [0, 0],
        [0, 1],
    ],
    dtype=cp.complex64,
)
state_matrix = one_state
for i in range(qubits_count):
    if i == 0:
        continue
    else:
        state_matrix = cp.kron(state_matrix, one_state)

observable_base = cp.array(
    [i for i in range(2**qubits_count)], dtype=cp.complex64
)
observable_matrix = cp.diag(observable_base)

hadamard_matrix = cp.array(
    [
        [1 / sqrt(2), 1 / sqrt(2)],
        [1 / sqrt(2), -1 / sqrt(2)],
    ],
    dtype=cp.complex64,
)
time_evolution_matrix = hadamard_matrix
for i in range(qubits_count):
    if i == 0:
        continue
    else:
        time_evolution_matrix = cp.kron(time_evolution_matrix, hadamard_matrix)

state = State(state_matrix, structure)
observable = Observable(observable_matrix, structure)
time_evolution = TimeEvolution(time_evolution_matrix, structure)

time_evolution.time_evolve(state)
print(f"observed value is {observable.observe(state)}")
