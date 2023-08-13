from math import sqrt

from pq_engine.observable import Observable
from pq_engine.settings import xp
from pq_engine.state import State
from pq_engine.time_evolution import TimeEvolution

qubits_count = 12
structure = xp.array([2 for i in range(qubits_count)], dtype=xp.int16)
zero_state = xp.array(
    [
        [1, 0],
        [0, 0],
    ],
    dtype=xp.complex64,
)
one_state = xp.array(
    [
        [0, 0],
        [0, 1],
    ],
    dtype=xp.complex64,
)
state_matrix = one_state
for i in range(qubits_count):
    if i == 0:
        continue
    else:
        state_matrix = xp.kron(state_matrix, one_state)

observable_base = xp.array(
    [i for i in range(2**qubits_count)], dtype=xp.complex64
)
observable_matrix = xp.diag(observable_base)

hadamard_matrix = xp.array(
    [
        [1 / sqrt(2), 1 / sqrt(2)],
        [1 / sqrt(2), -1 / sqrt(2)],
    ],
    dtype=xp.complex64,
)
time_evolution_matrix = hadamard_matrix
for i in range(qubits_count):
    if i == 0:
        continue
    else:
        time_evolution_matrix = xp.kron(time_evolution_matrix, hadamard_matrix)

state = State(state_matrix, structure)
observable = Observable(observable_matrix, structure)
time_evolution = TimeEvolution(time_evolution_matrix, structure)

time_evolution.time_evolve(state)
print(f"observed value is {observable.observe(state)}")
