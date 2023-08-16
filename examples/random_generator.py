from pq_engine.matricies import hadamard_matrix, zero_state_matrix
from pq_engine.settings import xp
from pq_engine.state import State
from pq_engine.observable import Observable
from pq_engine.time_evolution import TimeEvolution

bits_count = 13

initial_state_matrix = zero_state_matrix
time_evolution_matrix = hadamard_matrix

for index in range(bits_count-1):
    initial_state_matrix = xp.kron(
        initial_state_matrix,
        zero_state_matrix,
    )
    time_evolution_matrix = xp.kron(
        time_evolution_matrix,
        hadamard_matrix,
    )

state = State(initial_state_matrix)
time_evolution = TimeEvolution(time_evolution_matrix)
time_evolution.time_evolve(state)

observable_matrix = xp.diag(xp.array(
    [i for i in range(2**bits_count)]
))
observable = Observable(observable_matrix)
random_value = int(observable.observe(state))

print(f"random value in [0,{2**bits_count}]: {random_value}")
