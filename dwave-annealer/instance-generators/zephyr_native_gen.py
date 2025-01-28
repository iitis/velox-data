import dwave_networkx as dnx
import numpy as np
import dimod
import random
from dwave.system import DWaveSampler

# qpu = DWaveSampler(solver={"topology__type": "zephyr", "qpu": True})
# qpu_graph = qpu.to_networkx_graph()
qpu_graph = dnx.zephyr_graph(15)

for n in range(1, 21):
    for num in range(1, 21):
        G = dnx.zephyr_graph(n)

        # subgraph_iso = dnx.zephyr_sublattice_mappings(G, qpu_graph)
        # fs = list(subgraph_iso)
        # # f = random.choice(fs)
        # f = fs[0]
        # print(len(fs))
        # Gnodes = [f(n) for n in G.nodes]
        # H = qpu_graph.subgraph(Gnodes)
        H = G

        Q = {}  # Create an empty QUBO dictionary

        for i in H.nodes:
            Q[(i, i)] = np.random.uniform(-1, 1)
        for i, j in H.edges:
            Q[(i, j)] = np.random.uniform(
                -1, 1
            )  # Random quadratic bias between qubit i and qubit j

        # save Q as list i j value
        print(
            f"Saving QUBO to file: native_zephyr/ocean/randomQuboDWaveZephyr_{n}_dense_{num}.txt"
        )
        with open(
            f"native_zephyr/ocean/instances/randomQuboDWaveZephyr_{n}_dense_{num}.txt", "w"
        ) as f:
            for i, j in Q:
                f.write(f"{i} {j} {Q[(i, j)]}\n")

        # print(f"Number of qubits in the simulated Zephyr graph: {len(Gnodes)}")


