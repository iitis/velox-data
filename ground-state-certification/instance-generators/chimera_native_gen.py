import dwave_networkx as dnx
import numpy as np
import dimod
from dwave.system import DWaveSampler

for chimera_cols in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
    for num in range(1, 2):
        print(f"Creating C(8,{chimera_cols}) instance, rep={num}")
        G = dnx.chimera_graph(8, chimera_cols, 4)
        Q = {}
        for i in G.nodes:
            Q[(i, i)] = np.random.random_integers(-31, 31)
        for i, j in G.edges:
            Q[(i, j)] = np.random.random_integers(-31,31)

        # save Q as list i j value
        with open(
            f"data/instances/ocean/randomQuboDWaveChimera_8_{chimera_cols}_dense_{num}.txt",
            "w",
        ) as f:
            for i, j in Q:
                f.write(f"{i} {j} {Q[(i, j)]}\n")
        print(f"Number of qubits in the simulated Chimera graph: {len(G.nodes)}")
