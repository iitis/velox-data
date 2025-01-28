import dwave_networkx as dnx
import networkx as nx
import minorminer
import numpy as np
import dimod
import dwave.inspector as inspector

for num_variables in [20, 40, 60, 80, 100, 120, 140, 160, 180]:
    for rep in range(20):

        clique_graph = nx.complete_graph(num_variables)

        Q = {}
        for i in clique_graph.nodes:
            Q[(i, i)] = np.random.uniform(-1, 1)
        for i, j in clique_graph.edges:
            Q[(i, j)] = np.random.uniform(-1, 1)

        # embedding is handles by the dwave run script
        with open(
            f"embeddable_pegasus/ocean/randomQuboDWaveClique_L={num_variables}_dense_{rep+1}.txt",
            "w",
        ) as f:
            for i, j in Q:
                f.write(f"{i} {j} {Q[(i, j)]}\n")
