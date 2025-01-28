import dwave_networkx as dnx
import numpy as np
import dimod
from dwave.system import DWaveSampler

qpu = DWaveSampler(solver={'topology__type': 'pegasus'})
qpu_graph = qpu.to_networkx_graph()
for pegasus_size in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]:
  for num in range(1, 11):

    G = dnx.pegasus_graph(pegasus_size)  # source graph

    subgraph_iso = dnx.pegasus_sublattice_mappings(G, qpu_graph) # mapping to nodes of target graph
    fs = list(subgraph_iso) 
    f = fs[0] 
    print(len(fs))
    Gnodes = [f(n) for n in G.nodes]
    H = qpu_graph.subgraph(Gnodes)
    
    Q = {}
    for i in H.nodes:
        Q[(i, i)] = np.random.uniform(-1, 1)  # Random linear bias for qubit i
    for i, j in H.edges:
        Q[(i, j)] = np.random.uniform(-1, 1)  # Random quadratic bias between qubit i and qubit j
    
    # save Q as list i j value
    print(f"Saving QUBO to file: native_pegasus/ocean/randomQuboDWavePegasus_{pegasus_size}_dense_{num}.txt")
    with open(f'native_pegasus/ocean/randomQuboDWavePegasus_{pegasus_size}_dense_{num}.txt', 'w') as f:
      for i, j in Q:
        f.write(f'{i} {j} {Q[(i, j)]}\n')
    print(f"Number of qubits in the simulated Pegasus graph: {len(Gnodes)}")