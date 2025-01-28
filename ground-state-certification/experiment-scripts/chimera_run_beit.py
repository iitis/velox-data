import numpy as np
import dimod
import dwave_networkx as dnx
from beit.qubo_solver.solver_connection import AWSSolverConnection
from beit.qubo_solver.beit_solver import BEITSolver
import json
import time
import os

connection = AWSSolverConnection("API KEY")
solver = BEITSolver(connection)
print(connection)
print(solver)

path_to_instance = (
    lambda cols, num=1: f"data/instances/ocean/randomQuboDWaveChimera_8_{cols}_dense_{num}.txt"
)
path_to_results = "data/results/beit_responses"

sizes = np.arange(2, 17, 1)
print(sizes)
tts = []
for size in sizes:
    fname = path_to_instance(size)
    res_file = f"{path_to_results}/response_{os.path.basename(fname).split('.')[0]}.json"
    if os.path.exists(res_file):
       print(f"Skipping {fname}, because response file exists")
       continue
    print(f"Sampling {fname}")
    Gsub = dnx.chimera_graph(8, size, 4)
    Gfull = dnx.chimera_graph(8, 16, 4)
    subgraph_iso = dnx.chimera_sublattice_mappings(Gsub, Gfull)
    fs = list(subgraph_iso)
    F = fs[0]
    print(f"There are {len(fs)} isomorphisms into C(8,16,4)")
    Q = {}
    vars = []
    with open(fname, "r") as f:
        for line in f:
            if line[0] == "#":
                continue
            i, j, v = line.split()
            i, j = int(i), int(j)
            v = float(v)
            Q[(F(i), F(j))] = v
            if i == j:
              vars.append(F(i))
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    variables = bqm.variables
    np.savetxt(f"{path_to_results}/variables_{os.path.basename(fname).split('.')[0]}.json", variables, fmt="%d")
    np.savetxt(f"{path_to_results}/variables2_{os.path.basename(fname).split('.')[0]}.json", vars, fmt="%d")
    try:
      tic = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
      response = solver.sample(bqm)
      toc = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
      tts.append((fname, toc - tic))
      with open(res_file, 'w') as f: 
        json.dump(response.to_serializable(), f)
      with open("data/tts_beit_exact.csv", "a") as f:
        f.write(f"{fname},{tts[-1][1]}\n")
    except Exception as e:
      print(f"Failed to solve {fname}, because {e}")

# # tts2 = sorted(tts, key=lambda x: (int(x[0].split("_")[2]), int(x[0].split("_")[-1][:-4])))
# with open("data/tts_beit_exact.csv", "w") as f:
#   for fname, tt in tts:
#     f.write(f"{fname},{tt}\n")
