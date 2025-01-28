## VeloxQ: A Fast and Efficient QUBO Solver

This repository contains data and scripts used in the paper `VeloxQ: A Fast and Efficient QUBO Solver`. It is organized into four directories, corresponding to sections of the paper:
- `dwave-annealer` - Benchmarks against D-Wave's quantum annealers,
- `kipu-quantum-hubo` - Benchmarks againts Kipu's quantum solver,
- `ground-state-certification` - Benchmarks againts solvers with ground state certification,
- `planted-solution` - Benchmarks against physics-inspired algorithms.

We include both the scripts used to generate benchmark instances and, due to storage constraints, small examples of instances. Larger instances are available from the corresponding author upon resonable request. In the process of instance generation we have used:

- `D-Wave Ocean SDK` for generating Pegasus and Zephyr instances in quantum annealer benchmark, and Chimera instances in ground state certification benchmark,
- `Chook` python suite for binary optimization problems with planted solutions, to prepare 3R3X, tile planting and Wishart ensemble problems
- custom `Julia` scripts for the HUBO instances, random 3-body Ising model and weighted max-3SAT problem.

To maintain possible transparency, we also present the raw date that is subsequently plotted in the paper. The VeloxQ, as well as our highly optimized Simulated Annealing and Parallel Annealing implementations are proprietary, thus we do not disclose the details about running experiments with these algorithms.