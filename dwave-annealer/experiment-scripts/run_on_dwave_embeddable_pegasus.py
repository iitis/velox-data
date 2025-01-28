import dimod
import dwave
from dwave.system.samplers import DWaveSampler
from dwave.system import embedding as eb
import json
import time
import os

dirname = "embeddable_pegasus/ocean"
fnames = os.listdir(dirname + "/instances")
fnames = [f for f in fnames if f.endswith(".txt")]
fnames = sorted(fnames, key=lambda x: (int(x.split("_")[1][2:]), int(x.split("_")[3][:-4])))
print(fnames)

def load_qubo(filename):
    Q = {}
    with open(filename, 'r') as f:
        for line in f:
            i, j, value = line.split()
            Q[(int(i), int(j))] = float(value)
    return Q

sampler = DWaveSampler(solver={'topology__type': 'pegasus', 'qpu': True})
tts = []
tts_emb = []
for fname in fnames:
  Q = {}
  res_file = dirname + "/responses/" + f"response_{os.path.basename(fname).split('.')[0]}.json"
  emb_file = dirname + "/embeddings/" + f"embedding_{os.path.basename(fname).split('.')[0]}.json"
  if os.path.exists(res_file):
    print(f"Skipping {fname}, because response file exists")
    continue
  print(f"Reading {fname}")
  with open(dirname + "/instances/" + fname, "r") as f:
      for line in f:
          if line[0] == "#":
              continue
          i, j, v = line.split()
          i, j = int(i), int(j)
          v = float(v)
          Q[(i, j)] = v
    
  try:
    bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
    tic_embed = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    emb = dwave.embedding.pegasus.find_clique_embedding(bqm.num_variables, target_graph=sampler.to_networkx_graph())
    emb_bqm = eb.embed_bqm(bqm, emb, sampler.adjacency)
    toc_embed = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    tts_emb.append((fname, toc_embed - tic_embed))
    tic = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    response = sampler.sample(emb_bqm, num_reads=2**10)
    toc = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    tts.append((fname, toc - tic))
    with open(emb_file, "w") as f:
      json.dump(emb, f)
    with open(res_file, "w") as f:
      json.dump(response.to_serializable(), f)
  except Exception as e:
    print(f"Failed to solve {fname}, because {e}")

tts2 = sorted(tts, key=lambda x: (int(x[0].split("_")[1][2:]), int(x[0].split("_")[3][:-4])))
tts2_emb = sorted(tts_emb, key=lambda x: (int(x[0].split("_")[1][2:]), int(x[0].split("_")[3][:-4])))
with open(dirname + "/dwave_embeddable_pegasus_tts_sol.csv", "w") as f:
  for fname, tt in tts2:
    f.write(f"{fname},{tt}\n")

with open(dirname + "/dwave_embeddable_pegasus_tts_emb.csv", "w") as f:
  for fname, tt in tts2_emb:
    f.write(f"{fname},{tt}\n")

responses = os.listdir(dirname + "/responses")
total_time_mu_sec = 0.0
for resp in responses:
  with open(dirname + "/responses/" + resp, "r") as f:
    try:
     data = json.load(f)
    except:
      continue
    total_time_mu_sec += data["info"]["timing"]["qpu_access_time"]

total_time_sec = total_time_mu_sec / 1e6
print(f"Total time spent on QPU: {total_time_sec} seconds")