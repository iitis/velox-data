import dimod
from dwave.system.samplers import DWaveSampler
import json
import os
import time

dirname = "native_zephyr/ocean"
fnames = os.listdir(dirname + "/instances")
fnames = [f for f in fnames if f.endswith(".txt")]
fnames = sorted(fnames, key=lambda x: int(x.split("_")[1]))
fnames = list(filter(lambda x: int(x.split("_")[1]) <= 6, fnames))
print(fnames)

time.clock_gettime_ns(time.CLOCK_MONOTONIC)
sampler = DWaveSampler(solver={'topology__type': 'zephyr', 'qpu': True})
tts = []
for fname in fnames:
  Q = {}
  res_file = dirname + "/responses/" + f"response_{os.path.basename(fname).split('.')[0]}.json"
  if os.path.exists(res_file):
    print(f"Skipping {fname}")
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
  
  bqm = dimod.BinaryQuadraticModel.from_qubo(Q)
  try:
    tic = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    response = sampler.sample(bqm, num_reads=2**10)
    toc = time.clock_gettime_ns(time.CLOCK_MONOTONIC)
    tts.append((fname, toc - tic))
    with open(dirname + "/responses/" + f"response_{os.path.basename(fname).split('.')[0]}.json", "w") as f:
      json.dump(response.to_serializable(), f)
  except Exception as e:
    print(f"Failed to solve {fname}, because {e}")

# save tts to a file as csv
tts2 = sorted(tts, key=lambda x: (int(x[0].split("_")[1]), int(x[0].split("_")[3][:-4])))
with open(dirname + "/tts_zephyr_native.csv", "w") as f:
  for fname, tt in tts2:
    f.write(f"{fname},{tt}\n")

responses = os.listdir(dirname + "/responses")
total_time_mu_sec = 0.0
for res in responses:
  with open(dirname + "/responses/" + res, "r") as f:
    try:
     data = json.load(f)
    except:
      continue
    # print(data["info"]["timing"])
    total_time_mu_sec += data["info"]["timing"]["qpu_access_time"]
total_time_sec = total_time_mu_sec / 1e6
total_time_sec


