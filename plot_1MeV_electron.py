import os
import uproot as up
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
ly = 760
# ly = 380
# ly = 1520
#/eos/user/p/plasorak/LiquidO/g4_1108/electrons/ly_760
path = Path(f'/eos/user/p/plasorak/LiquidO/g4_1108/electrons/ly_{ly}')
files = os.listdir(path)

n_hits = []

for file in files:
    print(f"Opening file {file}")
    try:
        data = up.open(os.path.join(path,file))
        n_hits += [data["op_hits"].num_entries]
    except Exception as e:
        print(f"Error in file {file}")
        print(e)

path = Path(f'/eos/user/p/plasorak/LiquidO/g4_1108/electrons/ly_{ly}_with_cherenkov')
files = os.listdir(path)

n_hits_with_cherenkov = []

for file in files:
    print(f"Opening file {file}")
    try:
        data = up.open(os.path.join(path,file))
        n_hits_with_cherenkov += [data["op_hits"].num_entries]
    except Exception as e:
        print(f"Error in file {file}")
        print(e)


f, ax = plt.subplots()
plt.hist(n_hits, label="no cherenkov", histtype='step')
plt.hist(n_hits_with_cherenkov, label="with cherenkov", histtype='step')
plt.xlabel('Number of hits')
plt.ylabel('Number of events')
plt.title(f'1 MeV electrons, light yield {ly}')
plt.legend()
plt.text(0.01, 0.9, f'Mean: {np.mean(n_hits):.2f}', transform = ax.transAxes)
plt.text(0.01, 0.85, f'Std dev: {np.std(n_hits):.2f}', transform = ax.transAxes)
plt.savefig('n_hits_1MeV_electrons.png')
