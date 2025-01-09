import matplotlib.pyplot as plt
import os
import pandas as pd

plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = 16

df = pd.read_csv(os.path.join("examples", "slimevolley", "results", "metrics.csv"))
fig, ax = plt.subplots(1, 1, figsize=(6, 3))

ax.plot(df.generation, df.best_fitness)

ax.set_xlabel("Generation")
ax.set_ylabel("Fitness")

fig.tight_layout()
fig.savefig("slimevolley.jpeg")