import matplotlib.pyplot as plt
import os
import pandas as pd


plt.rcParams["font.family"] = "Helvetica"
plt.rcParams["font.size"] = 16


fig, ax = plt.subplots(1, 1, figsize=(6, 3))

num_seeds = 3
for seed in range(num_seeds):
    df = pd.read_csv(os.path.join("examples", "slimevolley", "results", f"seed={seed}", "metrics.csv"))
    ax.plot(df.generation, df.best_fitness, label=f"seed={seed}")

ax.set_xlabel("Generation")
ax.set_ylabel("Fitness")

fig.legend(loc="lower center", bbox_to_anchor=[0.5, 0.], ncols=3)
fig.tight_layout()
fig.subplots_adjust(bottom=0.375)

fig.savefig("slimevolley.jpeg")