import numpy as np
import matplotlib.pyplot as plt

config_no = snakemake.wildcards.config_no
fermion_mass = float(snakemake.wildcards.fermion_mass)
n_mini_batches = int(snakemake.wildcards.n_mini_batches)
learn_rate = float(snakemake.wildcards.learn_rate)

loss = np.load(snakemake.input[0])

plt.plot(loss, label=f"c.no {config_no} m={fermion_mass} lr={learn_rate}")
plt.xlabel("mini batch")
plt.ylabel("loss")
plt.title("Loss of 2->2->2->1 Smoother")
plt.yscale("log")

plt.legend()
plt.savefig(snakemake.output[0])
