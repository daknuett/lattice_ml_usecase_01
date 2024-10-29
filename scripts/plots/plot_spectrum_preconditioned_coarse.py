import numpy as np
import matplotlib.pyplot as plt

vals_mg = np.load(snakemake.input[0])
vals_prec = np.load(snakemake.input[1])

fermion_mass = float(snakemake.wildcards.fermion_mass)
n_basis = snakemake.wildcards.n_basis
config_no = snakemake.wildcards.config_no
n_mini_batches = int(snakemake.wildcards.n_mini_batches)
learn_rate = float(snakemake.wildcards.learn_rate)


plt.plot(np.real(vals_mg), np.imag(vals_mg)
         , '.', label=f"Eig(D) m={fermion_mass}, n_basis={n_basis}, config_no={config_no}")
plt.plot(np.real(vals_prec), np.imag(vals_prec)
         , '.', label=f"Eig(MD)")

plt.xlabel("Re")
plt.ylabel("Im")
plt.legend()
plt.grid()
plt.title(f"Spectrum of preconditioned coarse operator {n_mini_batches} mini batches, lr={learn_rate}")
plt.xlim(-.5, 3)

plt.savefig(snakemake.output[0])

