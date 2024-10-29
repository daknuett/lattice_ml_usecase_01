import numpy as np
import matplotlib.pyplot as plt

vals_mg = np.load(snakemake.input[0])

fermion_mass = float(snakemake.wildcards.fermion_mass)
n_basis = snakemake.wildcards.n_basis
config_no = snakemake.wildcards.config_no


plt.plot(np.real(vals_mg), np.imag(vals_mg)
         , '.', label=f"m={fermion_mass}, n_basis={n_basis}, config_no={config_no}")

plt.xlabel("Re")
plt.ylabel("Im")
plt.legend()
plt.grid()
plt.title("Spectrum of the coarse operator using Multigrid")
plt.xlim(-.5, 3)

plt.savefig(snakemake.output[0])
