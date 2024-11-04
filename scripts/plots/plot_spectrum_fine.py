import numpy as np
import matplotlib.pyplot as plt

vals_hm = np.load(snakemake.input[0])
vals_lm = np.load(snakemake.input[1])

fermion_mass = float(snakemake.wildcards.fermion_mass)
config_no = snakemake.wildcards.config_no


plt.plot(np.real(vals_lm), np.imag(vals_lm)
         , 'C0.', label=f"m={fermion_mass}, config_no={config_no}")
plt.plot(np.real(vals_hm), np.imag(vals_hm)
         , 'C0.')

plt.xlabel("Re")
plt.ylabel("Im")
plt.legend()
plt.grid()
plt.title(f"Spectrum of the fine operator ({vals_lm.shape[0] + vals_hm.shape[0]} eigenvalues)")
#plt.xlim(-.5, 3)

plt.savefig(snakemake.output[0])

