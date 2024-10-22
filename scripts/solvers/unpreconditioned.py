import numpy as np
import qcd_ml
import torch


config_no = snakemake.wildcards.config_no
fermion_mass = float(snakemake.wildcards.fermion_mass)
n_statistic = int(snakemake.wildcards.n_statistic)

U = torch.tensor(np.load(snakemake.input[0]))
w = qcd_ml.qcd.dirac.dirac_wilson_clover(U, fermion_mass, 1)

src = torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.cdouble)

iterations = np.zeros(n_statistic)
for i in range(n_statistic):
    src = torch.randn_like(src)

    x, info = qcd_ml.util.solver.GMRES(w, src, torch.clone(src), eps=1e-6, maxiter=60000, inner_iter=30, verbose=True)
    print(f"{i}: {info['k']}, residual: {info['res']}")
    iterations[i] = info["k"]

np.save(snakemake.output[0], iterations)
np.save(snakemake.output[1], np.array([np.mean(iterations), np.std(iterations)]))
