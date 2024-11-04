import torch
import numpy as np

import qcd_ml
from qcd_ml.qcd.dirac import dirac_wilson_clover
from qcd_ml.util.qcd.multigrid import ZPP_Multigrid
from qcd_ml.util.solver import GMRES

from scipy import sparse
import scipy

U = torch.tensor(np.load(snakemake.input[0]))
mg = ZPP_Multigrid.load(snakemake.input[1])
fermion_mass = float(snakemake.wildcards.fermion_mass)
n_statistic = int(snakemake.wildcards.n_statistic)

w = dirac_wilson_clover(U, fermion_mass, 1)
w_coarse = mg.get_coarse_operator(w)



with torch.no_grad():
    src = torch.randn(2, 2, 2, 4, 12, dtype=torch.cdouble)

    iterations = np.zeros(n_statistic)
    for i in range(n_statistic):
        src = torch.randn_like(src)

        x, info = GMRES(w_coarse, src, torch.clone(src), eps=1e-6, maxiter=600, inner_iter=20)
        print(f"{i}: {info['k']}, residual: {info['res']}")
        iterations[i] = info["k"]

    np.save(snakemake.output[0], iterations)
    np.save(snakemake.output[1], np.array([np.mean(iterations), np.std(iterations)]))

