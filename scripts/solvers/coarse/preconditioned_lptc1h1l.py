import torch
import numpy as np

import qcd_ml
from qcd_ml.qcd.dirac import dirac_wilson_clover
from qcd_ml.util.qcd.multigrid import ZPP_Multigrid
from qcd_ml.util.solver import GMRES
from qcd_ml.qcd.dirac.coarsened import coarse_9point_op_NG

from scipy import sparse
import scipy

U = torch.tensor(np.load(snakemake.input[0]))
state_dict = torch.load(snakemake.input[1], weights_only=True)

w_coarse = coarse_9point_op_NG(state_dict["pseudo_gauge_forward"], state_dict["pseudo_gauge_backward"], state_dict["pseudo_mass"], state_dict["L_coarse"])

fermion_mass = float(snakemake.wildcards.fermion_mass)
n_basis = int(snakemake.wildcards.n_basis)
n_statistic = int(snakemake.wildcards.n_statistic)

paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(3, -1)]]
layer = qcd_ml.nn.lptc.v_LPTC_NG(1, 1, paths, w_coarse.L_coarse, n_basis)
layer.load_state_dict(torch.load(snakemake.input[2]))


with torch.no_grad():
    src = torch.randn(2, 2, 2, 4, 12, dtype=torch.cdouble)

    iterations = np.zeros(n_statistic)
    for i in range(n_statistic):
        src = torch.randn_like(src)

        x, info = GMRES(w_coarse, src, torch.clone(src), eps=1e-6, maxiter=600, inner_iter=20, preconditioner=lambda x: layer.forward(torch.stack([x]))[0])
        print(f"{i}: {info['k']}, residual: {info['res']}")
        iterations[i] = info["k"]

    np.save(snakemake.output[0], iterations)
    np.save(snakemake.output[1], np.array([np.mean(iterations), np.std(iterations)]))
