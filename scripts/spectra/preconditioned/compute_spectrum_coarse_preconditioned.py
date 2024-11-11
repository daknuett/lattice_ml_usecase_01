import torch
import numpy as np

import qcd_ml
from qcd_ml.qcd.dirac import dirac_wilson_clover
from qcd_ml.qcd.dirac.coarsened import coarse_9point_op_NG

from scipy import sparse
import scipy

U = torch.tensor(np.load(snakemake.input[0]))
state_dict = torch.load(snakemake.input[1], weights_only=True)
n_basis = int(snakemake.wildcards.n_basis)

w_coarse_mg = coarse_9point_op_NG(state_dict["pseudo_gauge_forward"], state_dict["pseudo_gauge_backward"], state_dict["pseudo_mass"], state_dict["L_coarse"])

fermion_mass = float(snakemake.wildcards.fermion_mass)
n_basis = int(snakemake.wildcards.n_basis)

paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(3, -1)]]
layer = qcd_ml.nn.lptc.v_LPTC_NG(1, 1, paths, w_coarse_mg.L_coarse, n_basis)
layer.load_state_dict(torch.load(snakemake.input[2], weights_only=True))


def mg_coarse_operator_scipy(v_scipy):
    v_torch = torch.tensor(v_scipy, dtype=torch.cdouble).reshape(2,2,2,4, n_basis)
    Wv = layer.forward(torch.stack([w_coarse_mg(v_torch)]))
    return Wv.reshape(2*2*2*4*n_basis).numpy()

with torch.no_grad():
    w_scipy = sparse.linalg.LinearOperator((2*2*2*4*n_basis,2*2*2*4*n_basis), mg_coarse_operator_scipy)
    vals_mg, vecs = sparse.linalg.eigs(w_scipy, k=(2*2*2*4*n_basis - 2), which="SM")

np.save(snakemake.output[0], vals_mg)
