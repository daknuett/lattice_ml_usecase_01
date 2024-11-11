import torch
import numpy as np

from qcd_ml.qcd.dirac import dirac_wilson_clover
from qcd_ml.qcd.dirac.coarsened import coarse_9point_op_NG

from scipy import sparse
import scipy

U = torch.tensor(np.load(snakemake.input[0]))
n_basis = int(snakemake.wildcards.n_basis)

state_dict = torch.load(snakemake.input[1], weights_only=True)

coarse_w = coarse_9point_op_NG(state_dict["pseudo_gauge_forward"], state_dict["pseudo_gauge_backward"], state_dict["pseudo_mass"], state_dict["L_coarse"])


def mg_coarse_operator_scipy(v_scipy):
    v_torch = torch.tensor(v_scipy, dtype=torch.cdouble).reshape(2,2,2,4, n_basis)
    Wv = coarse_w(v_torch)
    return Wv.reshape(2*2*2*4*n_basis).numpy()

w_scipy = sparse.linalg.LinearOperator((2*2*2*4*n_basis,2*2*2*4*n_basis), mg_coarse_operator_scipy)
vals_mg, vecs = sparse.linalg.eigs(w_scipy, k=(2*2*2*4*n_basis - 2), which="SM")

np.save(snakemake.output[0], vals_mg)

