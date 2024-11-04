import torch
import numpy as np

import qcd_ml
from qcd_ml.qcd.dirac import dirac_wilson_clover
from qcd_ml.util.qcd.multigrid import ZPP_Multigrid

from scipy import sparse
import scipy

U = torch.tensor(np.load(snakemake.input[0]))
mg = ZPP_Multigrid.load(snakemake.input[1])

fermion_mass = float(snakemake.wildcards.fermion_mass)
n_basis = int(snakemake.wildcards.n_basis)

paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(3, -1)]]
layer = qcd_ml.nn.lptc.v_LPTC_NG(1, 1, paths, mg.L_coarse, n_basis)
layer.load_state_dict(torch.load(snakemake.input[2]))

w = dirac_wilson_clover(U, fermion_mass, 1)

w_coarse_mg = mg.get_coarse_operator(w)
def mg_coarse_operator_scipy(v_scipy):
    v_torch = torch.tensor(v_scipy, dtype=torch.cdouble).reshape(2,2,2,4, mg.n_basis)
    Wv = layer.forward(torch.stack([w_coarse_mg(v_torch)]))
    return Wv.reshape(2*2*2*4*mg.n_basis).numpy()

with torch.no_grad():
    w_scipy = sparse.linalg.LinearOperator((2*2*2*4*mg.n_basis,2*2*2*4*mg.n_basis), mg_coarse_operator_scipy)
    vals_mg, vecs = sparse.linalg.eigs(w_scipy, k=(2*2*2*4*mg.n_basis - 2), which="SM")

np.save(snakemake.output[0], vals_mg)


