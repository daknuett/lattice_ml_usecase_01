import torch
import numpy as np

from qcd_ml.qcd.dirac import dirac_wilson_clover
from qcd_ml.util.qcd.multigrid import ZPP_Multigrid

from scipy import sparse
import scipy

U = torch.tensor(np.load(snakemake.input[0]))
fermion_mass = float(snakemake.wildcards.fermion_mass)

w = dirac_wilson_clover(U, fermion_mass, 1)

def operator_scipy(v_scipy):
    v_torch = torch.tensor(v_scipy, dtype=torch.cdouble).reshape(8,8,8,16, 4,3)
    Wv = w(v_torch)
    return Wv.reshape(8*8*8*16* 4*3).numpy()

w_scipy = sparse.linalg.LinearOperator((8*8*8*16* 4*3, 8*8*8*16* 4*3), operator_scipy)
vals_mg, vecs = sparse.linalg.eigs(w_scipy, k=(2*2*2*4*12 - 2), which="SM")

np.save(snakemake.output[0], vals_mg)
