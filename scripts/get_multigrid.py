import torch
import numpy as np

from qcd_ml.qcd.dirac import dirac_wilson_clover
from qcd_ml.util.solver import GMRES
from qcd_ml.util.qcd.multigrid import ZPP_Multigrid


U = torch.tensor(np.load(snakemake.input[0]))
fermion_mass = float(snakemake.wildcards.fermion_mass)
n_basis = int(snakemake.wildcards.n_basis)

w = dirac_wilson_clover(U, fermion_mass, 1)


vec = torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.cdouble)

orig_vecs = [torch.randn_like(vec) for _ in range(n_basis)]

mg = ZPP_Multigrid.gen_from_fine_vectors(orig_vecs, [4, 4, 4, 4], lambda b, x0: GMRES(w, b, x0, maxiter=60000, eps=1e-7, inner_iter=20), verbose=True)
mg.save(snakemake.output[0])
