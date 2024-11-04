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

class counting:
    def __init__(self, Q):
        self.Q = Q
        self.k = 0
    def __call__(self, x):
        self.k += 1
        return self.Q(x)

class Lvl2MultigridPreconditioner:    
    def __init__(self, q, mg, inner_solver_kwargs, smoother_kwargs):    
        self.q = q    
        self.mg = mg    
        self.inner_solver_kwargs = inner_solver_kwargs    
        self.smoother_kwargs = smoother_kwargs    
        self.q_coarse = self.mg.get_coarse_operator(self.q)    

    def __call__(self, b):    
        x_coarse, info_coarse = GMRES(self.q_coarse, self.mg.v_project(b), self.mg.v_project(b), **self.inner_solver_kwargs)    
        x = self.mg.v_prolong(x_coarse)    
        x, info_smoother = GMRES(self.q, torch.clone(b), torch.clone(x), **self.smoother_kwargs)    
        return x

counting_w = counting(w)

mg_prec = Lvl2MultigridPreconditioner(counting_w, mg, {"eps": 5e-2, "maxiter": 50, "inner_iter": 25}, {"eps": 1e-15, "maxiter": 8, "inner_iter": 4})


with torch.no_grad():
    src = torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.cdouble)

    iterations = np.zeros((n_statistic, 2))
    for i in range(n_statistic):
        counting_w.k = 0
        src = torch.randn_like(src)

        x, info = GMRES(w, src, torch.clone(src), eps=1e-6, maxiter=60000, inner_iter=30, preconditioner=mg_prec)
        print(f"{i}: {info['k']}, residual: {info['res']}")
        iterations[i, 0] = info["k"]
        iterations[i, 1] = counting_w.k

    np.save(snakemake.output[0], iterations)
    np.save(snakemake.output[1], np.array([np.mean(iterations, axis=0), np.std(iterations, axis=0)]))

