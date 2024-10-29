import os
import qcd_ml
import numpy as np

import torch


config_no = snakemake.wildcards.config_no
fermion_mass = float(snakemake.wildcards.fermion_mass)
n_mini_batches = int(snakemake.wildcards.n_mini_batches)
learn_rate = float(snakemake.wildcards.learn_rate)
n_statistic = int(snakemake.wildcards.n_statistic)

L = [8, 8, 8, 16]

# Smoother model
class Smoother(torch.nn.Module):
    def __init__(self, U, paths):
        super(Smoother, self).__init__()

        self.U = U
        self.paths = paths
        self.l0 = qcd_ml.nn.ptc.v_PTC(2,2,self.paths,self.U)
        self.l1 = qcd_ml.nn.ptc.v_PTC(2,2,self.paths,self.U)
        self.l2 = qcd_ml.nn.ptc.v_PTC(2,2,self.paths,self.U)
        self.l3 = qcd_ml.nn.ptc.v_PTC(2,1,self.paths,self.U)

    def forward(self, v):
        v = self.l0(v)
        v = self.l1(v)
        v = self.l2(v)
        v = self.l3(v)
        return v

# load gauge field
U = torch.tensor(np.load(snakemake.input[0]))

# one hop in every space time direction
paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]

# create the smoother model
model = Smoother(U, paths)

# Wilson-Clover Dirac operator
w = qcd_ml.qcd.dirac.dirac_wilson_clover(U, fermion_mass, 1)

print(snakemake.input[1])
model.load_state_dict(torch.load(snakemake.input[1]))

null = torch.zeros(8, 8, 8, 16, 4, 3, dtype=torch.cdouble)

with torch.no_grad():
    src = torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.cdouble)

    iterations = np.zeros(n_statistic)
    for i in range(n_statistic):
        src = torch.randn_like(src)

        x, info = qcd_ml.util.solver.GMRES(w, src, torch.clone(src), eps=1e-6, maxiter=60000, inner_iter=30, preconditioner=lambda x: model.forward(torch.stack([x, null]))[0])
        print(f"{i}: {info['k']}, residual: {info['res']}")
        iterations[i] = info["k"]

    np.save(snakemake.output[0], iterations)
    np.save(snakemake.output[1], np.array([np.mean(iterations), np.std(iterations)]))

