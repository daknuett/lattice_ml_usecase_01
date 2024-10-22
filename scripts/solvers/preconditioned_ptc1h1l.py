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

with torch.no_grad():

    # load gauge field
    U = torch.tensor(np.load(snakemake.input[0]))

    # one hop in every space time direction
    paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]

    # Parallel Transport Convolution layer
    layer = qcd_ml.nn.ptc.v_PTC(1, 1, paths, U)

    # Wilson-Clover Dirac operator
    w = qcd_ml.qcd.dirac.dirac_wilson_clover(U, fermion_mass, 1)

    layer.load_state_dict(torch.load(snakemake.input[1]))


    src = torch.randn(8, 8, 8, 16, 4, 3, dtype=torch.cdouble)

    iterations = np.zeros(n_statistic)
    for i in range(n_statistic):
        src = torch.randn_like(src)

        x, info = qcd_ml.util.solver.GMRES(w, src, torch.clone(src), eps=1e-6, maxiter=60000, inner_iter=30, preconditioner=lambda x: layer.forward(torch.stack([x]))[0])
        print(f"{i}: {info['k']}, residual: {info['res']}")
        iterations[i] = info["k"]

    np.save(snakemake.output[0], iterations)
    np.save(snakemake.output[1], np.array([np.mean(iterations), np.std(iterations)]))
