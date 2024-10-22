import os
import qcd_ml
import numpy as np

import torch


config_no = snakemake.wildcards.config_no
fermion_mass = float(snakemake.wildcards.fermion_mass)
n_mini_batches = int(snakemake.wildcards.n_mini_batches)
learn_rate = float(snakemake.wildcards.learn_rate)

L = [8, 8, 8, 16]

# load gauge field
U = torch.tensor(np.load(snakemake.input[0]))

# one hop in every space time direction
paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]

# Parallel Transport Convolution layer
layer = qcd_ml.nn.ptc.v_PTC(1, 1, paths, U)

# Wilson-Clover Dirac operator
w = qcd_ml.qcd.dirac.dirac_wilson_clover(U, fermion_mass, 1)



def complex_mse(output, target):
    err = (output - target)
    return (err * err.conj()).real.sum()

def l2norm(v):
    return (v * v.conj()).real.sum()


optimizer = torch.optim.Adam(layer.parameters(), lr=learn_rate)
loss = np.zeros(n_mini_batches)

for t in range(n_mini_batches):
    src = torch.randn(*L, 4, 3, dtype=torch.cdouble)
    Dsrc = w(src)

    nrm = l2norm(Dsrc) ** 0.5
    inp = torch.stack([Dsrc / nrm])
    out = torch.stack([src / nrm])

    cost = complex_mse(layer.forward(inp), out)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    loss[t] = cost.item()

    if t % 20 == 0:
        print(f"{t:4d} ({t / n_mini_batches * 100: 3.2f} %): {loss[t]:.2e}")

torch.save(layer.state_dict(), snakemake.output[0])
np.save(snakemake.output[1], loss)
