import os
import qcd_ml
import numpy as np

import torch

from qcd_ml.qcd.dirac import dirac_wilson_clover
from qcd_ml.qcd.dirac.coarsened import coarse_9point_op_NG

config_no = snakemake.wildcards.config_no
fermion_mass = float(snakemake.wildcards.fermion_mass)
n_mini_batches = int(snakemake.wildcards.n_mini_batches)
learn_rate = float(snakemake.wildcards.learn_rate)

L = [8, 8, 8, 16]
L_coarse = [2, 2, 2, 4]
n_basis = int(snakemake.wildcards.n_basis)

# load gauge field
U = torch.tensor(np.load(snakemake.input[0]))
state_dict = torch.load(snakemake.input[1], weights_only=True)

coarse_w = coarse_9point_op_NG(state_dict["pseudo_gauge_forward"], state_dict["pseudo_gauge_backward"], state_dict["pseudo_mass"], state_dict["L_coarse"])

paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(3, -1)]]


# create coarse LPTC layer
layer = qcd_ml.nn.lptc.v_LPTC_NG(1, 1, paths, coarse_w.L_coarse, n_basis)

# initialize weights
layer.weights.data = 0.01 * torch.randn_like(layer.weights.data, dtype=torch.cdouble)
layer.weights.data[:,:,0] += torch.eye(n_basis)

# function to calculate mse
def complex_mse(output, target):
    err = (output - target)
    return (err * err.conj()).real.sum()

# function to calculate l2norm
def l2norm(v):
    return (v * v.conj()).real.sum()

optimizer = torch.optim.Adam(layer.parameters(), lr=learn_rate)
loss = np.zeros(n_mini_batches)

for t in range(n_mini_batches):
    src = torch.randn(*L_coarse, n_basis, dtype=torch.cdouble)
    #print("input shape", src.shape)
    Dsrc = coarse_w(src)
    #print("output shape", Dsrc.shape)

    nrm = l2norm(Dsrc) ** 0.5
    inp = torch.stack([Dsrc / nrm])
    out = torch.stack([src / nrm])

    #print("inp, out", inp.shape, out.shape)
    #print(layer.forward(inp).shape)
    cost = complex_mse(layer.forward(inp), out)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    loss[t] = cost.item()

    if t % 20 == 0:
        print(f"{t:4d} ({t / n_mini_batches * 100: 3.2f} %): {loss[t]:.2e}")

torch.save(layer.state_dict(), snakemake.output[0])
np.save(snakemake.output[1], loss)
