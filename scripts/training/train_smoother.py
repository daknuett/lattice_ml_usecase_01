import qcd_ml
import torch
import numpy as np

U = torch.tensor(np.load(snakemake.input[0]))

fermion_mass = float(snakemake.wildcards.fermion_mass)
n_mini_batches = int(snakemake.wildcards.n_mini_batches)
learn_rate = float(snakemake.wildcards.learn_rate)

# 1 hop paths
paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]

# create PTC layer
layer = qcd_ml.nn.ptc.v_PTC(1, 1, paths, U)
layer.load_state_dict(torch.load(snakemake.input[1]))


def ukn(b, mh, w, n):
    uk = torch.zeros_like(b)

    for k in range(n):
        result = b
        result -= w(uk)
        result = mh(result)
        uk += result
    return uk


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

# create the smoother model
model = Smoother(U, paths)

# initialize weights
for li in [model.l0, model.l1, model.l2, model.l3]:
    li.weights.data = 0.001 * torch.randn_like(li.weights.data, dtype=torch.cdouble)
    li.weights.data[:,:,0] += torch.eye(4)

w = qcd_ml.qcd.dirac.dirac_wilson_clover(U, fermion_mass, 1)


# function to calculate mse
def complex_mse(output, target):
    err = (output - target)
    return (err * err.conj()).real.sum()

# function to calculate l2norm
def l2norm(v):
    return (v * v.conj()).real.sum()

optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)

cost = np.zeros(n_mini_batches)

null = torch.zeros(8,8,8,16,4,3, dtype=torch.cdouble)
mh = lambda x: layer.forward(torch.stack([x]))[0]

for t in range(n_mini_batches):
    source = torch.randn(8,8,8,16,4,3, dtype=torch.cdouble)
    source /= l2norm(source)

    in_vec = torch.stack([source, null])
    out_vec = torch.stack([ukn(source, mh, w, 2)])

    curr_cost = complex_mse(model.forward(in_vec), out_vec)
    cost[t] = curr_cost.item()
    optimizer.zero_grad()
    curr_cost.backward()
    optimizer.step()


torch.save(model.state_dict(), snakemake.output[0])
np.save(snakemake.output[1], cost)
