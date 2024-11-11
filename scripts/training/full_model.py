import sys

with open(snakemake.log[0], "w") as f:
    sys.stdout = f
    sys.stderr = f

    import datetime
    def log(*args, **kwargs):
        now = datetime.datetime.now()
        print(now.strftime("[%Y-%m-%d %H:%M:%S]"), *args, **kwargs)

    import os
    log(os.system("uname -a"))
    # Logging with snakemake...


    import qcd_ml
    import numpy as np

    import torch

    from qcd_ml.qcd.dirac import dirac_wilson_clover
    from qcd_ml.qcd.dirac.coarsened import coarse_9point_op_NG
    from qcd_ml.util.qcd.multigrid import ZPP_Multigrid


    def complex_mse(output, target):
        err = (output - target)
        return (err * err.conj()).real.sum()

    def l2norm(v):
        return (v * v.conj()).real.sum()

    config_no = snakemake.wildcards.config_no
    fermion_mass = float(snakemake.wildcards.fermion_mass)
    n_mini_batches = int(snakemake.wildcards.n_mini_batches)
    learn_rate = float(snakemake.wildcards.learn_rate)

    L = [8, 8, 8, 16]
    L_coarse = [2, 2, 2, 4]
    n_basis = int(snakemake.wildcards.n_basis)

    # load gauge field
    U = torch.tensor(np.load(snakemake.input[0]))
    mg = ZPP_Multigrid.load(snakemake.input[1])

    state_dict = torch.load(snakemake.input[2], weights_only=True)
    coarse_w = coarse_9point_op_NG(state_dict["pseudo_gauge_forward"], state_dict["pseudo_gauge_backward"], state_dict["pseudo_mass"], state_dict["L_coarse"])

    paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(mu, -1)] for mu in range(4)]
    coarse_paths = [[]] + [[(mu, 1)] for mu in range(4)] + [[(3, -1)]]

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

    class FullModel:
        def __init__(self, U, mg, smoother_paths, coarse_paths, L_coarse, n_basis):
            self.U = U
            self.mg = mg
            self.smoother_paths = smoother_paths
            self.coarse_paths = coarse_paths
            self.L_coarse = L_coarse
            self.n_basis = n_basis

            self.smoother = Smoother(self.U, self.smoother_paths)

            self.coarse_solver = qcd_ml.nn.lptc.v_LPTC_NG(1, 1, coarse_paths, L_coarse, n_basis)

        def forward(self, v):
            if v.shape[0] != 1:
                raise ValueError("only single features are supported")
            v_coarse = self.mg.v_project(v[0])
            v_coarse = self.coarse_solver.forward(torch.stack([v_coarse]))[0]
            v_lowmodes = self.mg.v_prolong(v_coarse)

            return self.smoother.forward(torch.stack([v[0], v_lowmodes])) 


    full_model = FullModel(U, mg, paths, coarse_paths, L_coarse, n_basis)

    # initialize weights
    full_model.smoother.load_state_dict(torch.load(snakemake.input[3], weights_only=True))
    full_model.coarse_solver.load_state_dict(torch.load(snakemake.input[4], weights_only=True))


    w = dirac_wilson_clover(U, fermion_mass, 1)

    class Lvl2MultigridPreconditioner:    
        def __init__(self, q, mg, q_coarse, inner_solver_kwargs, smoother_kwargs):    
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

    mg_prec = Lvl2MultigridPreconditioner(counting_w, mg, counting_w_coarse, {"eps": 5e-2, "maxiter": 50, "inner_iter": 25}, {"eps": 1e-15, "maxiter": 8, "inner_iter": 4})

    def Dinv(x):
        x, info = GMRES(w, x, torch.clone(x), eps=1e-5, maxiter=3000, preconditioner=mg_prec)
        log(f"{config_no} Dinv: iters: {info['k']}, residual: {info['res']}, converged: {info['converged']}")
        return x

    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    cost_record = np.zeros((n_train, 3))
    for t in range(n_mini_batches):
        src = torch.randn(*L, 4, 3, dtype=torch.cdouble)

        Dsrc = w(src)

        nrm = l2norm(Dsrc) ** 0.5
        inp = torch.stack([Dsrc / nrm])
        out = torch.stack([src / nrm])

        cost1 = complex_mse(full_model.forward(inp), out)

        src2 = torch.randn(*L, 4, 3, dtype=torch.cdouble)
        with torch.no_grad():
            source2 = torch.randn_like(source)
            source2 /= l2norm(source2)**0.5
            Dinvsource = Dinv(source2)

        cost_2 = complex_mse(model.forward(torch.stack([source2])), torch.stack([Dinvsource]))

        cost = cost_1 + cost_2

        cost_record[t, 0] = cost_1.item()
        cost_record[t, 1] = cost_2.item()
        cost_record[t, 2] = cost.item()
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        if t % 100 == 0:
            log(f"t = {t}, cost = {cost_record[t,2]}")


    torch.save(model.state_dict(), snakemake.output[0])
    np.save(snakemake.output[1], cost_record)
