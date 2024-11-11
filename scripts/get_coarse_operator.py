import torch
import numpy as np

from qcd_ml.qcd.dirac import dirac_wilson_clover
from qcd_ml.util.qcd.multigrid import ZPP_Multigrid

from qcd_ml.qcd.dirac.coarsened import coarse_9point_op_NG

from scipy import sparse
import scipy

U = torch.tensor(np.load(snakemake.input[0]))
mg = ZPP_Multigrid.load(snakemake.input[1])
fermion_mass = float(snakemake.wildcards.fermion_mass)

w = dirac_wilson_clover(U, fermion_mass, 1)

w_coarse_mg = coarse_9point_op_NG.from_operator_and_multigrid(w, mg)

torch.save({
                "pseudo_mass": w_coarse_mg.pseudo_mass
                , "pseudo_gauge_forward": w_coarse_mg.pseudo_gauge_forward
                , "pseudo_gauge_backward": w_coarse_mg.pseudo_gauge_backward
                , "L_coarse": w_coarse_mg.L_coarse
            }
           , snakemake.output[0])
