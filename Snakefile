config_nos = [1200, 1500]
fermion_masses = [-0.6 + 0.037, -0.6 + 0.0412]

# Set to 8 for KNL, 1 for other CPUs
n_threads = 8

config_and_mass = list(f"{cno}_{mass}" for cno, mass in zip(config_nos, fermion_masses))

rule all:
    input: 
        expand("spectra/fine/spectrum_hm_{config_and_mass}.npy",
               config_and_mass=config_and_mass),
        expand("plots/spectra/fine/spectrum_{config_and_mass}.png",
               config_and_mass=config_and_mass),
        expand("models/ptc1h1l/{config_and_mass}_{n_mini_batches}_{learn_rate}.pt",
               config_and_mass=config_and_mass,
               n_mini_batches=[300, 1000],
               learn_rate=[1e-2]),
        expand("plots/cost/ptc1h1l/{config_and_mass}_{n_mini_batches}_{learn_rate}.png",
               config_and_mass=config_and_mass,
               n_mini_batches=[300, 1000],
               learn_rate=[1e-2]),
        expand("iterations/unpreconditioned/{config_and_mass}_{n_statistic}.npy",
               config_and_mass=config_and_mass,
               n_statistic=[20]),
        expand("iterations/preconditioned/ptc1h1l/{config_and_mass}_{n_mini_batches}_{learn_rate}_{n_statistic}.npy",
               config_and_mass=config_and_mass,
               n_mini_batches=[1000],
               learn_rate=[1e-2],
               n_statistic=[20]),
        expand("models/smoother/{config_and_mass}_{n_mini_batches}_{learn_rate}.pt",
               config_and_mass=config_and_mass,
               n_mini_batches=[1000],
               learn_rate=1e-2),
        expand("plots/cost/smoother/{config_and_mass}_{n_mini_batches}_{learn_rate}.png",
               config_and_mass=config_and_mass,
               n_mini_batches=[1000],
               learn_rate=[1e-2]),
        expand("iterations/preconditioned/smoother/{config_and_mass}_{n_mini_batches}_{learn_rate}_{n_statistic}.npy",
               config_and_mass=config_and_mass,
               n_mini_batches=[1000],
               learn_rate=[1e-2],
               n_statistic=[20]),
        expand("iterations/multigrid/{config_and_mass}_{n_basis}_{n_statistic}.npy",
               config_and_mass=config_and_mass,
               n_basis=[12],
               n_statistic=[20]),
        expand("multigrid_setup/multigrid_setup_{config_and_mass}_{n_basis}.pt",
               config_and_mass=config_and_mass,
               n_basis=[12]),
        expand("spectra/multigrid/spectrum_multigrid_{config_and_mass}_{n_basis}.npy",
               config_and_mass=config_and_mass,
               n_basis=[12]),
        expand("plots/spectra/multigrid/spectrum_multigrid_{config_and_mass}_{n_basis}.png",
               config_and_mass=config_and_mass,
               n_basis=[12]),

        expand("models/coarse_lptc/{config_and_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.pt",
               config_and_mass=config_and_mass,
               n_basis=[12], learn_rate=[1e-2], n_mini_batches=[3000]),
        expand("plots/cost/coarse_lptc/{config_and_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.png",
               config_and_mass=config_and_mass,
               n_basis=[12], learn_rate=[1e-2], n_mini_batches=[3000]),


        expand("models/coarse_lptc/{config_and_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.pt",
               config_and_mass=config_and_mass,
               n_basis=[12], learn_rate=[1e-2, 1e-3, 1e-4], n_mini_batches=[3000]),
        expand("plots/cost/coarse_lptc/{config_and_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.png",
               config_and_mass=config_and_mass,
               n_basis=[12], learn_rate=[1e-2, 1e-3, 1e-4], n_mini_batches=[3000]),
        expand("spectra/preconditioned/coarse/spectrum_coarse_lptc_{config_and_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.npy",
               config_and_mass=config_and_mass,
               n_basis=[12], learn_rate=[1e-2, 1e-3, 1e-4], n_mini_batches=[3000]),
        expand("plots/spectra/preconditioned/coarse/spectrum_coarse_lptc_{config_and_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.png",
               config_and_mass=config_and_mass,
               n_basis=[12], learn_rate=[1e-2, 1e-3, 1e-4], n_mini_batches=[3000]),


        expand("models/coarse_lptc/{config_and_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.pt",
               config_and_mass=config_and_mass,
               n_basis=[12], learn_rate=[1e-3, 6e-4], n_mini_batches=[200, 2000, 3000, 6000]),
        expand("plots/cost/coarse_lptc/{config_and_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.png",
               config_and_mass=config_and_mass,
               n_basis=[12], learn_rate=[1e-3, 6e-4], n_mini_batches=[200, 2000, 3000, 6000]),
        expand("spectra/preconditioned/coarse/spectrum_coarse_lptc_{config_and_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.npy",
               config_and_mass=config_and_mass,
               n_basis=[12], learn_rate=[1e-3, 6e-4], n_mini_batches=[200, 2000, 3000, 6000]),
        expand("plots/spectra/preconditioned/coarse/spectrum_coarse_lptc_{config_and_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.png",
               config_and_mass=config_and_mass,
               n_basis=[12], learn_rate=[1e-3, 6e-4], n_mini_batches=[200, 2000, 3000, 6000]),

        expand("iterations/coarse/unpreconditioned/{config_and_mass}_{n_basis}_{n_statistic}.npy",
               config_and_mass=config_and_mass,
               n_basis=[12], n_statistic=[20]),
        expand("iterations/coarse/preconditioned/{config_and_mass}_{n_mini_batches}_{learn_rate}_{n_basis}_{n_statistic}.npy",
               config_and_mass=config_and_mass,
               n_basis=[12], learn_rate=[0.001, 0.001, 0.0006], n_mini_batches=[3000, 6000, 6000], n_statistic=[20]),

        expand("models/full_model/{config_and_mass}_{n_mini_batches_fine}_{learn_rate_fine}_{n_mini_batches_coarse}_{learn_rate_coarse}_{n_mini_batches}_{learn_rate}_{n_basis}.pt",
               config_and_mass=config_and_mass,
               n_basis=[12], learn_rate=[1e-3], n_mini_batches=[200], n_mini_batches_fine=[1000], learn_rate_fine=[1e-2], learn_rate_coarse=[1e-3], n_mini_batches_coarse=[6000]),

rule train_ptc1h1l:
    threads: n_threads
    input:
        "../test/assets/{config_no}.config.npy"
    output:
        "models/ptc1h1l/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}.pt",
        "cost/ptc1h1l/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}.npy"
    script:
        "scripts/training/ptc1h1l.py"

rule plot_cost_ptc1h1l:
    threads: n_threads
    input:
        "cost/ptc1h1l/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}.npy"
    output:
        "plots/cost/ptc1h1l/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}.png"
    script:
        "scripts/plots/loss_ptc1h1l.py"

rule plot_cost_smoother:
    threads: n_threads
    input:
        "cost/smoother/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}.npy"
    output:
        "plots/cost/smoother/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}.png"
    script:
        "scripts/plots/loss_smoother.py"

rule get_iteration_unpreconditioned:
    threads: n_threads
    input: 
        "../test/assets/{config_no}.config.npy"
    output:
        "iterations/unpreconditioned/{config_no}_{fermion_mass}_{n_statistic}.npy",
        "iterations/unpreconditioned/{config_no}_{fermion_mass}_{n_statistic}_mean_std.npy"
    script:
        "scripts/solvers/unpreconditioned.py"


rule get_iteration_ptc1h1l:
    threads: n_threads
    input: 
        "../test/assets/{config_no}.config.npy",
        "models/ptc1h1l/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}.pt"
    output:
        "iterations/preconditioned/ptc1h1l/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_statistic}.npy",
        "iterations/preconditioned/ptc1h1l/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_statistic}_mean_std.npy"
    script:
        "scripts/solvers/preconditioned_ptc1h1l.py"

rule get_iteration_smoother:
    threads: n_threads
    input: 
        "../test/assets/{config_no}.config.npy",
        "models/smoother/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}.pt"
    output:
        "iterations/preconditioned/smoother/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_statistic}.npy",
        "iterations/preconditioned/smoother/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_statistic}_mean_std.npy"
    script:
        "scripts/solvers/preconditioned_smoother.py"

rule train_smoother:
    threads: 1
    input:
        "../test/assets/{config_no}.config.npy",
        "models/ptc1h1l/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}.pt"
    output:
        "models/smoother/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}.pt",
        "cost/smoother/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}.npy"
    script:
        "scripts/training/train_smoother.py"


rule multigrid_setup:
    threads: n_threads 
    input: 
        "../test/assets/{config_no}.config.npy"
    output:
        "multigrid_setup/multigrid_setup_{config_no}_{fermion_mass}_{n_basis}.pt"
    script:
        "scripts/get_multigrid.py"

rule get_coarse_operator:
    threads: n_threads
    input:
        "../test/assets/{config_no}.config.npy",
        "multigrid_setup/multigrid_setup_{config_no}_{fermion_mass}_{n_basis}.pt"
    output:
        "operators/coarse/coarse_operator_{config_no}_{fermion_mass}_{n_basis}.npy"
    script:
        "scripts/get_coarse_operator.py"

rule spectrum_mg:
    threads: 1 
    input: 
        "../test/assets/{config_no}.config.npy",
        "operators/coarse/coarse_operator_{config_no}_{fermion_mass}_{n_basis}.npy"
    output:
        "spectra/multigrid/spectrum_multigrid_{config_no}_{fermion_mass}_{n_basis}.npy"
    script:
        "scripts/spectra/compute_spectrum_mg.py"


rule plot_spectrum_mg:
    threads: 1
    input:
        "spectra/multigrid/spectrum_multigrid_{config_no}_{fermion_mass}_{n_basis}.npy"
    output:
        "plots/spectra/multigrid/spectrum_multigrid_{config_no}_{fermion_mass}_{n_basis}.png"
    script:
        "scripts/plots/plot_spectrum_mg.py"

rule train_coarse_lptc:
    threads: n_threads
    input:
        "../test/assets/{config_no}.config.npy",
        "operators/coarse/coarse_operator_{config_no}_{fermion_mass}_{n_basis}.npy"
    output:
        "models/coarse_lptc/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.pt",
        "cost/coarse_lptc/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.npy"
    script:
        "scripts/training/train_coarse.py"

rule plot_cost_coarse_lptc:
    threads: 1
    input:
        "cost/coarse_lptc/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.npy"
    output:
        "plots/cost/coarse_lptc/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.png"
    script:
        "scripts/plots/loss_coarse.py"

rule spectrum_prec_coarse:
    threads: 1
    input:
        "../test/assets/{config_no}.config.npy",
        "operators/coarse/coarse_operator_{config_no}_{fermion_mass}_{n_basis}.npy",
        "models/coarse_lptc/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.pt"
    output:
        "spectra/preconditioned/coarse/spectrum_coarse_lptc_{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.npy"
    script:
        "scripts/spectra/preconditioned/compute_spectrum_coarse_preconditioned.py"


rule plot_spectrum_prec_coarse:
    threads: 1
    input:
        "spectra/multigrid/spectrum_multigrid_{config_no}_{fermion_mass}_{n_basis}.npy",
        "spectra/preconditioned/coarse/spectrum_coarse_lptc_{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.npy"
    output:
        "plots/spectra/preconditioned/coarse/spectrum_coarse_lptc_{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.png"
    script:
        "scripts/plots/plot_spectrum_preconditioned_coarse.py"

rule spectrum_fine_lm:
    threads: 1
    input:
        "../test/assets/{config_no}.config.npy"
    output:
        "spectra/fine/spectrum_lm_{config_no}_{fermion_mass}.npy"
    script:
        "scripts/spectra/compute_spectrum_fine_lm.py"

rule spectrum_fine_hm:
    threads: 1
    input:
        "../test/assets/{config_no}.config.npy"
    output:
        "spectra/fine/spectrum_hm_{config_no}_{fermion_mass}.npy"
    script:
        "scripts/spectra/compute_spectrum_fine_hm.py"

rule plot_spectrum_fine:
    threads: 1
    input:
        "spectra/fine/spectrum_hm_{config_no}_{fermion_mass}.npy",
        "spectra/fine/spectrum_lm_{config_no}_{fermion_mass}.npy"
    output:
        "plots/spectra/fine/spectrum_{config_no}_{fermion_mass}.png"
    script:
        "scripts/plots/plot_spectrum_fine.py"

rule get_iterations_multigrid:
    threads: n_threads
    input:
        "../test/assets/{config_no}.config.npy",
        "multigrid_setup/multigrid_setup_{config_no}_{fermion_mass}_{n_basis}.pt",
        "operators/coarse/coarse_operator_{config_no}_{fermion_mass}_{n_basis}.npy"
    output:
        "iterations/multigrid/{config_no}_{fermion_mass}_{n_basis}_{n_statistic}.npy",
        "iterations/multigrid/{config_no}_{fermion_mass}_{n_basis}_{n_statistic}_mean_std.npy"
    script:
        "scripts/solvers/preconditioned_mg.py"

rule get_iterations_coarse_unpreconditioned:
    threads: 1
    input:
        "../test/assets/{config_no}.config.npy",
        "operators/coarse/coarse_operator_{config_no}_{fermion_mass}_{n_basis}.npy"
    output:
        "iterations/coarse/unpreconditioned/{config_no}_{fermion_mass}_{n_basis}_{n_statistic}.npy",
        "iterations/coarse/unpreconditioned/{config_no}_{fermion_mass}_{n_basis}_{n_statistic}_mean_std.npy"
    script:
        "scripts/solvers/coarse/unpreconditioned.py"

rule get_iterations_coarse_preconditioned:
    threads: 1
    input:
        "../test/assets/{config_no}.config.npy",
        "operators/coarse/coarse_operator_{config_no}_{fermion_mass}_{n_basis}.npy",
        "models/coarse_lptc/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_basis}.pt"
    output:
        "iterations/coarse/preconditioned/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_basis}_{n_statistic}.npy",
        "iterations/coarse/preconditioned/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_basis}_{n_statistic}_mean_std.npy"
    script:
        "scripts/solvers/coarse/preconditioned_lptc1h1l.py"


rule plot_iteration_comparison:
    threads: 1
    input:
        "iterations/unpreconditioned/{config_no}_{fermion_mass}_{n_statistic}.npy",
        "iterations/preconditioned/ptc1h1l/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_statistic}.npy",
        "iterations/preconditioned/smoother/{config_no}_{fermion_mass}_{n_mini_batches}_{learn_rate}_{n_statistic}.npy",
        "iterations/multigrid/{config_no}_{fermion_mass}_{n_basis}_{n_statistic}.npy",
        "iterations/coarse/unpreconditioned/{config_no}_{fermion_mass}_{n_basis}_{n_statistic}.npy",
        "iterations/coarse/preconditioned/{config_no}_{fermion_mass}_{n_mini_batches_coarse}_{learn_rate_coarse}_{n_basis}_{n_statistic}.npy"
    output:
        "plots/iterations/{config_no}_{fermion_mass}_{n_statistic}.png"
    notebook:
        "scripts/notebooks/iteration_comparison.ipynb"

rule train_full_model:
    threads: n_threads
    input:
        "../test/assets/{config_no}.config.npy",
        "multigrid_setup/multigrid_setup_{config_no}_{fermion_mass}_{n_basis}.pt",
        "operators/coarse/coarse_operator_{config_no}_{fermion_mass}_{n_basis}.npy",
        "models/smoother/{config_no}_{fermion_mass}_{n_mini_batches_fine}_{learn_rate_fine}.pt",
        "models/coarse_lptc/{config_no}_{fermion_mass}_{n_mini_batches_coarse}_{learn_rate_coarse}_{n_basis}.pt"
    output:
        "models/full_model/{config_no}_{fermion_mass}_{n_mini_batches_fine}_{learn_rate_fine}_{n_mini_batches_coarse}_{learn_rate_coarse}_{n_mini_batches}_{learn_rate}_{n_basis}.pt",
        "cost/full_model/{config_no}_{fermion_mass}_{n_mini_batches_fine}_{learn_rate_fine}_{n_mini_batches_coarse}_{learn_rate_coarse}_{n_mini_batches}_{learn_rate}_{n_basis}.npy"
    log:
        "logs/full_model/{config_no}_{fermion_mass}_{n_mini_batches_fine}_{learn_rate_fine}_{n_mini_batches_coarse}_{learn_rate_coarse}_{n_mini_batches}_{learn_rate}_{n_basis}.log"
    script:
        "scripts/training/full_model.py"
