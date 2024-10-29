config_nos = [1200, 1500]
fermion_masses = [-0.6 + 0.037, -0.6 + 0.0412]

# Set to 8 for KNL, 1 for other CPUs
n_threads = 8

config_and_mass = list(f"{cno}_{mass}" for cno, mass in zip(config_nos, fermion_masses))

rule all:
    input: 
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


rule spectrum_mg:
    threads: 1 
    input: 
        "../test/assets/{config_no}.config.npy",
        "multigrid_setup/multigrid_setup_{config_no}_{fermion_mass}_{n_basis}.pt"
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
        "multigrid_setup/multigrid_setup_{config_no}_{fermion_mass}_{n_basis}.pt"
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
        "multigrid_setup/multigrid_setup_{config_no}_{fermion_mass}_{n_basis}.pt",
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
