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
               n_mini_batches=[300, 1000],
               learn_rate=[1e-2],
               n_statistic=[20]),

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



rule multigrid_setup:
    threads: n_threads 
    input: 
        "configs/config_{config_no}.pt"
    output:
        "multigrid_setup/multigrid_setup_{config_no}_{fermion_mass}_{n_basis}.pt"
    script:
        "scripts/get_multigrid_setup.py"
