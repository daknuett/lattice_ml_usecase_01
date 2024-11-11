Lattice Machine Learning Usecase
********************************

This usecase/workflow reproduces [1]_ using the pytorch implementation of several lattice
related functions ``qcd_ml`` [2]_. 

Authors: Daniel Knüttel [CA]_ and Simon Pfahler.      
University of Regensburg, Institute for Theoretical Physics.

Funded by the Deutsche Forschungsgemeinschaft (DFG, German Research
Foundation) – Projektnummer 460248186.


.. [CA] daniel.knuettel (at) physik.uni-regensburg.de

.. contents:: Table of Contents
   :depth: 2
   :local:

Getting Started
===============

This usecase is based on the pytorch implementation of the lattice functions in the
``qcd_ml`` package. We recommend to install the requirements in a virtual environment::

    mkvirtualenv lattice_use_case
    pip install -r requirements.txt

Note that you may have to install a specific version of torch depending on your
local setup. Refer to the `pytorch installation guide
<https://pytorch.org/get-started/locally/>`_ for more information.

- In case you need specific hardware support, you may have to compile pytorch from source.
  ``qcd_ml`` is a pure python library and does not require any compilation.
- You may see a warning ``Using slow python implementation of v_pool4d and v_unpool4d (install qcd_ml_accel for faster implementation)``.
  This is not an issue for this usecase because these kernels are not used.

Run::

    snakemake --cores all

to run the entire workflow.

.. note::

    The workflow is designed to run on a single machine. If you want to run it on a cluster,
    you have to adapt the snakemake configuration accordingly.


Workflow
========

The workflow consists of the following steps:

    - Calculation of the multigrid setup (i.e., basis vectors).
    - Training of a small high mode model.
    - Training of the smoother model.
    - Projection of the Dirac operator onto the coarse grid.
    - Training of the coarse grid model.
    - Training of the full model.

For many of these steps, extra information is provided in the form of plots and tables.

The rule dag of the workflow is shown below:

.. image:: _static/dag.pdf
    :width: 100%


References
==========

.. [1] `arXiv:2302.05419 <https://arxiv.org/abs/2302.05419>`_
.. [2] `qcd_ml <https://github.com/daknuett/qcd_ml>`_
