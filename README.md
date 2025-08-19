# Finite Element Solver for Conservation Laws

This repository contains a **flexible finite element solver** for nonlinear conservation laws, implemented in **Python with [DOLFINx v0.7.2](https://github.com/FEniCS/dolfinx)**.  

The solver supports:
- Continuous finite element (FE) spatial discretizations (P<sub>k</sub>, Q<sub>k</sub>).  
- Explicit Runge–Kutta and Strong-Stability-Preserving Runge–Kutta (SSP-RK) time integration schemes.  
- Residual-based artificial viscosity for stabilization of shocks/discontinuities.  
- Lumped mass matrix with anti-dispersive correction.  
- Parallel execution with **MPI** (via PETSc).  

The code is designed to be **extensible**: you can define custom conservation laws, polynomial degrees, RK schemes, and stabilization parameters.

---

## 🚀 Features

- Conservation laws (advection, Burgers, rotating wave, etc.).  
- Modular structure for experimenting with different discretizations and stabilization techniques.  
- Parallel runs with `mpirun`(not for PBC).  
- Includes numerical experiments for accuracy and performance.

---

## 📦 Installation

We recommend installing DOLFINx **v0.7.2** using **conda**.  

1. **Create and activate a new environment:**

    conda create -n fenicsx-env
    conda activate fenicsx-env

2. **Install DOLFINx v0.7.2 and dependencies:**

- **Linux/macOS:**
  
    conda install -c conda-forge fenics-dolfinx=0.7.2 mpich pyvista

- **Windows:**
  
    conda install -c conda-forge fenics-dolfinx=0.7.2 pyvista pyamg
    
---



## ✨ Acknowledgments

This project builds on the [FEniCSx](https://github.com/FEniCS/dolfinx) framework and PETSc.  