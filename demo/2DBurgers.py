from mpi4py import MPI
from petsc4py.PETSc import ScalarType

import numpy as np

import ufl
from dolfinx import fem, mesh, plot, io
from dolfinx.mesh import CellType, GhostMode
from ufl import dx, grad, inner

import sys
from pathlib import Path

# Make local project modules importable (expects src/ next to this file's parent)
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from Pdesolver import AdvectionPDE          # problem wrapper (assembly, dt, viscosity, BDF)
from Visualizer import Visualizer            # lightweight plotting helper
from RungeKutta import RungeKutta            # SSPRK(5,4) time-stepping

rank = MPI.COMM_WORLD.rank  # for rank-0 logging


# -----------------------------------------------------------------------------
# Mesh and function space
# -----------------------------------------------------------------------------
# 2D triangular mesh on the box [-0.25, 1.75] × [-0.25, 1.75], 100×100 cells.
# GhostMode.none keeps only owned entities (fine for this single-file example).
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((-0.25, -0.25), (1.75, 1.75)),
    n=(100, 100),
    cell_type=CellType.quadrilateral,
    ghost_mode=GhostMode.none,
)

# Continuous Lagrange elements of degree 1 (P1 on triangles)
degree = 1
V = fem.functionspace(msh, ("Lagrange", degree))


# -----------------------------------------------------------------------------
# Boundary conditions: homogeneous Dirichlet on the outer box
# -----------------------------------------------------------------------------
tdim = msh.topology.dim
fdim = tdim - 1  # facet dimension

def on_square_boundary(x, tol=1e-14):
    """Mark all four sides of the rectangular boundary."""
    return np.logical_or.reduce((
        np.isclose(x[0], -0.25, atol=tol),
        np.isclose(x[0],  1.75, atol=tol),
        np.isclose(x[1], -0.25, atol=tol),
        np.isclose(x[1],  1.75, atol=tol),
    ))

facets = mesh.locate_entities_boundary(msh, fdim, on_square_boundary)
dofs = fem.locate_dofs_topological(V, fdim, facets)
bc = fem.dirichletbc(ScalarType(0), dofs, V)  # enforce u = 0 on ∂Ω


# -----------------------------------------------------------------------------
# Unknowns, auxiliaries, and initial condition
# -----------------------------------------------------------------------------
uh  = fem.Function(V)  # current solution u^n
udt = fem.Function(V)  # time-derivative estimate (for BDF bootstrap/viscosity)
u1  = fem.Function(V)  # previous solution u^{n-1}
u2  = fem.Function(V)  # solution two steps back u^{n-2}
mu  = fem.Function(V)  # artificial viscosity field

# Initial condition: piecewise constant square centered at (1,1)
uh.interpolate(lambda x: np.where((np.abs(x[0] - 1.0) <= 0.5) & (np.abs(x[1] - 1.0) <= 0.5), 1.0, -0.75))
uh.x.scatter_forward()  # ensure ghost values (if any) are updated

# Final time and CFL target for adaptive time step selection
T   = 0.75
CFL = 0.2


# -----------------------------------------------------------------------------
# Variational forms
# -----------------------------------------------------------------------------
# Trial/Test symbols
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Mass matrix: ∫ u v dx
a = u * v * dx

# Nonlinear advection-diffusion (Burgers-type) semi-discrete RHS:
#   u_t + u u_x + u u_y = ∇·(mu ∇u)
# Weak form: ∫ [ -u u_x v - u u_y v - mu ∇u·∇v ] dx
L = (-uh * uh.dx(0) * v * dx
     - uh * uh.dx(1) * v * dx
     - mu * inner(grad(uh), grad(v)) * dx)

# Residual magnitude used by viscosity/step control:
#   |u_t + u u_x + u u_y| tested against v
Lres = abs(udt + uh * uh.dx(0) + uh * uh.dx(1)) * v * dx

# Start with zero artificial viscosity (it will be recomputed adaptively)
mu.x.array[:] = 0.0
mu.x.scatter_forward()


# -----------------------------------------------------------------------------
# PDE wrapper and time integrator
# -----------------------------------------------------------------------------
# AdvectionPDE collects forms, boundary conditions, and provides:
#   • compute_dt(...)    : CFL-based time step
#   • compute_BDF(...)   : multi-step time derivative for startup
#   • compute_viscosity(): residual-based artificial viscosity
PDE = AdvectionPDE(a, L, Lres, uh, bcs=[bc], correction=1)

# SSPRK(5,4) strong-stability-preserving Runge–Kutta time stepping
RK = RungeKutta(PDE)


# -----------------------------------------------------------------------------
# Time loop bookkeeping
# -----------------------------------------------------------------------------
t   = 0.0
N   = 0          # number of completed steps
dt  = 0.0        # initial guess (will be overwritten)
dt1 = 0.0        # previous dt
dt2 = 0.0        # dt from two steps back


# -----------------------------------------------------------------------------
# Output (ParaView): write viscosity field over time
# -----------------------------------------------------------------------------
# Note: ensure the folder "results_burgers" exists before running.
with io.VTKFile(msh.comm, "results_burgers/viscosity.pvd", "w") as vtk:
    vtk.write_mesh(msh)
    vtk.write_function(mu, t=t)

    # Main time-integration loop
    while t < T - 1.0e-7:
    
        # Compute stable dt from current field (flux_prime ~ wave speeds)
        dt = PDE.compute_dt(flux_prime=[uh, uh], currenttime=t, finaltime=T, cfl=CFL)

        # After two steps, estimate time derivative and recompute viscosity
        if N >= 2:
            PDE.compute_BDF(udt, uh, u1, u2, dt1, dt2)
            PDE.compute_viscosity(uh, mu, flux_prime=[uh, uh])

        N += 1

        # Shift solution history: u2 <- u1 <- uh
        u2.x.array[:] = u1.x.array[:] 
        u1.x.array[:] = uh.x.array[:] 
        u1.x.scatter_forward()
        u2.x.scatter_forward()

        # Slide stored time steps
        dt2 = dt1
        dt1 = dt

        # Advance one SSPRK(5,4) step
        uh = RK.SSPRK54(uh, dt)

        # Advance physical time, write viscosity snapshot, log progress
        t += dt
        vtk.write_function(mu, t=t)
        if rank == 0:
            print(f"current time: {t:.6f}, time step: {dt:.6e}")


# -----------------------------------------------------------------------------
# Quick visualization (e.g., Matplotlib via helper)
# -----------------------------------------------------------------------------
viz = Visualizer()
viz.plot_function(V, uh)
