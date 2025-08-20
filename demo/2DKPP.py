from mpi4py import MPI
from petsc4py.PETSc import ScalarType  

import numpy as np

import ufl
from dolfinx import fem, mesh, io, plot
from dolfinx.mesh import CellType, GhostMode
from ufl import dx, grad, inner
from dolfinx.io import VTKFile

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from Pdesolver import AdvectionPDE
from RungeKutta import RungeKutta


# --- Problem setup ------------------------------------------------------------
# Structured 2D rectangle: [-2,2] x [-2.5,1.5], 100x100 quads, no ghosting.
msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((-2, -2.5), (2, 1.5)),
    n=(100, 100),
    cell_type=CellType.quadrilateral,
    ghost_mode=GhostMode.none,
)

# Continuous Lagrange space of order 1 (Q1 on quads).
degree = 1
V = fem.functionspace(msh, ("Lagrange", degree))


# --- Boundary conditions ------------------------------------------------------
# Homogeneous Dirichlet on the four outer edges of the rectangle.
tdim = msh.topology.dim
fdim = tdim - 1

def on_square_boundary(x, tol=1e-9):
    return np.logical_or.reduce((
        np.isclose(x[0], -2.0, atol=tol),
        np.isclose(x[0],  2.0, atol=tol),
        np.isclose(x[1], -2.5, atol=tol),
        np.isclose(x[1],  1.5, atol=tol),
    ))

facets = mesh.locate_entities_boundary(msh, fdim, on_square_boundary)
dofs = fem.locate_dofs_topological(V, fdim, facets)
bc = fem.dirichletbc(ScalarType(0), dofs, V)  # u = 0 on boundary


# --- Unknowns, helpers, and initial condition --------------------------------
# uh  : current solution
# udt : time-derivative estimate (used by BDF start-up)
# u1,u2: previous solutions for multi-step initialization
uh  = fem.Function(V)
udt = fem.Function(V)
u1  = fem.Function(V)
u2  = fem.Function(V)

# mu  : artificial viscosity (computed adaptively)
mu = fem.Function(V)

# Radial step function: high value inside unit circle, small outside.
uh.interpolate(lambda x: np.where((x[0] * x[0] + x[1] * x[1]) < 1.0, 3.5 * np.pi, 0.25 * np.pi))
uh.x.scatter_forward()

# b = (cos(uh), -sin(uh)) is the advecting velocity evaluated at uh.
b1 = fem.Function(V)
b2 = fem.Function(V)
for i in range(len(uh.x.array)):
    b1.x.array[i] = np.cos(uh.x.array[i])
    b2.x.array[i] = -np.sin(uh.x.array[i])
b1.x.scatter_forward()
b2.x.scatter_forward()

# Final time and CFL (time step is computed adaptively from flux + mesh size).
T   = 1.0
CFL = 0.2


# --- Variational forms --------------------------------------------------------
# Trial/Test for assembling linear systems/forms.
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# Mass bilinear form (LHS): ∫ u v dx
a = u * v * dx

# Semi-discrete RHS for u_t + b·∇u = div(mu ∇u):
#   - (cos(uh) * ∂x uh) + (sin(uh) * ∂y uh) - ∇·(mu ∇uh)
# Weak form: ∫ [ -cos(uh) * uh_x * v + sin(uh) * uh_y * v - mu ∇uh·∇v ] dx
L = (-ufl.cos(uh) * uh.dx(0) * v * dx
     + ufl.sin(uh) * uh.dx(1) * v * dx
     - mu * inner(grad(uh), grad(v)) * dx)

# Residual measure (for viscosity/step control): |u_t + b·∇u| in L2 against v.
Lres = abs(udt + ufl.cos(uh) * uh.dx(0) - ufl.sin(uh) * uh.dx(1)) * v * dx

# Start with zero artificial viscosity (it will be recomputed as needed).
for i in range(len(mu.x.array)):
    mu.x.array[i] = 0.0


# --- PDE wrapper and time integrator -----------------------------------------
# AdvectionPDE provides: matrix assembly, dt suggestion, BDF bootstrap, viscosity.
PDE = AdvectionPDE(a, L, Lres, uh, bcs=[bc], correction=-1)

# SSP Runge–Kutta(5,4) time-stepping for nonlinear advection–diffusion.
RK = RungeKutta(PDE)


# --- Time loop bookkeeping ----------------------------------------------------
t   = 0.0
udt = fem.Function(V)
N   = 0            # number of accepted steps
dt  = 0.0          # current step
dt1 = 0.0          # previous step
dt2 = 0.0          # step before previous


# --- Output (ParaView) --------------------------------------------------------
# Writes mesh once, then appends solution states; ensure "results_4c" exists.
with io.VTKFile(msh.comm, "results/viscosity.pvd", "w") as vtk:
    vtk.write_mesh(msh)
    vtk.write_function(mu, t=t)

    # --- Main time integration loop ------------------------------------------
    while t < T:
        # Compute stable dt from CFL using the characteristic speeds in b.
        dt = PDE.compute_dt(flux_prime=[b1, b2], currenttime=t, finaltime=T, cfl=CFL)

        # Multi-step warm-up: estimate u_t via BDF and refresh artificial viscosity.
        if N >= 2:
            PDE.compute_BDF(udt, uh, u1, u2, dt1, dt2)
            PDE.compute_viscosity(uh, mu, flux_prime=[b1, b2])

        N += 1

        # Shift solution history: u2 <- u1 <- uh
        u2.x.array[:] = u1.x.array[:]
        u1.x.array[:] = uh.x.array[:]
        u1.x.scatter_forward()
        u2.x.scatter_forward()

        # Slide the time step history as well.
        dt2 = dt1
        dt1 = dt

        # Advance one SSPRK(5,4) step for the nonlinear PDE.
        uh = RK.SSPRK54(uh, dt)

        # Update the advecting velocity b = (cos(uh), -sin(uh)) at the new state.
        for i in range(len(uh.x.array)):
            b1.x.array[i] = np.cos(uh.x.array[i])
            b2.x.array[i] = -np.sin(uh.x.array[i])
        b1.x.scatter_forward()
        b2.x.scatter_forward()

        # Advance physical time, dump state, log step size.
        t += dt
        vtk.write_function(mu, t=t)
        print(t, dt)
