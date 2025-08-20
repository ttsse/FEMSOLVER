from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx import fem, mesh, la
from dolfinx.fem.forms import form as _create_form
from dolfinx.fem.assemble import assemble_scalar
import numpy as np
import ufl
from ufl import dx, inner, grad

import gmsh
from dolfinx.io import gmshio

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))
from Pdesolver import AdvectionPDE
from RungeKutta import RungeKutta
from Visualizer import Visualizer

# --------------------------------------------------
# MPI setup
# --------------------------------------------------
rank = MPI.COMM_WORLD.rank
comm = MPI.COMM_WORLD

# --------------------------------------------------
# Load mesh from .msh file (exported previously by Gmsh)
# --------------------------------------------------
if rank == 0:
    gmsh.initialize()
    gmsh.open("demo/2DLinear/circle.msh")
    model = gmsh.model
else:
    model = None

msh, cell_tags, facet_tags = gmshio.model_to_mesh(model, comm, rank=0)

if rank == 0:
    gmsh.finalize()

# --------------------------------------------------
# Define function spaces
# --------------------------------------------------
degree = 1
V = fem.functionspace(msh, ("Lagrange", degree))

# Reference space (create it early to avoid communicator overflow)
degree_ref = degree + 2
V_ref = fem.functionspace(msh, ("Lagrange", degree_ref))

# -----------------------------------------------------------------------------
# Boundary conditions: homogeneous Dirichlet on the outer circle
# -----------------------------------------------------------------------------
tdim = msh.topology.dim
fdim = tdim - 1  # facet dimension

def on_circle_boundary(x, tol=1e-8):
    r = np.sqrt(x[0]**2 + x[1]**2)
    return np.isclose(r, 1.0, atol=tol)   # unit circle

facets = mesh.locate_entities_boundary(msh, fdim, on_circle_boundary)
bc_dofs = fem.locate_dofs_topological(V, fdim, facets)
bc = fem.dirichletbc(ScalarType(0), bc_dofs, V)

# --------------------------------------------------
# Initial condition
# --------------------------------------------------
uh = fem.Function(V)
udt = fem.Function(V)
u1 = fem.Function(V)
u2 = fem.Function(V)
mu = fem.Function(V)

x0 = 0.4
a = 0.3
uh.interpolate(lambda x: 0.5 * (1 - np.tanh(((x[0]-x0)**2 + x[1]**2)/(a**2) - 1.0)))
uh.x.scatter_forward()

# --------------------------------------------------
# PDE setup
# --------------------------------------------------
T = 1.0
CFL = 0.2

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

b1 = fem.Function(V)
b1.interpolate(lambda x: -2*np.pi*x[1])
b1.x.scatter_forward()

b2 = fem.Function(V)
b2.interpolate(lambda x: 2*np.pi*x[0])
b2.x.scatter_forward()

a_form = u * v * dx
L_form = (-b1 * uh.dx(0) - b2 * uh.dx(1)) * v * dx - mu * inner(grad(uh), grad(v))*dx
Lres_form = abs(udt + b1 * uh.dx(0) + b2 * uh.dx(1)) * v * dx 

mu.x.array[:] = 0.0
mu.x.scatter_forward()

PDE = AdvectionPDE(a_form, L_form, Lres_form, uh, bcs=[bc], correction=4)
RK = RungeKutta(PDE)

# --------------------------------------------------
# Time stepping
# --------------------------------------------------
t = 0.0
N = 0
dt1 = dt2 = 0.0

viz = Visualizer()
viz.plot_function(V, uh, filename="initial.png")

while t < T - 1.0e-8:
    dt = PDE.compute_dt(flux_prime=[b1, b2], currenttime=t, finaltime=T, cfl=CFL)

    if N >= 2:
        PDE.compute_BDF(udt, uh, u1, u2, dt1, dt2)
        PDE.compute_viscosity(uh, mu, flux_prime=[b1, b2])

    # Shift old solutions
    u2.x.array[:] = u1.x.array[:] 
    u1.x.array[:] = uh.x.array[:] 
    u1.x.scatter_forward()
    u2.x.scatter_forward()

    dt2 = dt1
    dt1 = dt

    uh = RK.RK4(uh, dt)
    t += dt
    N += 1

    if rank == 0:
        print(f"current time: {t}, time step: {dt}")

viz.plot_function(V, uh, filename="solution.png")




# --------------------------------------------------
# Reference solution and error norms
# --------------------------------------------------
""" uref = fem.Function(V_ref)
uref.interpolate(lambda x: 0.5 * (1 - np.tanh(((x[0]-0.4)**2 + x[1]**2)/(0.3**2) - 1.0)))
uref.x.scatter_forward()

# Norms of exact solution
L1 = abs(uref) * dx
L2 = abs(uref) * abs(uref) * dx
l1norm_u = assemble_scalar(_create_form(L1))
l2norm_u = np.sqrt(assemble_scalar(_create_form(L2)))
infnorm_u = uref.x.norm(la.Norm.linf)

if rank == 0:
    print("Reference norms:", l1norm_u, l2norm_u, infnorm_u)

# Error norms
L1_err = abs(uh - uref) * dx
L2_err = abs(uh - uref) * abs(uh - uref) * dx
l1errnorm = assemble_scalar(_create_form(L1_err)) / l1norm_u
l2errnorm = np.sqrt(assemble_scalar(_create_form(L2_err))) / l2norm_u

# For Linf error we need the same space as uh
uex = fem.Function(V)
uex.interpolate(lambda x: 0.5 * (1 - np.tanh(((x[0]-0.4)**2 + x[1]**2)/(0.3**2) - 1.0)))
uex.x.scatter_forward()

diff = fem.Function(V)
diff.x.array[:] = uh.x.array - uex.x.array
diff.x.scatter_forward()

inferrnorm = diff.x.norm(la.Norm.linf) / infnorm_u

if rank == 0:
    print("Relative errors:", l1errnorm, l2errnorm, inferrnorm) """
