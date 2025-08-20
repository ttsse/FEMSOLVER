from mpi4py import MPI
from petsc4py.PETSc import ScalarType
from dolfinx.fem.forms import form as _create_form
from dolfinx.fem.assemble import assemble_scalar
from dolfinx import la, plot

import numpy as np

import ufl
from dolfinx import fem, mesh
from ufl import dx


import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from Pdesolver import AdvectionPDE
from RungeKutta import RungeKutta
from PBC import PeriodicBC
from Visualizer import Visualizer

rank = MPI.COMM_WORLD.rank

# Create a 1D mesh (interval)
msh = mesh.create_interval(MPI.COMM_WORLD, nx=240, points=[-1.0,1.0])

# Define the function space
degree = 1
V = fem.functionspace(msh, ("Lagrange", degree))


tol = 1.0e-5
def pbc_condition(x):  # puppets at x=1
        return np.isclose(x[0], 1.0, atol=tol)

def pbc_relation(x):   # map (1) -> (0)
    y = x.copy()
    y[0] = -1.0
    return y

pbc = PeriodicBC()
V = pbc.create_periodic_condition(V, pbc_condition, pbc_relation)

# Define the initial data
uh = fem.Function(V)
udt = fem.Function(V)
u1 = fem.Function(V)
u2 = fem.Function(V)
mu = fem.Function(V)

#uh.interpolate(lambda x: np.exp(-0.5 * ((x[0])*5)**2))
uh.interpolate(lambda x: 0.5 * (1 - np.tanh((x[0]**2)/(0.3**2) - 1.0)))
uh.x.scatter_forward()


# Final time
T = 2.0

# CFL number
CFL = 0.2

# Define the LHS and RHS terms
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
b = fem.Function(V)
b.interpolate(lambda x: np.full((1, x.shape[1]), ScalarType(1.0)))
b.x.scatter_forward()
a = u * v * dx
L = -b * uh.dx(0) * v * dx - mu * uh.dx(0) * v.dx(0) * dx
Lres = abs(udt + b * uh.dx(0)) * v * dx 

# Call the solver and obtain the solution
PDE = AdvectionPDE(a, L, Lres, uh, bcs=[], correction=-1)


t = 0.0

RK = RungeKutta(PDE)

N = 0
dt = 0.0 
dt1 = 0.0
dt2 = 0.0 

viz = Visualizer()
viz.plot_function(V, uh) 


showtime = 0.0


from dolfinx import io
with io.VTKFile(msh.comm, "results/solution.pvd", "w") as vtk:
    vtk.write_mesh(msh)
    vtk.write_function(uh, t=t)


while t < T - 1.0e-8:
    
    dt = PDE.compute_dt(flux_prime = [b], currenttime=t, finaltime=T, cfl=CFL)

    if N >= 2:
        PDE.compute_BDF(udt, uh, u1, u2, dt1, dt2)
        PDE.compute_viscosity(uh, mu, flux_prime = [b])
    
    N = N + 1


    u2.x.array[:] = u1.x.array[:] 
    u1.x.array[:] = uh.x.array[:] 
    u1.x.scatter_forward()
    u2.x.scatter_forward()

    dt2 = dt1
    dt1 = dt

    uh = RK.SSPRK54(uh,dt)
    t += dt

    if rank == 0:
        print(f"current time: {t}, time step: {dt}")

    """ if t>= showtime:
         viz.plot_function(V, uh) 
         showtime += 0.1 """



viz.plot_function(V, uh, filename="solution.png")  



# --------------------------------------------------
# Reference solution
# --------------------------------------------------
degree_ref = degree + 2
V_ref = fem.functionspace(msh, ("Lagrange", degree_ref))
uref = fem.Function(V_ref)
uref.interpolate(lambda x: 0.5 * (1 - np.tanh((x[0]**2)/(0.3**2) - 1.0)))
uref.x.scatter_forward()

# Error norms of exact solution
L1 = abs(uref) * dx
L2 = abs(uref) * abs(uref) * dx
l1norm_u = assemble_scalar(_create_form(L1))
l2norm_u = np.sqrt(assemble_scalar(_create_form(L2)))
infnorm_u = uref.x.norm(la.Norm.linf)
print(l1norm_u, l2norm_u, infnorm_u)

# Error norms of numerical solution
L1 = abs(uh - uref) * dx
L2 = abs(uh - uref) * abs(uh - uref) * dx
l1errnorm = assemble_scalar(_create_form(L1)) / l1norm_u
l2errnorm = np.sqrt(assemble_scalar(_create_form(L2))) / l2norm_u

uex = fem.Function(V)
uex.interpolate(lambda x: 0.5 * (1 - np.tanh((x[0]**2)/(0.3**2) - 1.0)))
uex.x.scatter_forward()

uh.x.array[:] = uh.x.array - uex.x.array
uh.x.scatter_forward()

inferrnorm = uh.x.norm(la.Norm.linf) / infnorm_u
print(l1errnorm, l2errnorm, inferrnorm)


