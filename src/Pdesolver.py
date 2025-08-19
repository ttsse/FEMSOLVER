from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
from dolfinx.fem import Function
from dolfinx.fem.forms import form as _create_form
from dolfinx.fem.petsc import assemble_matrix_mat, create_matrix, create_vector, assemble_vector
from dolfinx.fem.assemble import apply_lifting, set_bc
import dolfinx, ufl
from ufl import dx


class AdvectionPDE:
    """
    Class encapsulating the assembly and operations for an advection-type PDE.

    Responsibilities:
      - Assembling system matrices and vectors.
      - Computing time step restrictions (CFL condition).
    """

    def __init__(self, a, L, uh, bcs):
        """
        Parameters
        ----------
        a : ufl form
            Bilinear form defining the PDE operator.
        L : ufl form
            Linear form for the system RHS.
        uh : Function
            Current solution function (provides function space).
        bcs : list
            Dirichlet boundary conditions.
        """
        self._a = _create_form(a)
        self._L = _create_form(L)
        self._bcs = bcs
        self._V = uh.function_space
        self.comm = MPI.COMM_WORLD

        self._solver = PETSc.KSP().create(self.comm)

        # ------------------------------------------------------------------
        # Mesh-dependent scaling: compute "h" (cell size function).
        # This is done by solving a mass matrix system for cell diameter / degree.
        self._h = Function(self._V)    

        ah = _create_form(ufl.TrialFunction(self._V) * ufl.TestFunction(self._V) * dx)
        Lh = _create_form(
            ufl.CellDiameter(self._V.mesh) / self._V.element.basix_element.degree
            * ufl.TestFunction(self._V) * dx
        )
        solverh = PETSc.KSP().create(self._V.mesh.comm)
        
        # Assemble mass matrix Ah
        Ah = create_matrix(ah)
        Ah.zeroEntries()
        assemble_matrix_mat(Ah, ah)
        Ah.assemble()
        solverh.setOperators(Ah)

        # Assemble RHS vector bh
        bh = create_vector(Lh)
        with bh.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(bh, Lh)
        bh.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)

        # Solve for h (store as function)
        xh = dolfinx.la.create_petsc_vector_wrap(self._h.x)
        solverh.solve(bh, xh)
        self._h.x.scatter_forward()

        # Compute global minimum mesh size hm (used in CFL)
        loc_hm = 1.0e5
        for i in range(len(self._h.x.array)):
            loc_hm = min(loc_hm, self._h.x.array[i])
        self._hm = self.comm.allreduce(loc_hm, op=MPI.MIN)

        # ------------------------------------------------------------------
        # Define global system matrix and RHS vectors
        self._A = create_matrix(self._a)
        self._b = create_vector(self._L)
        self.assemble_matrix()


    # ----------------------------------------------------------------------
    def assemble_matrix(self):
        """Assemble system matrix A."""
        self._A.zeroEntries()
        assemble_matrix_mat(self._A, self._a, bcs=self._bcs)
        self._A.assemble()
        self._solver.setOperators(self._A)
        
    # ----------------------------------------------------------------------
    def assemble_vector(self):
        """Assemble RHS vector b with boundary conditions applied."""
        with self._b.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(self._b, self._L)

        # Apply boundary conditions
        apply_lifting(self._b, [self._a], bcs=[self._bcs])
        self._b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(self._b, self._bcs) 

    # ----------------------------------------------------------------------
    def compute_dt(self, flux_prime, currenttime, finaltime, cfl):
        """
        Compute time step size from CFL condition.
        """
        fp = flux_prime
        N = len(fp)
        dt_loc = 1.0e5
        
        for i in range(len(self._h.x.array)):
            beta = np.sqrt(sum(fp[j].x.array[i] ** 2 for j in range(N)))
            if beta > 1e-14:  
                dt_loc = min(dt_loc, cfl * self._hm / beta)

        # Global minimum dt
        dt = self.comm.allreduce(dt_loc, op=MPI.MIN)
        
        # Clip to final time
        if currenttime + dt >= finaltime:
            dt = np.abs(finaltime - currenttime)

        return dt

    # ----------------------------------------------------------------------
    def solve(self):
        
        udt = dolfinx.fem.Function(self._V)
        x = dolfinx.la.create_petsc_vector_wrap(udt.x)
        self._solver.solve(self._b, x)
        udt.x.scatter_forward()
        return udt



