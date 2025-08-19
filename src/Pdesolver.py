from petsc4py import PETSc
from mpi4py import MPI
import numpy as np
from dolfinx.fem import Function
from dolfinx.fem.forms import form as _create_form
from dolfinx.fem.petsc import assemble_matrix_mat, create_matrix, create_vector, assemble_vector
from dolfinx.fem.assemble import apply_lifting, set_bc
import dolfinx, ufl
from ufl import dx
from Dofmap import DofMappings
from Lumpedmass import LumpedMass

class AdvectionPDE:
    """
    Class encapsulating the assembly and operations for an advection-type PDE.

    Responsibilities:
      - Assembling system matrices and vectors.
      - Computing time step restrictions (CFL condition).
      - Computing viscosity terms (for stabilization).
      - Computing residuals and BDF approximations.
    """

    def __init__(self, a, L, Lres, uh, bcs, correction):
        """
        Parameters
        ----------
        a : ufl form
            Bilinear form defining the PDE operator.
        L : ufl form
            Linear form for the system RHS.
        Lres : ufl form
            Residual form (used in viscosity calculation).
        uh : Function
            Current solution function (provides function space).
        bcs : list
            Dirichlet boundary conditions.
        """
        self._a = _create_form(a)
        self._L = _create_form(L)
        self._Lres = _create_form(Lres)
        self._bcs = bcs
        self._V = uh.function_space
        self.correction = correction
        self.comm = MPI.COMM_WORLD


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
        self._bres = create_vector(self._Lres)
        self.assemble_matrix()

        # Dof mapping (cell → dofs)
        self._DM = DofMappings().get_cell_dofs(self._V.mesh, self._V)

        # Lumped mass operator (with optional correction)
        self._LM = LumpedMass(self._A, correction)


    # ----------------------------------------------------------------------
    def assemble_matrix(self):
        """Assemble system matrix A."""
        self._A.zeroEntries()
        assemble_matrix_mat(self._A, self._a, bcs=self._bcs)
        self._A.assemble()

        
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
        
        """
        Solve mass system (consistent or lumped) for current RHS vector.
        """
        udt = self._LM.get_dk(self._b, self._V)
        return udt
    
    # ----------------------------------------------------------------------
    def compute_BDF(self, udt, u, u1, u2, kn, kn0):
        """
        Compute BDF approximation of the time derivative using three solutions.

        Parameters
        ----------
        u, u1, u2 : Functions
            Current and previous time step solutions.
        kn, kn0 : float
            Time step sizes.
        """        
        dt1 = kn
        dt2 = kn + kn0
        C1 = (dt2 * dt2 - dt1 * dt1) / (dt1 * dt2)
        C2 = -dt2 / dt1
        C3 = dt1 / dt2

        for i in range(len(udt.x.array)):
            udt.x.array[i] = (
                C1 * u.x.array[i]
                + C2 * u1.x.array[i]
                + C3 * u2.x.array[i]
            ) / (dt2 - dt1)

    # ----------------------------------------------------------------------
    def Linf_u(self, u):
        """
        Compute L-infinity norm of u relative to its global average.
        """
        usum = 0.0
        for i in range(len(u.x.array)):
            usum += u.x.array[i]

        # Compute global average
        sum_glb = self.comm.allreduce(usum, op=MPI.SUM)
        avg_glb = sum_glb / u.vector.getSize()

        # Local max deviation
        umax  = -1.0e5
        for i in range(len(u.x.array)):
            umax = max(umax, np.abs(u.x.array[i] - avg_glb))

        # Global max
        max_glb = self.comm.allreduce(umax, op=MPI.MAX)
        return max_glb
        
    # ----------------------------------------------------------------------
    def compute_residual(self):
        """
        Assemble residual vector and divide by lumped mass to get residual function.
        """
        with self._bres.localForm() as b_loc:
            b_loc.set(0)
        assemble_vector(self._bres, self._Lres)

        ures = dolfinx.fem.Function(self._V)
        x = dolfinx.la.create_petsc_vector_wrap(ures.x)
        x.pointwiseDivide(self._bres, self._LM._lumpedM)
        ures.x.scatter_forward()

        return ures

    # ----------------------------------------------------------------------
    def compute_difference(self, uh, umax, umin):
        """
        Compute local (per-cell) max and min values of uh.
        Fill into umax and umin functions for slope limiting.
        """
        local_max = uh.x.array.max()
        global_max = self.comm.allreduce(local_max, op=MPI.MAX)
        local_min = uh.x.array.min()
        global_min = self.comm.allreduce(local_min, op=MPI.MIN)

        num_entities = self._V.mesh.topology.index_map(self._V.mesh.topology.dim).size_local
        closure_dofs = self._V.dofmap.cell_dofs(0).size
        
        ii = 0
        for i in range(num_entities):
            loc_max = -1.0e5
            loc_min = 1.0e5
            # Loop over dofs in this cell
            for j in range(closure_dofs):
                loc_max = max(loc_max, uh.x.array[self._DM[ii]])
                loc_min = min(loc_min, uh.x.array[self._DM[ii]])
                ii += 1
            
            # Reset index for cell update
            ii -= closure_dofs
            for j in range(closure_dofs):
                umax.x.array[self._DM[ii]] = loc_max
                umin.x.array[self._DM[ii]] = loc_min
                ii += 1

        umax.x.scatter_forward()
        umin.x.scatter_forward()
        return global_max, global_min

    # ----------------------------------------------------------------------
    def compute_viscosity(self, uh, mu, flux_prime):
        """
        Compute cell-based artificial viscosity mu using residual and
        difference-based limiter.
        """
        fp = flux_prime
        N = len(fp)

        # Max/min functions
        umax = dolfinx.fem.Function(self._V)
        umin = dolfinx.fem.Function(self._V)
       
        global_max, global_min = self.compute_difference(uh, umax, umin)

        # Norm of deviation from average
        S = self.Linf_u(uh)

        # Residual function
        ures = self.compute_residual()

        num_entities = self._V.mesh.topology.index_map(self._V.mesh.topology.dim).size_local
        closure_dofs = self._V.dofmap.cell_dofs(0).size

        ii = 0
        for i in range(num_entities):
            beta = -1.0e5
            res = -1.0e5
            # Compute characteristic speed beta and residual per cell
            for j in range(closure_dofs):
                beta = max(
                    beta,
                    np.sqrt(sum(fp[k].x.array[self._DM[ii]] ** 2 for k in range(N)))
                )
                res = max(res, abs(ures.x.array[self._DM[ii]]))
                ii += 1
            
            # Reset index
            ii -= closure_dofs
            for j in range(closure_dofs):
                S_loc = S * (1 - 0.5 * (umax.x.array[self._DM[ii]] - umin.x.array[self._DM[ii]]) / (global_max - global_min))
                mu.x.array[self._DM[ii]] = min(
                    0.5 * self._h.x.array[self._DM[ii]] * beta,
                    self._h.x.array[self._DM[ii]] ** 2 * res / S_loc
                )
    
                ii += 1
             
        mu.x.scatter_forward()



