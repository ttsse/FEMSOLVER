from petsc4py import PETSc
from mpi4py import MPI
import dolfinx


class LumpedMass:

    def __init__(self, M, correction):
        self._M = M
        self._c = correction
        self._lumpedM = None
        self._I = None
        self._A = None
        self._lumpedMatI = None

        self.compute_lumpedmass()

        if self._c == -1:
            comm = MPI.COMM_WORLD
            self._solver = PETSc.KSP().create(comm)
            self._solver.setOperators(self._M)
        elif self._c >= 1:
            self.compute_I()
            self.compute_A()

    # ------------------------------------------------------------------
    def compute_lumpedmass(self):
        """Compute lumped mass vector (row sum of M)."""
        self._lumpedM = self._M.getRowSum()
        self._lumpedM.assemblyBegin()
        self._lumpedM.assemblyEnd()

    # ------------------------------------------------------------------
    def compute_I(self):
        """Build identity matrix with same layout as M."""
        m, n = self._M.getSize()
        if m != n:
            raise ValueError("Matrix must be square for identity.")

        self._I = PETSc.Mat().createAIJ(
            size=self._M.getSizes(),
            comm=self._M.comm
        )
        self._I.setPreallocationNNZ(1)
        self._I.setUp()

        d = self._M.createVecLeft()
        d.set(1.0)
        self._I.setDiagonal(d)
        self._I.assemble()

    # ------------------------------------------------------------------
    def compute_A(self):
        """Compute A = I - Minv*M, where Minv = diag(1/rowSums(M))."""
        m, n = self._M.getSize()
        if m != n:
            raise ValueError("Matrix must be square for A.")

        # Minv vector
        Minv = self._M.createVecLeft()
        self._lumpedM.copy(result=Minv)
        Minv.reciprocal()

        # Diagonal inverse mass matrix
        self._lumpedMatI = PETSc.Mat().createAIJ(
            size=self._M.getSizes(),
            comm=self._M.comm
        )
        self._lumpedMatI.setPreallocationNNZ(1)
        self._lumpedMatI.setUp()
        self._lumpedMatI.setDiagonal(Minv)
        self._lumpedMatI.assemble()

        # MM = Minv * M
        MM = self._lumpedMatI.matMult(self._M)

        # A = I - MM   (now same distribution as M)
        self._A = self._I.copy()
        self._A.axpy(-1.0, MM)

    # ------------------------------------------------------------------
    def get_dk(self, b, V):
        if self._c == -1:
            return self.solve_consistentmass(b, V)
        elif self._c == 0:
            return self.solve_0correction(b, V)
        elif self._c == 1:
            return self.solve_1correction(b, V)
        elif self._c == 2:
            return self.solve_2correction(b, V)
        elif self._c == 3:
            return self.solve_3correction(b, V)
        elif self._c == 4:
            return self.solve_4correction(b, V)

    # ------------------------------------------------------------------
    def solve_consistentmass(self, rhs, V):
        udt = dolfinx.fem.Function(V)
        x = dolfinx.la.create_petsc_vector_wrap(udt.x)
        self._solver.solve(rhs, x)
        udt.x.scatter_forward()
        return udt

    def solve_0correction(self, rhs, V):
        udt = dolfinx.fem.Function(V)
        x = dolfinx.la.create_petsc_vector_wrap(udt.x)
        x.pointwiseDivide(rhs, self._lumpedM)
        udt.x.scatter_forward()
        return udt

    # --- Higher-order correction solvers ------------------------------
    def _apply_correction(self, rhs, V, order):
        # Build polynomial I + A + A^2 + ... + A^order
        M = self._I.copy()
        Ak = self._A.copy()
        for k in range(1, order + 1):
            M.axpy(1.0, Ak)
            if k < order:
                Ak = Ak.matMult(self._A)

        M = M.matMult(self._lumpedMatI)

        udt = dolfinx.fem.Function(V)
        x = dolfinx.la.create_petsc_vector_wrap(udt.x)
        M.mult(rhs, x)
        udt.x.scatter_forward()
        return udt

    def solve_1correction(self, rhs, V):
        return self._apply_correction(rhs, V, 1)

    def solve_2correction(self, rhs, V):
        return self._apply_correction(rhs, V, 2)

    def solve_3correction(self, rhs, V):
        return self._apply_correction(rhs, V, 3)

    def solve_4correction(self, rhs, V):
        return self._apply_correction(rhs, V, 4)
