import numpy as np
from dolfinx.fem import Function

class RungeKutta:
    """
    Explicit RK / SSPRK schemes for semi-discrete PDEs.

    Expected PDE interface:
      - assemble_vector(): assemble the RHS/residual for the CURRENT state in `u`
      - solve(): return a Function udt with the time derivative at the CURRENT state
    """

    def __init__(self, PDE):
        self._PDE = PDE

    # --------------------------------------------------------------------------
    def RK1(self, u, dt):
        """Forward Euler (RK1)."""
        u0 = Function(u.function_space)
        u0.x.array[:] = u.x.array[:]
        u0.x.scatter_forward()

        self._PDE.assemble_vector()
        k1 = self._PDE.solve()

        u.x.array[:] = u0.x.array[:] + dt * k1.x.array[:]
        u.x.scatter_forward()
        return u

    # --------------------------------------------------------------------------
    def RK2(self, u, dt):
        """Heun's explicit RK2 (order 2)."""
        u0 = Function(u.function_space)
        u0.x.array[:] = u.x.array[:]
        u0.x.scatter_forward()

        # k1 = f(u0)
        self._PDE.assemble_vector()
        k1 = self._PDE.solve()

        # u* = u0 + (dt/2) k1
        u.x.array[:] = u0.x.array[:] + 0.5 * dt * k1.x.array[:]
        u.x.scatter_forward()

        # k2 = f(u*)
        self._PDE.assemble_vector()
        k2 = self._PDE.solve()

        # u^{n+1} = u0 + dt k2
        u.x.array[:] = u0.x.array[:] + dt * k2.x.array[:]
        u.x.scatter_forward()
        return u

    # --------------------------------------------------------------------------
    def RK3(self, u, dt):
        """
        Classical 3rd-order RK (Kutta's RK3).
        Butcher tableau:
          0
          1/2   1/2
          1     -1   2
          ----------------
                1/6  2/3  1/6
        """
        u0 = Function(u.function_space)
        u0.x.array[:] = u.x.array[:]
        u0.x.scatter_forward()

        # k1 = f(u0)
        self._PDE.assemble_vector()
        k1 = self._PDE.solve()

        # k2 = f(u0 + 0.5*dt*k1)
        u.x.array[:] = u0.x.array[:] + 0.5 * dt * k1.x.array[:]
        u.x.scatter_forward()
        self._PDE.assemble_vector()
        k2 = self._PDE.solve()

        # k3 = f(u0 - dt*k1 + 2*dt*k2)
        u.x.array[:] = u0.x.array[:] - dt * k1.x.array[:] + 2.0 * dt * k2.x.array[:]
        u.x.scatter_forward()
        self._PDE.assemble_vector()
        k3 = self._PDE.solve()

        # combine
        u.x.array[:] = (
            u0.x.array[:]
            + dt * ( (1.0/6.0) * k1.x.array[:] + (2.0/3.0) * k2.x.array[:] + (1.0/6.0) * k3.x.array[:] )
        )
        u.x.scatter_forward()
        return u

    # --------------------------------------------------------------------------
    def RK4(self, u, dt):
        """Classical 4th-order RK."""
        u0 = Function(u.function_space)
        u0.x.array[:] = u.x.array[:]
        u0.x.scatter_forward()

        # k1
        self._PDE.assemble_vector()
        k1 = self._PDE.solve()

        # k2
        u.x.array[:] = u0.x.array[:] + 0.5 * dt * k1.x.array[:]
        u.x.scatter_forward()
        self._PDE.assemble_vector()
        k2 = self._PDE.solve()

        # k3
        u.x.array[:] = u0.x.array[:] + 0.5 * dt * k2.x.array[:]
        u.x.scatter_forward()
        self._PDE.assemble_vector()
        k3 = self._PDE.solve()

        # k4
        u.x.array[:] = u0.x.array[:] + dt * k3.x.array[:]
        u.x.scatter_forward()
        self._PDE.assemble_vector()
        k4 = self._PDE.solve()

        # combine
        u.x.array[:] = u0.x.array[:] + dt * (
            (1.0/6.0) * k1.x.array[:] + (1.0/3.0) * k2.x.array[:] +
            (1.0/3.0) * k3.x.array[:] + (1.0/6.0) * k4.x.array[:]
        )
        u.x.scatter_forward()
        return u

