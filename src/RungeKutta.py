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

    # --------------------------------------------------------------------------
    def SSPRK22(self, u, dt):
        """SSPRK(2,2) (TVD RK2)."""
        w0 = Function(u.function_space)
        w1 = Function(u.function_space)

        # w0 = u^n
        w0.x.array[:] = u.x.array[:]
        w0.x.scatter_forward()

        # w1 = w0 + dt f(w0)
        self._PDE.assemble_vector()
        udt = self._PDE.solve()
        w1.x.array[:] = w0.x.array[:] + dt * udt.x.array[:]
        w1.x.scatter_forward()

        # u^{n+1} = 0.5*w0 + 0.5*(w1 + dt f(w1))
        u.x.array[:] = w1.x.array[:]
        u.x.scatter_forward()
        self._PDE.assemble_vector()
        udt = self._PDE.solve()
        u.x.array[:] = 0.5 * w0.x.array[:] + 0.5 * (w1.x.array[:] + dt * udt.x.array[:])
        u.x.scatter_forward()
        return u

    # --------------------------------------------------------------------------
    def SSPRK33(self, u, dt):
        """SSPRK(3,3) (Shu–Osher)."""
        w0 = Function(u.function_space)
        w1 = Function(u.function_space)
        w2 = Function(u.function_space)

        # w0 = u^n
        w0.x.array[:] = u.x.array[:]
        w0.x.scatter_forward()

        # w1 = w0 + dt f(w0)
        self._PDE.assemble_vector()
        udt = self._PDE.solve()
        w1.x.array[:] = w0.x.array[:] + dt * udt.x.array[:]
        w1.x.scatter_forward()

        # w2 = 3/4 w0 + 1/4 (w1 + dt f(w1))
        u.x.array[:] = w1.x.array[:]
        u.x.scatter_forward()
        self._PDE.assemble_vector()
        udt = self._PDE.solve()
        w2.x.array[:] = 0.75 * w0.x.array[:] + 0.25 * (w1.x.array[:] + dt * udt.x.array[:])
        w2.x.scatter_forward()

        # u^{n+1} = 1/3 w0 + 2/3 (w2 + dt f(w2))
        u.x.array[:] = w2.x.array[:]
        u.x.scatter_forward()
        self._PDE.assemble_vector()
        udt = self._PDE.solve()
        u.x.array[:] = (1.0/3.0) * w0.x.array[:] + (2.0/3.0) * (w2.x.array[:] + dt * udt.x.array[:])
        u.x.scatter_forward()
        return u

    # --------------------------------------------------------------------------
    def SSPRK43(self, u, dt):
        """
        SSPRK(4,3): 4-stage, 3rd-order Shu–Osher SSP scheme.
        One common form:
          u1 = u0 + 1/2 h f(u0)
          u2 = u1 + 1/2 h f(u1)
          u3 = 2/3 u0 + 1/3 (u2 + h f(u2))
          u4 = u3 + 1/2 h f(u3)
          u^{n+1} = u4
        """
        u0 = Function(u.function_space)
        u0.x.array[:] = u.x.array[:]
        u0.x.scatter_forward()

        # u1 = u0 + 0.5 h f(u0)
        self._PDE.assemble_vector()
        udt = self._PDE.solve()
        u1 = Function(u.function_space)
        u1.x.array[:] = u0.x.array[:] + 0.5 * dt * udt.x.array[:]
        u1.x.scatter_forward()

        # u2 = u1 + 0.5 h f(u1)
        u.x.array[:] = u1.x.array[:]
        u.x.scatter_forward()
        self._PDE.assemble_vector()
        udt = self._PDE.solve()
        u2 = Function(u.function_space)
        u2.x.array[:] = u1.x.array[:] + 0.5 * dt * udt.x.array[:]
        u2.x.scatter_forward()

        # u3 = 2/3 u0 + 1/3 (u2 + h f(u2))
        u.x.array[:] = u2.x.array[:]
        u.x.scatter_forward()
        self._PDE.assemble_vector()
        udt = self._PDE.solve()
        u3 = Function(u.function_space)
        u3.x.array[:] = (2.0/3.0) * u0.x.array[:] + (1.0/3.0) * (u2.x.array[:] + dt * udt.x.array[:])
        u3.x.scatter_forward()

        # u4 = u3 + 0.5 h f(u3)
        u.x.array[:] = u3.x.array[:]
        u.x.scatter_forward()
        self._PDE.assemble_vector()
        udt = self._PDE.solve()
        u4 = Function(u.function_space)
        u4.x.array[:] = u3.x.array[:] + 0.5 * dt * udt.x.array[:]
        u4.x.scatter_forward()

        # final
        u.x.array[:] = u4.x.array[:]
        u.x.scatter_forward()
        return u

    # --------------------------------------------------------------------------
    def SSPRK54(self, u, dt):
        """
        SSPRK(5,4): 5-stage, 4th-order optimal SSP scheme (Spiteri–Ruuth).
        Written in a Shu–Osher / low-storage friendly form.
        """
        u0 = Function(u.function_space)
        u0.x.array[:] = u.x.array[:]
        u0.x.scatter_forward()

        # w1 = u0 + 0.391752226571890 * h f(u0)
        self._PDE.assemble_vector()
        udt = self._PDE.solve()
        w1 = Function(u.function_space)
        w1.x.array[:] = u0.x.array[:] + 0.391752226571890 * dt * udt.x.array[:]
        w1.x.scatter_forward()

        # w2 = 0.444370493651235*u0 + 0.555629506348765*w1 + 0.368410593050371*h f(w1)
        u.x.array[:] = w1.x.array[:]
        u.x.scatter_forward()
        self._PDE.assemble_vector()
        udt = self._PDE.solve()
        w2 = Function(u.function_space)
        w2.x.array[:] = (
            0.444370493651235 * u0.x.array[:] +
            0.555629506348765 * w1.x.array[:] +
            0.368410593050371 * dt * udt.x.array[:]
        )
        w2.x.scatter_forward()

        # w3 = 0.620101851488403*u0 + 0.379898148511597*w2 + 0.251891774271694*h f(w2)
        u.x.array[:] = w2.x.array[:]
        u.x.scatter_forward()
        self._PDE.assemble_vector()
        udt = self._PDE.solve()
        w3 = Function(u.function_space)
        w3.x.array[:] = (
            0.620101851488403 * u0.x.array[:] +
            0.379898148511597 * w2.x.array[:] +
            0.251891774271694 * dt * udt.x.array[:]
        )
        w3.x.scatter_forward()

        # w4 = 0.178079954393132*u0 + 0.821920045606868*w3 + 0.544974750228521*h f(w3)
        u.x.array[:] = w3.x.array[:]
        u.x.scatter_forward()
        self._PDE.assemble_vector()
        udt = self._PDE.solve()
        w4 = Function(u.function_space)
        w4.x.array[:] = (
            0.178079954393132 * u0.x.array[:] +
            0.821920045606868 * w3.x.array[:] +
            0.544974750228521 * dt * udt.x.array[:]
        )
        w4.x.scatter_forward()

        # w5 = 0.517231671970585*w2 + 0.096059710526147*w3 + 0.386708617503269*w4
        #      + 0.063692468666290*h f(w3) + 0.226007483236906*h f(w4)
        # (two f-evaluations at w3 and w4 are part of this canonical form)
        # First term with f(w3)
        u.x.array[:] = w3.x.array[:]
        u.x.scatter_forward()
        self._PDE.assemble_vector()
        udt3 = self._PDE.solve()

        # Second term with f(w4)
        u.x.array[:] = w4.x.array[:]
        u.x.scatter_forward()
        self._PDE.assemble_vector()
        udt4 = self._PDE.solve()

        w5 = Function(u.function_space)
        w5.x.array[:] = (
            0.517231671970585 * w2.x.array[:] +
            0.096059710526147 * w3.x.array[:] +
            0.386708617503269 * w4.x.array[:] +
            0.063692468666290 * dt * udt3.x.array[:] +
            0.226007483236906 * dt * udt4.x.array[:]
        )
        w5.x.scatter_forward()

        u.x.array[:] = w5.x.array[:]
        u.x.scatter_forward()
        return u
