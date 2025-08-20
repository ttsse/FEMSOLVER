"""
Visualizer.py

Simple wrapper for plotting dolfinx Functions with pyvista.
"""

try:
    import pyvista
    from dolfinx import plot
except ModuleNotFoundError:
    pyvista = None


class Visualizer:
    """
    Wrap pyvista visualization of dolfinx Functions.
    """

    def __init__(self):
        if pyvista is None:
            print("'pyvista' is required to visualise the solution")
            print("Install 'pyvista' with pip: python3 -m pip install pyvista")

        # Force off-screen rendering (useful for saving files on clusters/headless servers)
        if pyvista is not None:
            pyvista.OFF_SCREEN = True

    def plot_function(self, V, uh, filename=None):
        """
        Plot a dolfinx.Function uh defined on function space V.

        Parameters
        ----------
        V : dolfinx.fem.FunctionSpace
            Function space of uh.
        uh : dolfinx.fem.Function
            Solution to plot.
        filename : str or None
            If provided, saves figure as an image file.
            If None, shows an interactive window.
        """
        if pyvista is None:
            return

        # Extract VTK mesh
        cells, types, x = plot.vtk_mesh(V)

        # Create unstructured grid with solution values
        grid = pyvista.UnstructuredGrid(cells, types, x)
        grid.point_data["u"] = uh.x.array.real
        grid.set_active_scalars("u")

        # Warp mesh by solution value
        warped = grid.warp_by_scalar()

        # Setup and render
        plotter = pyvista.Plotter(off_screen=True)
        plotter.add_mesh(warped)

        if filename is not None:
            # Render once before screenshot
            plotter.show(auto_close=False)  
            plotter.screenshot(filename)
            plotter.close()
            print(f"✅ Saved screenshot to {filename}")
        else:
            # Interactive mode
            pyvista.OFF_SCREEN = False
            plotter = pyvista.Plotter()
            plotter.add_mesh(warped)
            plotter.show()
