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

    def plot_function(self, V, uh, filename="uh_poisson.png"):
        """
        Plot a dolfinx.Function uh defined on function space V.

        If pyvista.OFF_SCREEN is True, save a screenshot to file.
        Otherwise, show an interactive plot window.
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
        plotter = pyvista.Plotter()
        plotter.add_mesh(warped)

        if pyvista.OFF_SCREEN:
            # Off-screen mode → save screenshot
            plotter.screenshot(filename)
            print(f"Saved screenshot to {filename}")
        else:
            # Interactive mode
            plotter.show()

