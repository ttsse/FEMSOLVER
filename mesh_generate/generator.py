import gmsh
import numpy as np

def generate_mesh(target_dofs, tol=0, max_iter=20):
    gmsh.initialize()
    gmsh.model.add("circle")

    r = 1.0
    circle = gmsh.model.occ.addDisk(0, 0, 0, r, r)
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(2, [circle], tag=1)
    gmsh.model.setPhysicalName(2, 1, "Domain")

    # Define search bounds for the characteristic length
    lc_min = 1e-3
    lc_max = 5.0

    best_lc = None
    best_diff = float('inf')

    for i in range(max_iter):
        lc_mid = (lc_min + lc_max) / 2
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc_mid)

        gmsh.model.mesh.clear()
        gmsh.model.mesh.generate(2)

        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        n_nodes = len(node_tags)

        diff = abs(n_nodes - target_dofs)

        if diff < best_diff:
            best_diff = diff
            best_lc = lc_mid

        if diff <= tol:
            break

        if n_nodes > target_dofs:
            # Mesh too fine, increase lc
            lc_min = lc_mid
        else:
            # Mesh too coarse, decrease lc
            lc_max = lc_mid

    # Generate final mesh with best_lc
    gmsh.model.mesh.clear()
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", best_lc)
    gmsh.model.mesh.generate(2)

    gmsh.write("circle.msh")
    print(f"Generated {len(gmsh.model.mesh.getNodes()[0])} nodes (target: {target_dofs}) with lc={best_lc:.5f}")

    gmsh.finalize()

generate_mesh(target_dofs=100)