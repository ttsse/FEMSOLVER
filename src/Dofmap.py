import numpy as np

class DofMappings:
    """
    A helper class to extract the mapping from mesh entities (cells, facets, edges)
    to the corresponding degrees of freedom (DoFs) in a given function space V.
    
    This is useful for boundary condition enforcement, local operations, or
    understanding how the global vector/matrix degrees of freedom correspond to
    geometric entities in the mesh.
    """

    def __init__(self):
        # No state is stored in this class — all methods are "query" type.
        pass

    # ----------------------------------------------------------------------
    def get_cell_dofs(self, mesh, V):
        """
        Return the DoFs associated with mesh cells (full-dimensional entities).
        """
        dim = mesh.topology.dim  # Topological dimension of the mesh
        entity_dofs = []
        self.get_dofs(mesh, V, entity_dofs, dim)
        return np.asarray(entity_dofs, dtype=np.int32)

    # ----------------------------------------------------------------------
    def get_facet_dofs(self, mesh, V):
        """
        Return the DoFs associated with facets (co-dimension 1 entities).
        In 2D, facets are edges; in 3D, facets are faces.
        """
        entity_dofs = []
        if mesh.topology.dim >= 2:
            dim = 2  # refers to faces in 3D or cells in 2D
            self.get_dofs(mesh, V, entity_dofs, dim)
        return np.asarray(entity_dofs, dtype=np.int32)

    # ----------------------------------------------------------------------
    def get_edge_dofs(self, mesh, V):
        """
        Return the DoFs associated with edges (1D entities).
        Only makes sense if the mesh dimension is >= 1.
        """
        entity_dofs = []
        if mesh.topology.dim >= 1:
            dim = 1
            self.get_dofs(mesh, V, entity_dofs, dim)
        return np.asarray(entity_dofs, dtype=np.int32)

    # ----------------------------------------------------------------------
    def get_dofs(self, mesh, V, entity_dofs, dim: int):
        """
        Low-level routine to fetch DoFs for entities of a given topological dimension.
        This function:
          - builds connectivity between entities and cells,
          - identifies the local index of the entity in a reference cell,
          - retrieves closure DoFs for that entity, and
          - maps them into global DoF indices.
        """
        topo = mesh.topology

        # Ensure connectivity (entity -> cell and cell -> entity) is built
        topo.create_connectivity(dim, topo.dim)
        topo.create_connectivity(topo.dim, dim)

        # Connectivity arrays: which cells a given entity belongs to, and vice versa
        dim_to_cell = topo.connectivity(dim, topo.dim)
        cell_to_dim = topo.connectivity(topo.dim, dim)

        # Block size (for vector-valued elements, bs > 1)
        bs = V.dofmap.bs

        # Layout of DoFs per topological entity for the element
        dl = V.dofmap.dof_layout         

        # Number of entities of given dimension owned locally by this rank
        num_entities = topo.index_map(dim).size_local

        # Loop over all entities (cells/edges/facets) owned locally
        for i in range(num_entities):
            # Find cells that this entity is attached to
            cells = dim_to_cell.links(i)
            if cells.size == 0:
                continue  # isolated entity (can happen in parallel halo)
            c0 = int(cells[0])  # take one attached cell as reference

            # Get all entities of type `dim` attached to that cell
            entities = cell_to_dim.links(c0)

            # Find the local index of this entity within that reference cell
            local_index = None
            for j in range(entities.size):
                if int(entities[j]) == int(i):
                    local_index = int(j)
                    break
            if local_index is None:
                continue  # safety check: shouldn't normally happen

            # Get the closure DoFs for this entity (e.g. DoFs on entity and sub-entities)
            closure_dofs = dl.entity_closure_dofs(dim, local_index)

            # Global DoFs for that cell
            cdofs = V.dofmap.cell_dofs(c0)

            # Map closure DoFs to global indices
            dofs = [int(cdofs[j]) for j in closure_dofs]

            # Handle block structure (vector-valued function spaces):
            # replicate each scalar DoF for each component
            for d in dofs:
                for k in range(bs):
                    entity_dofs.append(d * bs + k)
