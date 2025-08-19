# PBC.py
# -----------------------------------------------------------------------------
# PeriodicBC for dolfinx 0.7.x (no MPC)
#
# Purpose (for users):
#   Given a function space V and two small callables,
#     - pbc_condition(x): marks the boundary side to be identified (the "slave"),
#     - pbc_relation(X):  maps those slave DOF coordinates to their target
#                         "master" coordinates,
#   this constructs a new function space W where the periodic DOFs share the
#   same global numbering.
#
# Minimal usage:
#   W = PeriodicBC().create_periodic_condition(V, pbc_condition, pbc_relation)
#
# Notes:
#   • Works in serial and MPI (all ranks must call it).
#   • Coordinates are matched by value within tolerance `eps`.
#   • Keep your pbc_condition fast (it is called inside boundary search).
# -----------------------------------------------------------------------------

from __future__ import annotations
import numpy as np
from mpi4py import MPI

from dolfinx import fem as _fem, mesh as _mesh
from dolfinx import cpp as _cpp
from dolfinx.cpp.common import IndexMap as _IndexMap
from dolfinx.cpp.fem import DofMap as _CPPDofMap
from dolfinx.cpp.graph import AdjacencyList_int32 as _AdjacencyList


def _l2g(imap, local_idx: np.ndarray) -> np.ndarray:
    """
    Convert local DOF indices to global DOF indices via IndexMap.

    Parameters
    ----------
    imap : dolfinx.cpp.common.IndexMap
        Index map associated with a DofMap.
    local_idx : array_like
        Local indices (any int type).

    Returns
    -------
    np.ndarray (int64)
        Global indices corresponding to `local_idx`.
    """
    return imap.local_to_global(np.asarray(local_idx, dtype=np.int32))


def _get_cell_adjacency(V: _fem.FunctionSpaceBase) -> tuple[np.ndarray, np.ndarray]:
    """
    Get cell→dof connectivity in the format required by the C++ DofMap constructor.

    Returns
    -------
    flat : np.ndarray (int32)
        Concatenated list of dofs for all cells.
    offs : np.ndarray (int32)
        CSR-style offsets; for cell i, its dofs are flat[offs[i]:offs[i+1]].

    Compatibility
    -------------
    Works whether V.dofmap.list is a C++ AdjacencyList (with .array/.offsets)
    or a 2D NumPy array.
    """
    adj = V.dofmap.list
    if hasattr(adj, "array") and hasattr(adj, "offsets"):
        flat = np.asarray(adj.array, dtype=np.int32)
        offs = np.asarray(adj.offsets, dtype=np.int32)
        return flat, offs
    arr = np.asarray(adj)
    if arr.ndim == 2:
        nrows, ncols = arr.shape
        flat = arr.astype(np.int32, copy=False).ravel()
        offs = (np.arange(nrows + 1, dtype=np.int32) * np.int32(ncols))
        return flat, offs
    raise RuntimeError("Unsupported dofmap.list shape; cannot infer offsets.")


class PeriodicBC:
    """Helper that rebuilds a function space with periodic identification."""

    def __init__(self) -> None:
        pass

    def __del__(self) -> None:
        pass

    def create_periodic_condition(
        self,
        V: _fem.FunctionSpaceBase,
        pbc_condition,
        pbc_relation,
        eps: float = 1.0e-6,
    ) -> _fem.FunctionSpaceBase:
        """
        Create a new FunctionSpace where boundary DOFs selected by `pbc_condition`
        are identified with target DOFs given by `pbc_relation`.

        Parameters
        ----------
        V : fem.FunctionSpaceBase
            Original function space on the mesh.
        pbc_condition : callable
            Boundary predicate used by `mesh.locate_entities_boundary`.
            Signature: mask = pbc_condition(x) where x has shape (gdim, N).
        pbc_relation : callable
            Coordinate map for slave DOFs.
            Signature: Y = pbc_relation(X) with X, Y of shape (3, M).
            (For 1D/2D, put coordinates in the first 1 or 2 rows; others can be zero.)
        eps : float
            Absolute tolerance when matching coordinates.

        Returns
        -------
        fem.FunctionSpaceBase
            New space with periodic DOFs merged into their masters.
        """
        # Mesh / MPI info
        comm = V.mesh.comm
        rank = comm.rank
        size = comm.size
        tdim = V.mesh.topology.dim
        gdim = V.mesh.geometry.dim

        # 1) Find boundary facets and DOFs to be identified (the "slaves")
        facets_puppet = _mesh.locate_entities_boundary(V.mesh, tdim - 1, pbc_condition)
        puppets_all = _fem.locate_dofs_topological(V, tdim - 1, facets_puppet)
        puppets_all = np.asarray(puppets_all, dtype=np.int32)

        owned = V.dofmap.index_map.size_local
        puppets = puppets_all[puppets_all < owned]  # only owned DOFs
        num_puppets = int(puppets.size)
        size_local_new = int(owned - num_puppets)

        # 2) Collect slave coordinates (pad to 3 components for consistency)
        X = V.tabulate_dof_coordinates()     # (n_local(+ghost), kdim) with kdim ∈ {1,2,3}
        X_owned = X[:owned, :] if X.shape[0] >= owned else X
        kdim = X_owned.shape[1]

        coords_puppet_flat = np.zeros(3 * num_puppets, dtype=np.float64)
        for i, d in enumerate(puppets.tolist()):
            row = np.ravel(X_owned[int(d), :])
            m = min(gdim, kdim)
            coords_puppet_flat[3 * i : 3 * i + m] = row[:m]

        # 3) Convert slave local indices to global indices
        puppets_glb_local = _l2g(V.dofmap.index_map, puppets)

        # 4) Share the slave global IDs and coordinates across ranks
        counts = np.array(comm.allgather(num_puppets), dtype=np.int32)
        rdispls = np.zeros_like(counts)
        if size > 1:
            rdispls[1:] = np.cumsum(counts[:-1])
        n_all = int(counts.sum())

        puppets_glb_all = np.empty(n_all, dtype=np.int64)
        comm.Allgatherv([puppets_glb_local, MPI.LONG_LONG],
                        [puppets_glb_all, counts, rdispls, MPI.LONG_LONG])

        counts_xyz = (counts * 3).astype(np.int32)
        rdispls_xyz = np.zeros_like(counts_xyz)
        if size > 1:
            rdispls_xyz[1:] = np.cumsum(counts_xyz[:-1])

        coords_puppet_all_flat = np.empty(int(counts_xyz.sum()), dtype=np.float64)
        comm.Allgatherv([coords_puppet_flat, MPI.DOUBLE],
                        [coords_puppet_all_flat, counts_xyz, rdispls_xyz, MPI.DOUBLE])

        # 5) Map slave coordinates to their master targets
        M = n_all
        coords_puppet_all_3xM = coords_puppet_all_flat.reshape(-1, 3).T
        mapped = np.asarray(pbc_relation(coords_puppet_all_3xM), dtype=np.float64)
        if mapped.shape != (3, M):
            mapped = mapped.reshape(3, M)

        # 6) For each mapped target, try to find the owned master DOF by coordinate match
        masters_local_idx = -np.ones(M, dtype=np.int64)

        for i in range(M):
            mx = mapped[:, i]

            def _mark(x):
                # Narrow the facet search to places where at least one component matches
                m = np.zeros(x.shape[1], dtype=bool)
                m |= np.isclose(x[0], mx[0], atol=eps)
                if gdim > 1:
                    m |= np.isclose(x[1], mx[1], atol=eps)
                if gdim > 2:
                    m |= np.isclose(x[2], mx[2], atol=eps)
                return m

            facets = _mesh.locate_entities_boundary(V.mesh, tdim - 1, _mark)
            if facets.size == 0:
                continue

            cand = _fem.locate_dofs_topological(V, tdim - 1, facets)
            for d in cand:
                if d < owned:
                    row = np.ravel(X_owned[int(d), :])
                    xd = np.zeros(3, dtype=np.float64)
                    m = min(gdim, row.shape[0])
                    xd[:m] = row[:m]
                    if np.allclose(xd, mx, atol=eps):
                        masters_local_idx[i] = int(d)
                        break

        masters_glb_local = np.zeros(M, dtype=np.int64)
        mask = masters_local_idx >= 0
        if mask.any():
            masters_glb_local[mask] = _l2g(V.dofmap.index_map, masters_local_idx[mask])

        masters_glb = np.empty(M, dtype=np.int64)
        comm.Allreduce(masters_glb_local, masters_glb, op=MPI.SUM)

        # 7) Create a new global numbering: remove slaves and alias them to their masters
        size_global = V.dofmap.index_map.size_global
        new_global_index = np.arange(size_global, dtype=np.int64)

        address_local = np.zeros(size_global, dtype=np.int32)
        lr0, lr1 = V.dofmap.index_map.local_range
        address_local[lr0:lr1] = rank
        address = np.empty_like(address_local)
        comm.Allreduce(address_local, address, op=MPI.SUM)

        order = np.argsort(puppets_glb_all)
        puppets_sorted = puppets_glb_all[order]
        masters_sorted = masters_glb[order]

        removed = 0
        k = 0
        for g in range(size_global):
            while k < puppets_sorted.size and puppets_sorted[k] == g:
                removed += 1
                k += 1
            new_global_index[g] -= removed

        for g_p, g_m in zip(puppets_sorted, masters_sorted):
            new_global_index[g_p] = new_global_index[g_m]
            address[g_p] = address[g_m]

        # 8) Build the reduced IndexMap (owners + ghosts) for the new numbering
        n_loc_ghost = V.dofmap.index_map.size_local + V.dofmap.index_map.num_ghosts
        old_locals = np.arange(n_loc_ghost, dtype=np.int32)
        old_globals = _l2g(V.dofmap.index_map, old_locals)

        ghosts_set: set[int] = set()
        ghosts: list[int] = []
        owners: list[int] = []
        for g_old in old_globals:
            g_new = int(new_global_index[g_old])
            owner = int(address[g_old])
            if owner != rank and g_new not in ghosts_set:
                ghosts_set.add(g_new)
                ghosts.append(g_new)
                owners.append(owner)

        indexmap_new = _IndexMap(comm, size_local_new, ghosts, owners)

        # 9) Map "old local" indices to "new local" indices
        n_loc_ghost_new = indexmap_new.size_local + indexmap_new.num_ghosts
        new_locals = np.arange(n_loc_ghost_new, dtype=np.int32)
        new_globals = _l2g(indexmap_new, new_locals)
        g2newlocal = {int(G): int(L) for L, G in enumerate(new_globals)}

        oldLocal2newLocal = np.empty(n_loc_ghost, dtype=np.int32)
        for i, g_old in enumerate(old_globals):
            oldLocal2newLocal[i] = g2newlocal[int(new_global_index[g_old])]

        # 10) Remap the cell→dof connectivity to the new local indices
        flat_old, offs = _get_cell_adjacency(V)
        flat_new = oldLocal2newLocal[flat_old.astype(np.int64)].astype(np.int32, copy=False)
        offs = offs.astype(np.int32, copy=False)
        adj_new = _AdjacencyList(flat_new, offs)

        # 11) Build a new C++ DofMap and wrap into a Python FunctionSpace
        dofmap_new_cpp = _CPPDofMap(
            V.dofmap.dof_layout,
            indexmap_new,
            V.dofmap.index_map_bs,
            adj_new,
            V.dofmap.bs,
        )

        # dolfinx 0.7.x: construct cpp FunctionSpace, then wrap as FunctionSpaceBase
        try:
            cppV = _cpp.fem.FunctionSpace_float64(V.mesh._cpp_object, V.element, dofmap_new_cpp)
        except TypeError:
            cppV = _cpp.fem.FunctionSpace_float32(V.mesh._cpp_object, V.element, dofmap_new_cpp)

        W = _fem.FunctionSpaceBase(V.mesh, V.ufl_element(), cppV)
        return W
