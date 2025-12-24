"""
Spatial indexing for efficient drone neighbor queries.

Provides O(1) average-case neighbor queries for large swarms,
replacing the naive O(n^2) approach.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set, Any
from collections import defaultdict
from dataclasses import dataclass
import heapq


@dataclass
class SpatialEntity:
    """Entity with spatial position."""
    id: str
    position: Tuple[float, float]
    data: Any = None


class GridSpatialIndex:
    """
    Grid-based spatial index for efficient neighbor queries.

    Divides space into cells and maintains entity lists per cell.
    Queries only check relevant cells, giving O(k) complexity
    where k is the number of entities in nearby cells.
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float],
        cell_size: float = 0.01,
    ):
        """
        Initialize spatial index.

        Args:
            bounds: (min_lat, min_lon, max_lat, max_lon)
            cell_size: Size of grid cells in degrees
        """
        self.min_lat, self.min_lon, self.max_lat, self.max_lon = bounds
        self.cell_size = cell_size

        # Calculate grid dimensions
        self.n_rows = max(1, int((self.max_lat - self.min_lat) / cell_size) + 1)
        self.n_cols = max(1, int((self.max_lon - self.min_lon) / cell_size) + 1)

        # Grid cells: (row, col) -> set of entity IDs
        self.grid: Dict[Tuple[int, int], Set[str]] = defaultdict(set)

        # Entity positions: id -> (lat, lon)
        self.positions: Dict[str, Tuple[float, float]] = {}

        # Entity to cell mapping: id -> (row, col)
        self.entity_cells: Dict[str, Tuple[int, int]] = {}

    def _get_cell(self, lat: float, lon: float) -> Tuple[int, int]:
        """Get grid cell for a position."""
        row = int((lat - self.min_lat) / self.cell_size)
        col = int((lon - self.min_lon) / self.cell_size)

        # Clamp to valid range
        row = max(0, min(row, self.n_rows - 1))
        col = max(0, min(col, self.n_cols - 1))

        return row, col

    def insert(self, entity_id: str, position: Tuple[float, float]):
        """Insert or update entity in index."""
        lat, lon = position

        # Remove from old cell if exists
        if entity_id in self.entity_cells:
            old_cell = self.entity_cells[entity_id]
            self.grid[old_cell].discard(entity_id)

        # Add to new cell
        cell = self._get_cell(lat, lon)
        self.grid[cell].add(entity_id)
        self.positions[entity_id] = position
        self.entity_cells[entity_id] = cell

    def remove(self, entity_id: str):
        """Remove entity from index."""
        if entity_id in self.entity_cells:
            cell = self.entity_cells[entity_id]
            self.grid[cell].discard(entity_id)
            del self.positions[entity_id]
            del self.entity_cells[entity_id]

    def update(self, entity_id: str, new_position: Tuple[float, float]):
        """Update entity position."""
        self.insert(entity_id, new_position)

    def query_radius(
        self,
        position: Tuple[float, float],
        radius: float,
        exclude_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """
        Query entities within radius of position.

        Args:
            position: Query center (lat, lon)
            radius: Search radius in degrees
            exclude_id: Entity ID to exclude from results

        Returns:
            List of (entity_id, distance) tuples, sorted by distance
        """
        lat, lon = position

        # Calculate cell range to check
        cells_to_check = int(radius / self.cell_size) + 1
        center_cell = self._get_cell(lat, lon)

        results = []

        # Check all cells in range
        for dr in range(-cells_to_check, cells_to_check + 1):
            for dc in range(-cells_to_check, cells_to_check + 1):
                cell = (center_cell[0] + dr, center_cell[1] + dc)

                if cell not in self.grid:
                    continue

                for entity_id in self.grid[cell]:
                    if entity_id == exclude_id:
                        continue

                    entity_pos = self.positions[entity_id]
                    dist = self._distance(position, entity_pos)

                    if dist <= radius:
                        results.append((entity_id, dist))

        # Sort by distance
        results.sort(key=lambda x: x[1])
        return results

    def query_k_nearest(
        self,
        position: Tuple[float, float],
        k: int,
        exclude_id: Optional[str] = None,
        max_radius: float = 1.0,
    ) -> List[Tuple[str, float]]:
        """
        Query k nearest entities to position.

        Args:
            position: Query center (lat, lon)
            k: Number of nearest neighbors to find
            exclude_id: Entity ID to exclude
            max_radius: Maximum search radius

        Returns:
            List of (entity_id, distance) tuples
        """
        lat, lon = position

        # Use expanding radius search
        radius = self.cell_size * 2
        results = []

        while len(results) < k and radius <= max_radius:
            results = self.query_radius(position, radius, exclude_id)
            radius *= 2

        return results[:k]

    def query_cell(self, position: Tuple[float, float]) -> Set[str]:
        """Get all entities in the same cell as position."""
        cell = self._get_cell(position[0], position[1])
        return self.grid.get(cell, set()).copy()

    def get_all_pairs_within_radius(
        self,
        radius: float,
    ) -> List[Tuple[str, str, float]]:
        """
        Get all pairs of entities within radius of each other.

        More efficient than querying each entity individually.

        Args:
            radius: Maximum distance between pairs

        Returns:
            List of (id1, id2, distance) tuples
        """
        pairs = []
        cells_to_check = int(radius / self.cell_size) + 1

        # Check each cell
        for cell, entities in self.grid.items():
            if not entities:
                continue

            entity_list = list(entities)

            # Check pairs within same cell
            for i, id1 in enumerate(entity_list):
                pos1 = self.positions[id1]

                for id2 in entity_list[i + 1:]:
                    pos2 = self.positions[id2]
                    dist = self._distance(pos1, pos2)

                    if dist <= radius:
                        pairs.append((id1, id2, dist))

            # Check neighboring cells
            for dr in range(-cells_to_check, cells_to_check + 1):
                for dc in range(-cells_to_check, cells_to_check + 1):
                    if dr == 0 and dc == 0:
                        continue

                    # Only check cells "ahead" to avoid duplicates
                    if dr < 0 or (dr == 0 and dc < 0):
                        continue

                    neighbor_cell = (cell[0] + dr, cell[1] + dc)

                    if neighbor_cell not in self.grid:
                        continue

                    for id1 in entity_list:
                        pos1 = self.positions[id1]

                        for id2 in self.grid[neighbor_cell]:
                            pos2 = self.positions[id2]
                            dist = self._distance(pos1, pos2)

                            if dist <= radius:
                                pairs.append((id1, id2, dist))

        return pairs

    def build_communication_graph(
        self,
        communication_range: float,
    ) -> Dict[str, List[str]]:
        """
        Build communication graph for all entities.

        Args:
            communication_range: Maximum communication distance

        Returns:
            Dictionary mapping entity_id -> list of connected entity_ids
        """
        graph = defaultdict(list)

        pairs = self.get_all_pairs_within_radius(communication_range)

        for id1, id2, dist in pairs:
            graph[id1].append(id2)
            graph[id2].append(id1)

        # Ensure all entities are in graph
        for entity_id in self.positions:
            if entity_id not in graph:
                graph[entity_id] = []

        return dict(graph)

    def _distance(
        self,
        pos1: Tuple[float, float],
        pos2: Tuple[float, float],
    ) -> float:
        """Calculate distance between two positions in degrees."""
        return np.sqrt(
            (pos1[0] - pos2[0]) ** 2 +
            (pos1[1] - pos2[1]) ** 2
        )

    def clear(self):
        """Clear all entities from index."""
        self.grid.clear()
        self.positions.clear()
        self.entity_cells.clear()

    def __len__(self) -> int:
        """Number of entities in index."""
        return len(self.positions)


class SpatialIndex:
    """
    KD-Tree based spatial index for more complex queries.

    Uses scipy's KDTree for efficient nearest neighbor queries.
    Rebuilds tree when entities change significantly.
    """

    def __init__(self, rebuild_threshold: float = 0.1):
        """
        Initialize spatial index.

        Args:
            rebuild_threshold: Fraction of changes before rebuilding tree
        """
        self.rebuild_threshold = rebuild_threshold

        self.entities: Dict[str, Tuple[float, float]] = {}
        self.id_to_idx: Dict[str, int] = {}
        self.idx_to_id: Dict[int, str] = {}

        self._tree = None
        self._positions_array = None
        self._changes_since_rebuild = 0

    def insert(self, entity_id: str, position: Tuple[float, float]):
        """Insert or update entity."""
        self.entities[entity_id] = position
        self._changes_since_rebuild += 1
        self._check_rebuild()

    def remove(self, entity_id: str):
        """Remove entity."""
        if entity_id in self.entities:
            del self.entities[entity_id]
            self._changes_since_rebuild += 1
            self._check_rebuild()

    def _check_rebuild(self):
        """Check if tree should be rebuilt."""
        if len(self.entities) == 0:
            return

        change_ratio = self._changes_since_rebuild / len(self.entities)
        if change_ratio >= self.rebuild_threshold or self._tree is None:
            self._rebuild_tree()

    def _rebuild_tree(self):
        """Rebuild KD-Tree from current entities."""
        if len(self.entities) == 0:
            self._tree = None
            return

        try:
            from scipy.spatial import KDTree

            # Build position array and mappings
            self.id_to_idx = {}
            self.idx_to_id = {}

            positions = []
            for idx, (entity_id, pos) in enumerate(self.entities.items()):
                self.id_to_idx[entity_id] = idx
                self.idx_to_id[idx] = entity_id
                positions.append(pos)

            self._positions_array = np.array(positions)
            self._tree = KDTree(self._positions_array)
            self._changes_since_rebuild = 0

        except ImportError:
            # Fall back to no tree
            self._tree = None

    def query_radius(
        self,
        position: Tuple[float, float],
        radius: float,
        exclude_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Query entities within radius."""
        if self._tree is None:
            self._rebuild_tree()

        if self._tree is None:
            # Fallback to brute force
            return self._brute_force_radius(position, radius, exclude_id)

        # Query tree
        indices = self._tree.query_ball_point(position, radius)

        results = []
        for idx in indices:
            entity_id = self.idx_to_id[idx]
            if entity_id == exclude_id:
                continue

            dist = np.sqrt(
                (position[0] - self._positions_array[idx, 0]) ** 2 +
                (position[1] - self._positions_array[idx, 1]) ** 2
            )
            results.append((entity_id, dist))

        results.sort(key=lambda x: x[1])
        return results

    def query_k_nearest(
        self,
        position: Tuple[float, float],
        k: int,
        exclude_id: Optional[str] = None,
    ) -> List[Tuple[str, float]]:
        """Query k nearest entities."""
        if self._tree is None:
            self._rebuild_tree()

        if self._tree is None:
            return self._brute_force_k_nearest(position, k, exclude_id)

        # Query extra to handle exclusion
        query_k = k + 1 if exclude_id else k

        distances, indices = self._tree.query(position, k=min(query_k, len(self.entities)))

        # Handle single result
        if not hasattr(indices, '__iter__'):
            indices = [indices]
            distances = [distances]

        results = []
        for dist, idx in zip(distances, indices):
            entity_id = self.idx_to_id[idx]
            if entity_id == exclude_id:
                continue
            results.append((entity_id, dist))

        return results[:k]

    def _brute_force_radius(
        self,
        position: Tuple[float, float],
        radius: float,
        exclude_id: Optional[str],
    ) -> List[Tuple[str, float]]:
        """Fallback brute force radius query."""
        results = []
        for entity_id, pos in self.entities.items():
            if entity_id == exclude_id:
                continue

            dist = np.sqrt(
                (position[0] - pos[0]) ** 2 +
                (position[1] - pos[1]) ** 2
            )

            if dist <= radius:
                results.append((entity_id, dist))

        results.sort(key=lambda x: x[1])
        return results

    def _brute_force_k_nearest(
        self,
        position: Tuple[float, float],
        k: int,
        exclude_id: Optional[str],
    ) -> List[Tuple[str, float]]:
        """Fallback brute force k-nearest query."""
        results = []
        for entity_id, pos in self.entities.items():
            if entity_id == exclude_id:
                continue

            dist = np.sqrt(
                (position[0] - pos[0]) ** 2 +
                (position[1] - pos[1]) ** 2
            )
            results.append((entity_id, dist))

        results.sort(key=lambda x: x[1])
        return results[:k]

    def __len__(self) -> int:
        return len(self.entities)
