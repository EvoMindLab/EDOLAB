"""AMPPSO_BC-specific subpopulation clustering."""

from __future__ import annotations

from dataclasses import dataclass

from utility.clustering import GreedyHierarchicalClusterer


@dataclass(slots=True)
class AMPPSOBCClusterer(GreedyHierarchicalClusterer):
    """MATLAB AMPPSO_BC complete-style clustering with singletons as remain."""

    min_group_size: int = 2
