"""AMPPSO-specific subpopulation clustering."""

from __future__ import annotations

from dataclasses import dataclass

from utility.clustering import GreedyHierarchicalClusterer


@dataclass(slots=True)
class AMPPSOClusterer(GreedyHierarchicalClusterer):
    """MATLAB AMPPSO complete-style clustering with singletons as remain."""

    min_group_size: int = 2
