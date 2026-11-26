"""AMPPSO_GI-specific subpopulation clustering."""

from __future__ import annotations

from dataclasses import dataclass

from utility.clustering import GreedyHierarchicalClusterer


@dataclass(slots=True)
class AMPPSOGIClusterer(GreedyHierarchicalClusterer):
    """MATLAB AMPPSO_GI complete-style clustering with singletons as remain."""

    min_group_size: int = 2
