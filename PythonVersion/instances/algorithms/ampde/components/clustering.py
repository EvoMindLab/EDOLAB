"""AMPDE-specific subpopulation clustering."""

from __future__ import annotations

from dataclasses import dataclass

from utility.clustering import GreedyHierarchicalClusterer


@dataclass(slots=True)
class AMPDEClusterer(GreedyHierarchicalClusterer):
    """MATLAB AMPDE complete-style clustering with small groups as remain."""

    min_group_size: int = 5
