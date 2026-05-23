"""
RSRA-4B Core Architecture
==========================

Core differentiable components of the Recursive Self-Reflective Architecture:

- :class:`ContinuousChecker` — Latent state verification network
- :class:`StateGenerator` — Hidden state generation (attention + FFN)
- :class:`RefinementOperator` — Contraction-constrained recursive refinement
- :class:`HierarchicalRouter` — 4-tier abstraction routing logic
- :class:`JointLoss` — Tri-objective joint loss function
- :class:`RSRABlock` — Full RSRA block composing all components
"""

from rsra.core.checker import ContinuousChecker
from rsra.core.generator import StateGenerator
from rsra.core.refinement import RefinementOperator
from rsra.core.hierarchy import HierarchicalRouter
from rsra.core.joint_loss import JointLoss
from rsra.core.rsra_block import RSRABlock

__all__ = [
    "ContinuousChecker",
    "StateGenerator",
    "RefinementOperator",
    "HierarchicalRouter",
    "JointLoss",
    "RSRABlock",
]
