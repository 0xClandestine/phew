"""Graph-level rewrite rules operating directly on phew.ir.Graph objects.

These run before e-graph encoding for:
  - primitive_subst: pattern-match and replace subgraphs with fast.* nodes
  - tensorops: detect M5 TensorOps opportunities
  - compile_boundaries: insert Compile/AsyncEval nodes

All other rules (algebraic, fusion, layout, precision, quantization) operate
inside the egglog e-graph via phew/egraph/rules_egglog.py.
"""

from .algebraic import AlgebraicPass
from .compile_boundaries import CompileBoundaryPass
from .fusion import ElementwiseFusionPass
from .graph_passes import run_all_passes
from .primitive_subst import PrimitiveSubstPass
from .tensorops import TensorOpsPass

__all__ = [
    "AlgebraicPass",
    "PrimitiveSubstPass",
    "CompileBoundaryPass",
    "ElementwiseFusionPass",
    "TensorOpsPass",
    "run_all_passes",
]
