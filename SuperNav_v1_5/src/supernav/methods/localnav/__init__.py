"""RGB point-conditioned LocalNav framework.

Public APIs here are model-agnostic.  The first learned implementation is
expected to be an image-guided diffusion policy, while Habitat and robot
execution remain thin adapters around ``VelocityAction``.
"""

from supernav.methods.localnav.adapters import HabitatActionAdapter, HabitatActionAdapterConfig
from supernav.methods.localnav.baselines import DebugPointPolicy, DebugPointPolicyConfig
from supernav.methods.localnav.contracts import (
    ControlDecision,
    GoalSnapshot,
    LocalNavError,
    LocalNavInputError,
    LocalNavPolicy,
    LocalNavRuntimeError,
    NormalizedPoint,
    PolicyInput,
    PolicyPrediction,
    VelocityAction,
)
from supernav.methods.localnav.controller import ControllerConfig, RecedingHorizonController
from supernav.methods.localnav.dataset import LocalNavTrainingSample
from supernav.methods.localnav.goal import (
    as_readonly_rgb,
    build_goal_snapshot,
    render_point_marker,
    snapshot_goal_from_registry,
)

__all__ = [
    "HabitatActionAdapter",
    "HabitatActionAdapterConfig",
    "DebugPointPolicy",
    "DebugPointPolicyConfig",
    "ControlDecision",
    "GoalSnapshot",
    "LocalNavError",
    "LocalNavInputError",
    "LocalNavPolicy",
    "LocalNavRuntimeError",
    "NormalizedPoint",
    "PolicyInput",
    "PolicyPrediction",
    "VelocityAction",
    "ControllerConfig",
    "RecedingHorizonController",
    "LocalNavTrainingSample",
    "as_readonly_rgb",
    "build_goal_snapshot",
    "render_point_marker",
    "snapshot_goal_from_registry",
]
