"""SuperNav's Habitat bridge, composed entirely from application-owned code.

Only the simulator operations import the public ``habitat_sim`` SDK. Request
dispatch, navigation methods, observation evidence, memory remain
owned by SuperNav; the external simulator never imports application code.
"""
from supernav.backends.habitat.loader import prepare_contract_import

prepare_contract_import()

from supernav.backends.habitat.live import SuperNavLiveMixin
from supernav.backends.habitat.observations import SuperNavObservationMixin
from supernav.backends.habitat.simulator.api import HabitatAdapterApiMixin
from supernav.backends.habitat.simulator.core import HabitatAdapterCoreMixin
from supernav.backends.habitat.simulator.motion import HabitatAdapterNavigationMixin
from supernav.backends.habitat.simulator.navigation_access import NavigationAccessMixin
from supernav.backends.habitat.simulator.session_scene import HabitatAdapterSessionSceneMixin
from supernav.backends.habitat.simulator.types import HabitatAdapterError, SUPPORTED_ACTIONS
from supernav.backends.habitat.simulator.visual_media import HabitatAdapterVisualMediaMixin
from supernav.backends.habitat.simulator.payload import SimulatorPayloadMixin


class HabitatAdapter(
    NavigationAccessMixin,
    SuperNavLiveMixin,
    SuperNavObservationMixin,
    HabitatAdapterCoreMixin,
    HabitatAdapterApiMixin,
    HabitatAdapterSessionSceneMixin,
    HabitatAdapterNavigationMixin,
    HabitatAdapterVisualMediaMixin,
    SimulatorPayloadMixin,
):
    """Application-owned sessions and protocol over the public simulator SDK."""

    SUPPORTED_ACTIONS = SUPPORTED_ACTIONS


__all__ = ["HabitatAdapter", "HabitatAdapterError"]
