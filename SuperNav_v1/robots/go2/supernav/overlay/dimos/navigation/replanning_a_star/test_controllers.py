import pytest

from dimos.core.global_config import GlobalConfig
from dimos.navigation.replanning_a_star.controllers import PController


def test_configured_minimum_angular_velocity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DIMOS_ASTAR_MIN_ANGULAR_RPS", raising=False)
    default = PController(GlobalConfig(), speed=0.55, control_frequency=10).rotate(0.4)
    monkeypatch.setenv("DIMOS_ASTAR_MIN_ANGULAR_RPS", "0.5")
    calibrated = PController(GlobalConfig(), speed=0.55, control_frequency=10).rotate(0.4)

    assert default.angular.z == pytest.approx(0.2)
    assert calibrated.angular.z == pytest.approx(0.5)
