"""Regression for the snapshot's synchronous session-persistence stop hook."""

from dimos.core.coordination.module_coordinator import ModuleCoordinator
from dimos.core.global_config import GlobalConfig


def test_stop_waits_for_prepare_stop_before_dispatching_stop():
    events = []

    class PreparedModule:
        rpcs = {"prepare_stop": object(), "stop": object()}

    class FakeProxy:
        def prepare_stop(self):
            events.append("prepare_stop")

        def stop(self):
            events.append("stop")

    coordinator = ModuleCoordinator(g=GlobalConfig(n_workers=0, viewer="none"))
    coordinator._managers = {}
    coordinator._deployed_modules[PreparedModule] = FakeProxy()
    coordinator.stop()
    assert events == ["prepare_stop", "stop"]
