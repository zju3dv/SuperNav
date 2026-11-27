"""Lazy selection of the built-in simulator implementations and MCP routes."""
from importlib import import_module

BACKENDS = {
    "habitat": {
        "implementation": "supernav.backends.habitat.experiment:HabitatBackend",
        "mcp_main": "supernav.methods.navigation.mcp_server:main",
        "mcp_spec": "supernav.backends.habitat.config:build_mcp_spec",
    },
    "ai2thor": {
        "implementation": "supernav.backends.ai2thor.experiment:Ai2ThorBackend",
        "mcp_main": "supernav.methods.demand_driven.mcp:main",
        "mcp_spec": None,
    },
}


def backend_callable(name, member):
    try:
        backend = BACKENDS[name]
    except KeyError:
        raise ValueError(f"Unknown environment backend {name!r}; expected {tuple(BACKENDS)}") from None
    target = backend[member]
    if target is None:
        return None
    module, attribute = target.split(":", 1)
    return getattr(import_module(module), attribute)


def get_backend(config):
    name = str(config.get("environment", {}).get("backend", "habitat"))
    return backend_callable(name, "implementation")()
