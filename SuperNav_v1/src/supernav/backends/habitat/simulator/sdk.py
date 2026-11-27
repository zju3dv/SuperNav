"""Load the optional public Habitat SDK only when a simulator operation needs it.

Importing SuperNav and its owned adapter does not require Habitat-Sim. The bridge
loader selects an installed SDK or an explicitly configured public checkout
before the first simulator operation.
"""
from importlib import import_module


def load():
    from supernav.backends.habitat.loader import prepare_simulator_import

    prepare_simulator_import()
    return import_module("habitat_sim")


def __getattr__(name):
    return getattr(load(), name)
