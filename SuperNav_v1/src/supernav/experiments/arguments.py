"""Explicit experiment selection for public commands."""

from __future__ import annotations

import argparse

from supernav.experiments.config import resolve_experiment


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment", help="Named recipe from supernav config list.")
    source.add_argument(
        "--config",
        help="Experiment JSON path.",
    )


def select_config(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    try:
        if args.experiment:
            args.config = str(resolve_experiment(args.experiment))

    except (OSError, ValueError) as exc:
        parser.error(str(exc))
