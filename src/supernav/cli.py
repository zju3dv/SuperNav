"""The public command tree; implementations are imported only when selected."""

import argparse
from functools import partial
from importlib import import_module
import sys

def _module_main(module: str) -> None:
    raise SystemExit(import_module(module).main())


def run() -> None:
    from supernav.experiments.sweep import main
    raise SystemExit(main())


def run_one() -> None:
    from supernav.experiments.episode import main
    raise SystemExit(main())


def build_prompt() -> None:
    from supernav.experiments.prompt import main
    raise SystemExit(main())


def score_objectnav() -> None:
    from supernav.evaluation.objectnav.score_hm3d import main
    raise SystemExit(main())


def score_multi_objectnav() -> None:
    from supernav.evaluation.objectnav.score_multi import main
    raise SystemExit(main())


def mcp() -> None:
    from supernav.backends import BACKENDS, backend_callable
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--backend", choices=BACKENDS, default="habitat")
    selected, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    backend_callable(selected.backend, "mcp_main")()


def bridge() -> None:
    from supernav.backends.habitat.http_server import main

    raise SystemExit(main())



def web() -> None:
    from supernav.web.server import main

    main()


def config() -> None:
    """Inspect experiment recipes without importing or starting a simulator."""
    import json

    from supernav.experiments.config import (
        list_experiments,
        load_experiment_config,
        resolve_experiment,
    )

    parser = argparse.ArgumentParser(description="Discover and inspect experiment configurations.")
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("list", help="List the named experiments in configs/experiments.")
    show = actions.add_parser("show", help="Print the composed JSON configuration.")
    source = show.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment", help="Name shown by supernav config list.")
    source.add_argument("--config", help="Path to an experiment JSON file.")
    args = parser.parse_args()
    try:
        if args.action == "list":
            for name in list_experiments():
                print(f"{name}\t{resolve_experiment(name)}")
        else:
            path = resolve_experiment(args.experiment) if args.experiment else args.config
            print(json.dumps(load_experiment_config(path), indent=2, ensure_ascii=False))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


def prepare() -> None:
    modules = {
        "global-tasks": "global_tasks",
        "multi-global-tasks": "multi_global_tasks",
        "objectnav": "objectnav_hm3d",
        "ovon": "ovon_mtu3d",
    }
    parser = argparse.ArgumentParser(description="Prepare experiment task manifests and recipes.")
    actions = parser.add_subparsers(dest="dataset", required=True)
    for name in modules:
        actions.add_parser(name, add_help=False)
    args, remaining = parser.parse_known_args()
    sys.argv = [f"supernav prepare {args.dataset}", *remaining]
    _module_main(f"supernav.experiments.preparation.{modules[args.dataset]}")


def main(argv: list[str] | None = None) -> int:
    """Dispatch lazily so help and configuration inspection need no agent SDKs."""
    commands = {
        "run": (run, "Run an experiment sweep."),
        "run-one": (run_one, "Run a single episode."),
        "config": (config, "List experiments or show a composed configuration."),
        "web": (web, "Open the live observation server."),
        "mcp": (mcp, "Start the navigation MCP server."),
        "habitat-bridge": (bridge, "Start the external Habitat bridge."),
        "prompt": (build_prompt, "Build a Habitat navigation prompt."),
        "score-objectnav": (score_objectnav, "Score ObjectNav evidence offline."),
        "score-multi-objectnav": (score_multi_objectnav, "Score multi-goal navigation evidence."),
        "prepare": (prepare, "Prepare task manifests and experiment recipes."),
    }
    for name, module, description in (
        ("viewer", "supernav.web.rerun_viewer", "Replay navigation evidence with Rerun."),
        ("habitat-adapter", "supernav.backends.habitat.stdio_adapter", "Start the Habitat stdio adapter."),
        ("collect-objectnav", "supernav.evaluation.objectnav.collection", "Collect ObjectNav scores."),
        ("aggregate", "supernav.evaluation.aggregate", "Aggregate experiment results."),
        ("video", "supernav.evaluation.video.render", "Render a run video."),
        ("videos", "supernav.evaluation.video.batch", "Render videos for a sweep."),
        ("gallery", "supernav.evaluation.video.gallery", "Render an ObjectNav gallery."),
        ("metrics", "supernav.evaluation.metrics_cli", "Compute run metrics."),
        ("timing", "supernav.runtime.timing_summary", "Summarize execution timing."),
        ("grounding-check", "supernav.evaluation.grounding", "Evaluate visual grounding."),
        ("image-prune-proxy", "supernav.runtime.image_prune_proxy", "Start the image context proxy."),
    ):
        commands[name] = (partial(_module_main, module), description)
    parser = argparse.ArgumentParser(
        prog="supernav",
        description="SuperNav: experiments, agents, navigation and evaluation.",
        epilog="Start with: supernav config list. Use supernav COMMAND --help for command options.",
    )
    subparsers = parser.add_subparsers(dest="command", metavar="COMMAND")
    for name, (_, description) in commands.items():
        # The existing command owns its flags and help. Do not consume them here.
        subparsers.add_parser(name, help=description, add_help=False)
    selected, remaining = parser.parse_known_args(argv)
    if selected.command is None:
        if remaining:
            parser.error(f"unrecognized arguments: {' '.join(remaining)}")
        parser.print_help()
        return 0
    previous_argv = sys.argv
    try:
        sys.argv = [f"supernav {selected.command}", *remaining]
        commands[selected.command][0]()
    finally:
        sys.argv = previous_argv
    return 0
