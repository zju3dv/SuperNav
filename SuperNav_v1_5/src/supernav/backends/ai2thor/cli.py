"""Native AI2-THOR engineering entry point. No success scorer is enabled."""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import signal
import sys
import time
import traceback

from supernav.evaluation.demand_driven.dataset import load_episode, materialize, write_json
from supernav.backends.ai2thor.protocol import NativeConfig
from supernav.backends.ai2thor.storage import configure_storage


def serve(session, port, policy_url):
    four_view = len(session.observation_views) == 4
    policy = None
    if policy_url:
        from supernav.methods.localnav.policy_client import LocalNavPolicyClient

        policy = LocalNavPolicyClient(policy_url, timeout_s=120)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, body):
            content = json.dumps(body, allow_nan=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self):
            if self.path == "/healthz":
                self.send(200, dict(ok=True, terminal=session.terminal,
                                    state="infra_failed" if session.infra_failed else "ready",
                                    views=list(session.observation_views),
                                    success_scoring="withheld", baseline_alignment_confirmed=False))
            elif self.path == "/observation":
                self.send(200, session.observe())
            else:
                self.send(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            try:
                length = int(self.headers.get("Content-Length", 0))
                if not 0 < length <= 16384:
                    raise ValueError("Invalid request size")
                args = json.loads(self.rfile.read(length))
                if self.path == "/step":
                    result = session.step(args["action"])
                    if four_view:
                        result = session.observe()
                elif self.path == "/claim":
                    if four_view:
                        result = session.claim_view(args["observation_id"], args["description"], args["pixel"], args["view"])
                    else:
                        result = session.claim(args["observation_id"], args["description"], args["pixel"])
                elif self.path == "/navigate":
                    if policy is None:
                        raise ValueError("No NoMaD endpoint configured")
                    result = session.local_navigate(args["observation_id"], args["pixel"], policy,
                                                    args.get("max_replans", 4),
                                                    **({"view": args["view"]} if four_view else {}))
                elif self.path == "/stop":
                    result = session.finish()
                elif self.path == "/evaluator-close":
                    # Not exposed by MCP: lifecycle cleanup must never synthesize STOP.
                    if args.get("reason") not in ("agent_returned_without_stop", "wall_timeout"):
                        raise ValueError("Invalid evaluator close reason")
                    result = session.finish(args["reason"])
                else:
                    raise ValueError("Unknown operation")
                self.send(200, result)
            except (ValueError, KeyError, TypeError) as exc:
                self.send(400, {"error": str(exc)})
            except Exception as exc:
                session.infra_failed = True
                session._journal("errors.jsonl", dict(error=traceback.format_exc(), endpoint=self.path))
                self.send(500, {"error": type(exc).__name__, "success_scoring": "withheld"})

    server = HTTPServer(("127.0.0.1", port), Handler)
    print(json.dumps(dict(state="serving", port=port, success_scoring="withheld")), flush=True)
    try:
        server.timeout = .5
        while True:
            server.handle_request()
            if four_view:
                session.expired()
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "validate", "smoke", "scene-audit", "serve"))
    parser.add_argument("--storage-root", type=Path, required=True)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--releases", type=Path)
    parser.add_argument("--episode-id")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--port", type=int, default=18861)
    parser.add_argument("--policy-url")
    parser.add_argument("--camera-height", type=float, choices=(1.25,))
    parser.add_argument("--views", choices=("front", "four"), default="front")
    args = parser.parse_args()

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupted)
    configure_storage(args.storage_root, f"{args.command}_{args.port}" if args.command == "serve" else args.command)
    if args.command == "prepare":
        if args.archive is None:
            parser.error("--archive is required for prepare")
        result = materialize(args.archive, args.dataset)
        print(json.dumps({k: v for k, v in result.items() if k != "episodes"}), flush=True)
        return
    if args.command == "validate":
        from supernav.evaluation.demand_driven.dataset import validate_dataset
        print(json.dumps(validate_dataset(args.dataset)), flush=True)
        return
    if args.limit <= 0:
        parser.error("--limit must be positive")
    if args.output is None or args.releases is None:
        parser.error("--output and --releases are required for native execution")
    args.output = args.output.resolve()
    config = NativeConfig(profile="neednav_h125_unscored", camera_height_m=1.25) if args.camera_height else NativeConfig()
    from supernav.backends.ai2thor.native import NativeSession, make_controller
    session_type = NativeSession
    if args.views == "four":
        from supernav.backends.ai2thor.fourview import FourViewSession, configuration
        config, session_type = configuration(), FourViewSession
    audit = json.loads((args.dataset / "audit.json").read_text())
    records = audit["episodes"]
    if args.episode_id:
        records = [row for row in records if row["episode_id"] == args.episode_id]
        if not records:
            parser.error("Unknown episode ID")
    records = records[:1 if args.command == "serve" else args.limit]
    if not records:
        parser.error("No episodes selected")
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "protocol.json", config.receipt())
    write_json(args.output / "dataset_provenance.json", {
        key: audit.get(key) for key in ("archive_sha256", "episode_count", "scene_count",
            "model_conditioned_selection", "replacement_count", "evaluation_status")
    })
    controller, summary = None, []
    for row in records:
        session = None
        began = time.monotonic()
        identity = row["episode_id"]
        try:
            write_json(args.output / "status.json", dict(state="loading", episode_id=identity,
                       completed=len(summary), total=len(records), success_scoring="withheld"))
            episode, house = load_episode(args.dataset, identity)
            print(json.dumps(dict(state="loading_house", episode_id=identity)), flush=True)
            if controller is None:
                controller = make_controller(house, config, args.output / "unity_runtime", args.releases, args.gpu)
            else:
                event = controller.reset(scene=house)
                if not event.metadata["lastActionSuccess"]:
                    raise RuntimeError("CreateHouse failed")
            session = session_type(controller, episode, config, args.output / identity, house=house)
            session.initialize()
            if args.views == "four":
                session.begin()
            print(json.dumps(dict(state="scene_verified", episode_id=identity)), flush=True)
            write_json(args.output / "status.json", dict(state="serving" if args.command == "serve" else "checking_actions",
                       episode_id=identity, completed=len(summary), total=len(records), success_scoring="withheld"))
            if args.command == "serve":
                serve(session, args.port, args.policy_url)
            else:
                # Fixed, score-independent plumbing sequence; not a policy rollout.
                initial = controller.last_event.metadata["agent"]
                initial = json.loads(json.dumps(initial))
                statuses = []
                for action in ("left", "right", "forward", "backward"):
                    statuses.append(session.step(action))
                    session.observe()
                if not all(status["action_success"] for status in statuses[:2]):
                    raise RuntimeError("Rotation primitive failed")
                if args.policy_url and args.command == "smoke":
                    from supernav.methods.localnav.policy_client import LocalNavPolicyClient
                    policy = LocalNavPolicyClient(args.policy_url, timeout_s=120)
                    health = policy.healthz()
                    if health.get("backend") != "nomad":
                        raise RuntimeError("Only the real NoMaD backend is accepted for a policy smoke")
                    write_json(session.private / "localnav_identity.json", health)
                    result = session.local_navigate(session.observation_id, [.5, .75], policy, max_replans=2)
                    write_json(session.private / "localnav_result.json", {k:v for k,v in result.items() if k not in ("rgb_jpeg_base64", "images")})
                write_json(session.private / "movement_receipt.json", dict(initial_agent=initial,
                           final_agent=controller.last_event.metadata["agent"], primitive_statuses=statuses))
            if not session.terminal:
                session.finish("engineering_probe_complete")
            summary.append(dict(episode_id=identity, valid=True, elapsed_s=time.monotonic() - began))
        except KeyboardInterrupt:
            if session:
                if session.terminal:
                    summary.append(dict(episode_id=identity, valid=not session.infra_failed,
                                        elapsed_s=time.monotonic() - began))
                else:
                    session.finish("operator_interrupt")
            break
        except Exception as exc:
            detail = traceback.format_exc()
            print(detail, file=sys.stderr, flush=True)
            write_json(args.output / (identity + ".infra_error.json"), dict(traceback=detail, success_scoring="withheld"))
            summary.append(dict(episode_id=identity, valid=False, error=f"{type(exc).__name__}: {exc}", elapsed_s=time.monotonic() - began))
            if session:
                session.infra_failed = True
                if not session.terminal:
                    session.finish("infrastructure_error")
            if controller:
                controller.stop()
                controller = None
            if args.command != "scene-audit":
                break
        finally:
            write_json(args.output / "summary.json", dict(records=summary, success_scoring="withheld",
                       valid=sum(r["valid"] for r in summary), invalid=sum(not r["valid"] for r in summary),
                       total=len(records), baseline_alignment_confirmed=False))
    if controller:
        controller.stop()
    state = ("infra_failed" if any(not r["valid"] for r in summary)
             else "interrupted" if len(summary) != len(records) else "complete")
    write_json(args.output / "status.json", dict(state=state,
               completed=len(summary), total=len(records), valid=sum(r["valid"] for r in summary),
               invalid=sum(not r["valid"] for r in summary), success_scoring="withheld"))
    if any(not row["valid"] for row in summary):
        raise SystemExit(1)
    if len(summary) != len(records):
        raise SystemExit(130)


if __name__ == "__main__":
    main()
