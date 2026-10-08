"""AI2-THOR bindings for the shared experiment runner (no simulator SDK imports)."""
from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from importlib.resources import files
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.request

from supernav.backends.base import BridgeStartupError, EpisodeTask
from supernav.evaluation.demand_driven.dataset import dataset_path, digest, load_episode
from supernav.runtime.config import instruction_rows, load_json, resolve_path, write_json
from supernav.runtime.streams import write_canonical_events
from supernav.runtime.mcp import validate_mcp_config
from supernav.paths import python_paths


def settings(config, root):
    env = dict(config.get("environment") or {})
    for key, variable in (("dataset", "SUPERNAV_AI2THOR_DATASET"), ("releases", "SUPERNAV_AI2THOR_RELEASES")):
        path = resolve_path(env.get(key) or os.environ.get(variable), base=root)
        if path is None:
            raise ValueError(f"AI2-THOR requires environment.{key} or {variable}")
        env[key] = str(path)
    env["python"] = os.path.expandvars(str(env.get("python") or os.environ.get("SUPERNAV_AI2THOR_PYTHON") or sys.executable))
    env["views"] = str(env.get("views", "four"))
    if env["views"] not in ("front", "four"):
        raise ValueError("environment.views must be front or four")
    return env


class RunningEpisode:
    def __init__(self, port, path):
        self.url = f"http://127.0.0.1:{port}"
        self.path = path
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(self, route, body=None):
        request = urllib.request.Request(self.url+route,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"})
        with self.opener.open(request, timeout=5) as response:
            return json.load(response)

    def finish(self, reason="agent_returned_without_stop"):
        if not self.request("/healthz")["terminal"]:
            self.request("/evaluator-close", {"reason": reason})
        return load_json(self.path/"status.json")


class Ai2ThorBackend:
    name = "ai2thor"
    default_port = 0

    def tasks(self, config, instructions, root):
        env = settings(config, root)
        dataset = Path(env["dataset"])
        rows = []
        for entry in load_json(dataset/"audit.json")["episodes"]:
            path = dataset_path(dataset, entry["episode_path"])
            if digest(path) != entry["episode_sha256"]:
                raise ValueError("Instruction/start/GT fingerprint changed")
            episode = load_json(path)
            rows.append(dict(task_id=entry["episode_id"], slug=entry["episode_id"],
                             text=episode["instruction"], scene=episode["scene_id"]))
        if instructions:
            selected = instruction_rows(instructions)
            original = {r["task_id"]: r for r in rows}
            for row in selected:
                key = row.get("task_id") or row.get("episode_id")
                if key not in original or row["text"] != original[key]["text"]:
                    raise ValueError("AI2-THOR manifest must select original dataset demands by task_id")
            rows = [original[r.get("task_id") or r.get("episode_id")] for r in selected]
        return rows

    def validate_config(self, config, arm):
        validate_mcp_config(config.get("mcp") or {})
        for environment in ((config.get("mcp") or {}).get("environment") or {}, arm.get("environment") or {}):
            if "HAB_MCP_GLOBAL_TASK" in environment:
                raise ValueError("HAB_MCP_GLOBAL_TASK is not supported in experiment config")

    def prepare_task(self, config, args, arm, root):
        self.validate_config(config, arm)
        env = settings(config, root)
        episode, _ = load_episode(Path(env["dataset"]), getattr(args, "task_id", None))
        if args.instruction != episode["instruction"]:
            raise ValueError("Instruction differs from the frozen dataset demand")
        if arm.get("grounding_warmup") or any(str(t).startswith("hab_") for t in arm.get("tool_whitelist", [])):
            raise ValueError("This arm uses Habitat tools; select an AI2-THOR-compatible experiment arm")
        if arm.get("tool_whitelist"):
            raise ValueError("AI2-THOR uses the preserved five-tool protocol; tool_whitelist is unsupported")
        if arm.get("skill_file") or arm.get("skill_mode") == "prompt":
            raise ValueError("AI2-THOR skills must use skill_mode=native or explicit agent_instructions")
        movement = arm.get("movement", "primitive")
        if movement not in ("primitive", "nomad"):
            raise ValueError("AI2-THOR movement must be primitive or nomad")
        if movement == "nomad" and not env.get("policy_url"):
            raise ValueError("The nomad arm requires environment.policy_url")
        config["environment"] = env
        config["benchmark_profile"] = "demand_driven"
        return EpisodeTask(episode["episode_id"], episode["instruction"], episode["scene_id"])

    def configure_agent(self, config, arm, task, root, run_dir):
        self.validate_config(config, arm)
        env, bridge = config["environment"], config["bridge"]
        if bridge.get("host", "127.0.0.1") != "127.0.0.1":
            raise ValueError("AI2-THOR experiments own a loopback bridge")
        if not bridge["port"]:
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                bridge["port"] = sock.getsockname()[1]
        # A new simulator directory also makes --overwrite safe for immutable native receipts.
        attempt = 1
        while (run_dir/f"simulator-{attempt}").exists():
            attempt += 1
        config["_simulator_output"] = str(run_dir/f"simulator-{attempt}")
        config["_policy_url"] = env.get("policy_url") if arm.get("movement") == "nomad" else None
        transport_env = {"PYTHONDONTWRITEBYTECODE": "1"}
        if python_paths():
            transport_env["PYTHONPATH"] = os.pathsep.join(python_paths())
        mcp = config.get("mcp") or {}
        for extra in (mcp.get("environment") or {}, arm.get("environment") or {}):
            transport_env.update({str(k): str(v) for k, v in extra.items()})
        server_args = ["-m", "supernav", "mcp", "--backend", "ai2thor", "--storage-root", str(run_dir),
                       "--bridge-port", str(bridge["port"]), "--views", env["views"]]
        transport = str(mcp.get("transport", "stdio"))
        args = mcp.get("args", [*server_args, "--transport", transport])
        http_args = mcp.get("http_args")
        if http_args is None and "args" not in mcp:
            http_args = [*server_args, "--transport", "streamable-http"]
        config["_mcp_launch"] = dict(
            name=str(mcp.get("name", "navigation")),
            command=str(mcp.get("command") or sys.executable),
            args=list(args), environment=transport_env, transport=transport,
            http_args=http_args,
        )
        if arm.get("movement", "primitive") == "nomad":
            if env["views"] == "front" and env.get("camera_height") != 1.25:
                raise ValueError("The original front NoMaD prompt requires environment.camera_height=1.25")
            name = "fourview_instructions.md" if env["views"] == "four" else "agent_instructions.md"
        else:
            name = "primitive_instructions.md"
        config.setdefault("agent_instructions", files("supernav.methods.demand_driven").joinpath(name).read_text())
        config.setdefault("agent_policy", "")
        write_json(run_dir/"backend.json", dict(backend=self.name, environment=env, bridge=bridge,
                   simulator_output=config["_simulator_output"], mcp=config["_mcp_launch"], success_scoring="withheld"))

    def prompt(self, *, task, arm_name, arm_cfg, workspace_root, prompts_dir):
        # Only the original demand crosses this boundary; evaluator metadata stays private.
        return task.instruction

    @contextmanager
    def episode(self, *, root, config, task, run_dir, dry_run, live_context):
        if dry_run:
            yield None
            return
        env = config["environment"]
        port = config["bridge"]["port"]
        with socket.socket() as probe:
            if probe.connect_ex(("127.0.0.1", port)) == 0:
                raise BridgeStartupError(f"Bridge port {port} is already occupied")
        output = Path(config["_simulator_output"])
        command = [env["python"], "-m", "supernav.backends.ai2thor.cli", "serve", "--storage-root", str(run_dir),
                   "--dataset", env["dataset"], "--releases", env["releases"], "--output", str(output),
                   "--episode-id", task.task_id, "--port", str(port), "--views", env["views"],
                   "--gpu", str(env.get("gpu", 0))]
        if env.get("camera_height") is not None:
            command.extend(["--camera-height", str(env["camera_height"])])
        if config.get("_policy_url"):
            command.extend(["--policy-url", str(config["_policy_url"])])
        process_env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        if python_paths():
            process_env["PYTHONPATH"] = os.pathsep.join(python_paths())
        if env.get("live_dir"):
            process_env["SUPERNAV_LIVE_DIR"] = str(resolve_path(env["live_dir"], base=root))
        process_env["SUPERNAV_LIVE_CONTEXT"] = json.dumps(live_context)
        log_path = run_dir/"simulator.log"
        session = RunningEpisode(port, output/task.task_id)
        with log_path.open("a") as log:
            process = subprocess.Popen(command, cwd=root, env=process_env, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            ready = False
            finish_on_exit = True
            try:
                deadline = time.monotonic()+float(config["bridge"].get("startup_timeout_s", 120))
                while True:
                    if process.poll() is not None:
                        raise BridgeStartupError(f"AI2-THOR exited; see {log_path}")
                    try:
                        health = session.request("/healthz")
                        if health.get("state") == "ready":
                            ready = True
                            break
                    except (OSError, ValueError):
                        pass
                    if time.monotonic() >= deadline:
                        raise BridgeStartupError(f"AI2-THOR startup timed out; see {log_path}")
                    time.sleep(.2)
                try:
                    yield session
                except subprocess.TimeoutExpired:
                    session.finish("wall_timeout")
                    raise
                except BaseException:
                    # SIGTERM closes the simulator as operator_interrupt, never synthetic STOP.
                    finish_on_exit = False
                    raise
            finally:
                if ready and finish_on_exit and process.poll() is None:
                    try:
                        session.finish()
                    except (OSError, ValueError):
                        pass
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                write_json(run_dir/"simulator-process.json", dict(command=command, returncode=process.returncode,
                                                               ready=ready, owned=True))

    def collect(self, *, config, root, run_dir, task, args, result, events, session):
        status = session.finish("wall_timeout") if getattr(result, "timed_out", False) else session.finish()
        if status["state"] == "infra_failed":
            raise RuntimeError(f"Simulator infrastructure failed; see {session.path}")
        write_canonical_events(run_dir/"canonical.jsonl", events)
        counts = Counter(e.get("name", "unknown") for e in events if e.get("type") == "tool_call")
        claims = session.path/"evaluator/claims.jsonl"
        return dict(arm=args.arm, slug=args.slug, rep=args.rep, success=None, spl=None,
                    success_scoring="withheld", success_semantics="not_evaluated", agent_terminal_claim=None,
                    session_id=None, simulator_status=status, simulator_evidence=str(session.path),
                    action_count=status["action_count"], stop_called=status["stop_called"],
                    resource_claim_count=len(claims.read_text().splitlines()) if claims.exists() else 0,
                    tool_calls=dict(counts), tool_calls_total=sum(counts.values()),
                    observation_views=["front", "right", "back", "left"] if config["environment"]["views"] == "four" else ["front"])

    def replay(self, run_dir, config, root):
        return {"replay_manifest_status": "native_evidence"}
