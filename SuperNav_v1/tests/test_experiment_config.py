from __future__ import annotations

import json
from pathlib import Path

import pytest

from supernav.experiments.config import (
    ExperimentConfigError,
    list_experiments,
    load_experiment_config,
    resolve_experiment,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, data: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_composition_order_and_replacement_semantics(tmp_path: Path) -> None:
    _write(tmp_path / "base.json", {
        "agent": "codex",
        "agents": {"codex": {"timeout_s": 100, "extra_args": ["base"], "provider": {"id": "a"}}},
        "arms": {"child": {"extends": "primitive"}},
        "disabled": {"enabled": True},
    })
    _write(tmp_path / "overlay.json", {
        "agents": {"codex": {"timeout_s": 200, "extra_args": ["overlay"], "provider": "user"}},
        "disabled": None,
    })
    path = _write(tmp_path / "experiment.json", {
        "extends": ["base.json", "overlay.json"],
        "agents": {"codex": {"timeout_s": 300}},
    })

    config = load_experiment_config(path)

    assert config == {
        "agent": "codex",
        "agents": {"codex": {"timeout_s": 300, "extra_args": ["overlay"], "provider": "user"}},
        "arms": {"child": {"extends": "primitive"}},
        "disabled": None,
    }


def test_each_parent_resolves_relative_to_its_declaring_file(tmp_path: Path, monkeypatch) -> None:
    _write(tmp_path / "profiles" / "base.json", {"output_dir": "relative/output"})
    _write(tmp_path / "profiles" / "agent.json", {"extends": "base.json", "agent": "codex"})
    path = _write(tmp_path / "recipes" / "run.json", {"extends": "../profiles/agent.json"})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    assert load_experiment_config(path) == {"output_dir": "relative/output", "agent": "codex"}


def test_unavailable_parent_path_is_not_redirected(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SUPERNAV_ASSET_ROOT", str(tmp_path))
    _write(tmp_path / "configs/benchmarks/main/config.json", {"agent": "codex"})
    path = _write(tmp_path / "recipes/run.json", {"extends": "../bench/config/main/config.json"})
    with pytest.raises(ExperimentConfigError, match="Cannot load experiment config"):
        load_experiment_config(path)


def test_shared_parent_is_not_an_inheritance_cycle(tmp_path: Path) -> None:
    _write(tmp_path / "base.json", {"bridge": {"host": "localhost"}})
    _write(tmp_path / "left.json", {"extends": "base.json", "agent": "codex"})
    _write(tmp_path / "right.json", {"extends": "base.json", "bridge": {"port": 0}})
    path = _write(tmp_path / "run.json", {"extends": ["left.json", "right.json"]})

    assert load_experiment_config(path) == {"agent": "codex", "bridge": {"host": "localhost", "port": 0}}


def test_cycle_error_identifies_the_files(tmp_path: Path) -> None:
    path = _write(tmp_path / "first.json", {"extends": "second.json"})
    _write(tmp_path / "second.json", {"extends": "first.json"})

    with pytest.raises(ExperimentConfigError, match=r"inheritance cycle:.*first.json.*second.json.*first.json"):
        load_experiment_config(path)


@pytest.mark.parametrize("parents", [None, 12, {}, "", "  ", ["base.json", 1]])
def test_invalid_extends_reports_declaring_file(tmp_path: Path, parents: object) -> None:
    path = _write(tmp_path / "invalid.json", {"extends": parents})

    with pytest.raises(ExperimentConfigError, match=r"'extends'.*invalid.json"):
        load_experiment_config(path)


def test_invalid_root_and_missing_parent_have_context(tmp_path: Path) -> None:
    invalid = _write(tmp_path / "invalid.json", [])
    with pytest.raises(ExperimentConfigError, match=r"root must be an object:.*invalid.json"):
        load_experiment_config(invalid)
    missing = _write(tmp_path / "missing.json", {"extends": "absent.json"})
    with pytest.raises(ExperimentConfigError, match=r"Cannot load experiment config.*absent.json"):
        load_experiment_config(missing)


def test_catalog_uses_asset_root_and_rejects_paths(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("SUPERNAV_ASSET_ROOT", str(tmp_path))
    first = _write(tmp_path / "configs" / "experiments" / "alpha.json", {})
    _write(tmp_path / "configs" / "experiments" / "zeta.json", {})
    _write(tmp_path / "configs" / "agents" / "codex.json", {})

    assert list_experiments() == ["alpha", "zeta"]
    assert resolve_experiment("alpha") == first
    with pytest.raises(ExperimentConfigError, match="available: alpha, zeta"):
        resolve_experiment("missing")
    with pytest.raises(ExperimentConfigError, match="use --config"):
        resolve_experiment("../agents/codex")


def test_supported_recipes_declare_their_backend_and_explicit_provider() -> None:
    for name, backend in [("ai2thor-primitive", "ai2thor"), ("habitat-geo-based-executor", "habitat")]:
        config = load_experiment_config(resolve_experiment(name))
        assert config["environment"]["backend"] == backend
        assert config["agents"]["codex"]["provider_mode"] == "user"
        assert "server" not in config.get("mcp", {})
        assert "python" not in config.get("mcp", {})


@pytest.mark.parametrize("field", ["server", "python", "module"])
def test_forbidden_mcp_launch_fields_fail_when_loading_recipe(field, tmp_path):
    path = _write(tmp_path / "recipe.json", {"mcp": {field: "forbidden-launch-field"}})
    with pytest.raises(ValueError, match=rf"mcp\.{field} is not supported"):
        load_experiment_config(path)
