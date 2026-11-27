from __future__ import annotations

import json
from pathlib import Path

import pytest

from supernav.runtime.agents import get_agent_backend
from supernav.runtime.providers import kimi_provider_environment, validate_provider_environment
from supernav.runtime.skill_visibility import probe_skill_visibility


@pytest.mark.parametrize("client", ["kimi", "opencode"])
def test_explicit_provider_is_run_local_and_does_not_copy_auth(client, tmp_path, monkeypatch):
    source = tmp_path / "global-home"
    source.mkdir()
    (source / "auth.json").write_text('{"oauth":"must-not-copy"}')
    provider = {"id": "yang", "base_url": "https://example.test/openai", "env_key": "TEST_API_KEY"}
    agent = {"provider": provider, "source_home": str(source)}
    backend = get_agent_backend(client)
    project = backend.prepare_project(
        run_dir=tmp_path / "run", workspace_root=tmp_path,
        config={"mcp": {"name": "world", "command": "mock"}, "agent_instructions": "test"},
        arm_cfg={}, agent_cfg=agent, model_cfg={"name": "gpt-test"},
    )
    monkeypatch.setenv("TEST_API_KEY", "only-in-environment")
    validate_provider_environment(agent)
    assert not list(project.rglob("auth.json"))
    for f in project.rglob("*"):
        if f.is_file():
            assert "only-in-environment" not in f.read_text()
    if client == "opencode":
        config = json.loads((project / "opencode.json").read_text())
        assert config["model"] == "yang/gpt-test"
        assert config["enabled_providers"] == ["yang"]
        assert config["provider"]["yang"]["options"]["apiKey"] == "{env:TEST_API_KEY}"
    else:
        env = kimi_provider_environment(agent, "gpt-test")
        assert env["KIMI_MODEL_API_KEY"] == "only-in-environment"
        assert env["KIMI_MODEL_PROVIDER_TYPE"] == "openai_responses"
    monkeypatch.delenv("TEST_API_KEY")
    with pytest.raises(ValueError, match="refusing auth fallback"):
        validate_provider_environment(agent)


def test_visibility_accepts_verbatim_inline_code_but_rejects_paraphrase(tmp_path, monkeypatch):
    from supernav.runtime import skill_visibility
    expected = tmp_path / ".opencode/skills/demo/SKILL.md"
    description = "Use local_navigate for scene navigation."
    def run(text):
        monkeypatch.setattr(skill_visibility, "_run", lambda *a, **kw: ("completed", 0, text, ""))
        return probe_skill_visibility(backend="opencode", project_dir=tmp_path, skill_name="demo", description=description, agent_cfg={})
    assert run(f"demo Use `local_navigate` for scene navigation. {expected}")["status"] == "visible"
    assert run(f"demo Navigate the scene with local tools. {expected}")["status"] == "not_visible"


def test_external_simulator_import_disables_only_its_editable_rebuild(tmp_path, monkeypatch):
    import sys
    from types import SimpleNamespace
    from supernav.backends.habitat.loader import _redirect_editable_habitat_sources
    package = tmp_path / "habitat_sim"
    package.mkdir()
    (package / "__init__.py").write_text("")
    simulator = SimpleNamespace(known_source_files={"habitat_sim": str(package / "__init__.py")}, rebuild_flag=True)
    unrelated = SimpleNamespace(known_source_files={"other_package": "other.py"}, rebuild_flag=True)
    monkeypatch.setattr(sys, "meta_path", [simulator, unrelated])
    _redirect_editable_habitat_sources(package)
    assert simulator.rebuild_flag is False
    assert unrelated.rebuild_flag is True
