"""Exercise release exclusions using small isolated sources and fake sentinels."""

from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
POLICY = runpy.run_path(str(ROOT / "scripts/export_release_source.py"))
SENTINEL = b"SUPERNAV_PRIVATE_FILE_SENTINEL\n"
PRIVATE_FILES = (
    ".env", ".env.production", "configs/providers/provider.env.local",
    "configs/providers/auth.json", "configs/auth/account.json", "configs/local/machine.json",
    "skills/example/.env", "skills/example/auth.json", "skills/example/credentials.yaml",
    "skills/example/nested/.codex/config.toml", "configs/agents/deep/.kimi/config.toml",
    ".claude/settings.json", ".opencode/state.json", "configs/deep/.opencode/state.json",
    ".claude.json", ".mcp.json", ".netrc", "settings.local.json", "src/example/credentials.yaml",
    "src/example/.codex/deep/settings.yaml", "data/runs/run.json",
    "AGENTS.md", "docs/development/handoff.md", "docs/README.md",
    "configs/benchmarks/example/tasks.json",
    "configs/benchmarks/example/answers.json.gz",
    "configs/tasks/private-index.json",
    "configs/benchmarks/example/targets.json",
    "configs/benchmarks/example/reference.png",
)


def run(source, *args):
    result = subprocess.run(args, cwd=source, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode == 0, result.stdout.decode(errors="replace")
    return result.stdout


@pytest.fixture
def source(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    for name in (".gitignore", ".gitattributes", "MANIFEST.in", "scripts/release_build.py", "scripts/export_release_source.py"):
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, target)
    (source / "pyproject.toml").write_text('''\
[build-system]
requires = ["setuptools>=77", "wheel"]
build-backend = "setuptools.build_meta"
[project]
name = "release-privacy-fixture"
version = "0.0.0"
[tool.setuptools]
include-package-data = false
[tool.setuptools.cmdclass]
egg_info = "scripts.release_build.SafeEggInfo"
build_py = "scripts.release_build.SafeBuildPy"
[tool.setuptools.package-dir]
"" = "src"
scripts = "scripts"
[tool.setuptools.packages.find]
where = ["src"]
namespaces = false
[tool.setuptools.package-data]
example = ["*.yaml", ".codex/**/*.yaml"]
[tool.setuptools.data-files]
"share/examples" = ["configs/providers/safe.env.example"]
"share/supernav" = ["AGENTS.md"]
"share/private" = ["configs/providers/auth.json", "configs/local/linked.md"]
"share/supernav/configs/benchmarks/example" = ["configs/benchmarks/example/tasks.json"]
"share/supernav/configs/benchmarks/images" = ["configs/benchmarks/example/reference.png"]
''')
    for name, content in {
        "src/example/__init__.py": "",
        "src/example/public.yaml": "public: true\n",
        "skills/example/SKILL.md": "Public skill\n",
        "configs/providers/safe.env.example": "# Empty deployment template\nAPI_KEY=\n",
    }.items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    for name in PRIVATE_FILES:
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(SENTINEL)
    # The linked target stays inside this disposable fixture, never in the workspace.
    (source / "configs/local/linked.md").symlink_to(source / "skills/example/auth.json")
    egg_info = source / "src/release_privacy_fixture.egg-info"
    egg_info.mkdir()
    (egg_info / "SOURCES.txt").write_text("\n".join(PRIVATE_FILES) + "\n")
    return source


def test_sdist_and_wheel_exclude_private_files_and_stale_sources(source, tmp_path):
    run(source, sys.executable, "-I", "-B", "-c",
        "from setuptools.build_meta import build_sdist,build_wheel; build_sdist('../sdist'); build_wheel('../direct-wheel')")
    sdist = next((tmp_path / "sdist").glob("*.tar.gz"))
    with tarfile.open(sdist) as archive:
        members = archive.getmembers()
        assert not any(POLICY["is_private_path"](member.name) or member.issym() for member in members)
        assert not any(SENTINEL in archive.extractfile(member).read() for member in members if member.isfile())
        assert any(member.name.endswith("/skills/example/SKILL.md") for member in members)
        archive.extractall(tmp_path / "extracted", filter="data")
    extracted = next((tmp_path / "extracted").iterdir())
    run(extracted, sys.executable, "-I", "-B", "-c", "from setuptools.build_meta import build_wheel; build_wheel('../../wheel')")
    for folder in ("direct-wheel", "wheel"):
        with zipfile.ZipFile(next((tmp_path / folder).glob("*.whl"))) as archive:
            assert not any(POLICY["is_private_path"](name) for name in archive.namelist())
            assert not any(SENTINEL in archive.read(name) for name in archive.namelist())
            assert "example/public.yaml" in archive.namelist()
            templates = [name for name in archive.namelist() if name.endswith("safe.env.example")]
            assert len(templates) == 1
            POLICY["check_empty_env_example"](templates[0], archive.read(templates[0]))


def test_archive_exclusions_and_export_rejection(source, tmp_path):
    run(source, "git", "init", "-q")
    run(source, "git", "add", "-f", ".")
    run(source, "git", "-c", "user.name=Release test", "-c", "user.email=release@example.invalid", "commit", "-qm", "fixture")
    archive_path = tmp_path / "raw-archive.tar"
    run(source, "git", "archive", "HEAD", f"--output={archive_path}")
    with tarfile.open(archive_path) as archive:
        assert not any(POLICY["is_private_path"](member.name) for member in archive.getmembers())
        assert not any(SENTINEL in archive.extractfile(member).read() for member in archive.getmembers() if member.isfile())
        assert "configs/providers/safe.env.example" in archive.getnames()
    with pytest.raises(ValueError, match="Sensitive path is tracked"):
        POLICY["export_source"](source, tmp_path / "rejected.tar.gz")
    run(source, "git", "rm", "-qr", "--cached", ".")
    (source / "configs/local/linked.md").unlink()
    run(source, "git", "add", ".")
    run(source, "git", "-c", "user.name=Release test", "-c", "user.email=release@example.invalid", "commit", "-qm", "clean fixture")
    # Development repositories track task data; export-ignore must exclude it.
    run(source, "git", "add", "-f", "configs/benchmarks/example/tasks.json")
    run(source, "git", "-c", "user.name=Release test", "-c", "user.email=release@example.invalid", "commit", "-qm", "development task data")
    POLICY["export_source"](source, tmp_path / "safe.tar.gz")
    with tarfile.open(tmp_path / "safe.tar.gz") as archive:
        assert not any(POLICY["is_private_path"](member.name) or member.name.startswith(".git/") for member in archive.getmembers())
    (source / "unreviewed.py").write_text("# Uncommitted change\n")
    with pytest.raises(ValueError, match="working tree has uncommitted files"):
        POLICY["export_source"](source, tmp_path / "dirty.tar.gz")


def test_populated_environment_template_is_rejected(source, tmp_path):
    template = source / "configs/providers/safe.env.example"
    template.write_text("API_KEY=NON_SECRET_TEST_SENTINEL\n")
    with pytest.raises(ValueError, match="Environment example contains a value"):
        POLICY["check_empty_env_example"](template.name, template.read_bytes())
    result = subprocess.run([sys.executable, "-I", "-B", "-c", "from setuptools.build_meta import build_sdist; build_sdist('../rejected')"],
                            cwd=source, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode != 0
    assert b"Environment example contains a value" in result.stdout


def test_direct_wheel_rejects_symlink_in_explicit_data_files(source):
    (source / "public-link.md").symlink_to(source / "skills/example/auth.json")
    config = source / "pyproject.toml"
    config.write_text(config.read_text().replace('"configs/local/linked.md"', '"public-link.md"'))
    result = subprocess.run([sys.executable, "-I", "-B", "-c", "from setuptools.build_meta import build_wheel; build_wheel('../rejected')"],
                            cwd=source, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode != 0
    assert b"Release sources must be regular files: public-link.md" in result.stdout


@pytest.mark.parametrize("name,is_link", [
    ("auth.json", False), ("nested/.codex/config.toml", False), ("old-resource.yaml", True),
])
def test_direct_wheel_rejects_stale_private_build_output(source, name, is_link):
    path = source / "build/lib/example" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if is_link:
        path.symlink_to(source / "skills/example/auth.json")
    else:
        path.write_bytes(SENTINEL)
    result = subprocess.run([sys.executable, "-I", "-B", "-c", "from setuptools.build_meta import build_wheel; build_wheel('../rejected')"],
                            cwd=source, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
    assert result.returncode != 0
    assert b"Private file or link remains in build output" in result.stdout
