"""Load and copy the Markdown skills shipped with the Go2 operator runtime."""

from pathlib import Path
import shutil


SKILLS_ROOT = Path(__file__).resolve().parent.parent / "skills"


def read_skill(name: str) -> str:
    return (SKILLS_ROOT / name / "SKILL.md").read_text(encoding="utf-8")


def read_references(name: str, filenames: tuple[str, ...]) -> dict[str, str]:
    """Keep the declared order when embedding references in legacy prompts."""
    return {
        filename: (SKILLS_ROOT / name / "references" / filename).read_text(encoding="utf-8")
        for filename in filenames
    }


def copy_skill(name: str, destination_root: Path) -> Path:
    """Copy the authored skill and its references into a run-local workspace."""
    source = SKILLS_ROOT / name
    # Fail before creating a partial destination if the installed asset is absent.
    if not (source / "SKILL.md").is_file():
        raise FileNotFoundError(f"Missing packaged skill: {source / 'SKILL.md'}")
    destination = destination_root / name
    shutil.copytree(source, destination, dirs_exist_ok=True)
    return destination / "SKILL.md"
