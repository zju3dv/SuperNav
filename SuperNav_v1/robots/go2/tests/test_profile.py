"""Check the shipped profile's dependency boundary without importing hardware code."""

import ast
from pathlib import Path
import tempfile
import unittest

from supernav.registry import regenerate

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "supernav/overlay"


class PixelGoalProfileTests(unittest.TestCase):
    def test_pixelgoal_imports_resolve_without_removed_modules(self):
        paths = list((OVERLAY / "dimos/robot/unitree/pixelgoal").glob("*.py"))
        paths += list((OVERLAY / "dimos/robot/unitree/go2/blueprints/pixelgoal").glob("*.py"))
        for path in paths:
            prefix = path.relative_to(OVERLAY).with_suffix("").parts[:-1]
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.ImportFrom):
                    continue
                parts = (list(prefix[:len(prefix) - node.level + 1]) if node.level else [])
                parts += (node.module or "").split(".") if node.module else []
                if not parts or parts[0] != "dimos":
                    continue
                self.assertNotIn("nomad", parts, str(path))
                relative = Path(*parts)
                exists = any(
                    (base / relative.with_suffix(".py")).is_file()
                    or (base / relative / "__init__.py").is_file()
                    for base in (OVERLAY, ROOT / "dimos")
                )
                self.assertTrue(exists, f"{path.name}: missing {'.'.join(parts)}")

    def test_registry_exposes_pixelgoal_without_policy_modules(self):
        # Registry generation needs only Python text, so no robot imports occur.
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            stage = Path(tmp)
            scanner_dir = stage / "dimos/robot"
            scanner_dir.mkdir(parents=True)
            for name in ("get_all_blueprints.py", "test_all_blueprints_generation.py"):
                shutil.copyfile(ROOT / "dimos/dimos/robot" / name, scanner_dir / name)
            shutil.copytree(OVERLAY / "dimos", stage / "dimos", dirs_exist_ok=True)
            regenerate(stage)
            registry = (scanner_dir / "all_blueprints.py").read_text()
            self.assertIn('"unitree-go2-pixelgoal"', registry)
            self.assertIn("PixelGoalMapNavModule", registry)
            for removed in ("NomadNavModule", "NomadChunkExecutorModule", "PixelGoalNavModule", "AWM", "awm_policy"):
                self.assertNotIn(removed, registry)


if __name__ == "__main__":
    unittest.main()
