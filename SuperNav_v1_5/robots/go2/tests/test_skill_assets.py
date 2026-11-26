"""Exercise authored Markdown loading and task copies without robot dependencies."""

import ast
import importlib.util
from pathlib import Path
import re
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT / "supernav/overlay/dimos/agents"


class SkillAssetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        # Model an installed package outside the source repository. No CWD or
        # developer checkout should be needed to locate runtime skill assets.
        installed = self.root / "installed/dimos/agents"
        (installed / "mcp").mkdir(parents=True)
        shutil.copytree(AGENTS / "skills", installed / "skills")
        loader_path = installed / "mcp/skill_assets.py"
        shutil.copyfile(AGENTS / "mcp/skill_assets.py", loader_path)
        spec = importlib.util.spec_from_file_location("isolated_skill_assets", loader_path)
        self.loader = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.loader)
        self.namespace = {
            "Path": Path,
            **{name: getattr(self.loader, name) for name in
               ("read_skill", "read_references", "copy_skill")},
        }
        # Execute the real resource declarations and writer entry points while
        # excluding unrelated hardware/MCP imports from the harness.
        tree = ast.parse((AGENTS / "mcp/external_harness_codex.py").read_text())
        nodes = []
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
                name = node.targets[0].id
                if name.endswith(("_SKILL_TEXT", "_REFERENCES")):
                    nodes.append(node)
            elif isinstance(node, ast.FunctionDef) and node.name.startswith("_write_") and "skill" in node.name:
                nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "harness_skill_io", "exec"), self.namespace)

    def test_harness_reads_assets_from_relocated_package(self):
        self.assertEqual(
            self.namespace["GO2_PIXELGOAL_SKILL_TEXT"],
            (AGENTS / "skills/go2-pixelgoal/SKILL.md").read_text(),
        )
        references = self.namespace["GO2_GLOBAL_NAVIGATION_PIXELGOAL_REFERENCES"]
        self.assertEqual(list(references), ["entrance-search.md", "exploration-and-return.md", "recovery-and-stop.md"])
        for name, content in references.items():
            self.assertEqual(content, (AGENTS / "skills/global-navigation-pixelgoal/references" / name).read_text())

    def test_all_harness_writers_copy_markdown_and_resolve_references(self):
        for name, writer in self.namespace.items():
            if not name.startswith("_write_"):
                continue
            with self.subTest(writer=name):
                destination = writer(self.root / name)
                skill = destination.parent.name
                original = AGENTS / "skills" / skill
                expected = {p.relative_to(original) for p in original.rglob("*.md")}
                actual = {p.relative_to(destination.parent) for p in destination.parent.rglob("*.md")}
                self.assertEqual(expected, actual)
                for relative in expected:
                    self.assertEqual((original / relative).read_bytes(), (destination.parent / relative).read_bytes())
                for reference in re.findall(r"references/[\w-]+\.md", destination.read_text()):
                    self.assertTrue((destination.parent / reference).is_file(), reference)

    def test_task_edits_do_not_change_authored_skill(self):
        destination = self.namespace["_write_workspace_go2_pixelgoal_skill"](self.root / "run")
        original = self.loader.read_skill("go2-pixelgoal")
        destination.write_text("local task edit\n")
        self.assertEqual(self.loader.read_skill("go2-pixelgoal"), original)
        self.namespace["_write_workspace_go2_pixelgoal_skill"](self.root / "run")
        self.assertEqual(destination.read_text(), original)

    def test_missing_packaged_skill_fails_without_inline_fallback(self):
        (self.loader.SKILLS_ROOT / "go2-pixelgoal/SKILL.md").unlink()
        with self.assertRaises(FileNotFoundError):
            self.loader.read_skill("go2-pixelgoal")
        with self.assertRaises(FileNotFoundError):
            self.loader.copy_skill("go2-pixelgoal", self.root / "run")
        self.assertFalse((self.root / "run").exists())


if __name__ == "__main__":
    unittest.main()
