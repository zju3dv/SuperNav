"""Check the shipped profile's dependency boundary without importing hardware code."""

import ast
import json
from pathlib import Path
import tempfile
import unittest

from supernav.registry import regenerate

ROOT = Path(__file__).resolve().parents[1]
OVERLAY = ROOT / "supernav/overlay"


class PixelGoalProfileTests(unittest.TestCase):
    def test_manifest_records_release_boundary_and_source_adaptations(self):
        manifest = json.loads((ROOT / "supernav/manifest.json").read_text())
        source = manifest["implementation_source"]
        self.assertEqual(
            source["base_commit"],
            "ae8160cc9c7ea0494b18a47395799e2698d6c891",
        )
        self.assertEqual(
            set(source["excluded_supernav_2_commits"]),
            {
                "8e33cea810b953c97e86bc33a76dec7bb7b424a6",
                "cfcc5de0025864e0f5b8538567f8bf8255e4044c",
            },
        )
        mapped = {entry["release"] for entry in source["mapped_source_paths"]}
        self.assertIn(
            "dimos/robot/unitree/pixelgoal/pixel_goal_map_module.py",
            mapped,
        )

    def test_structured_spatial_memory_pipeline_is_shipped(self):
        required = [
            "dimos/agents/harness/go2_graph_memory.py",
            "dimos/agents/harness/go2_prior_run.py",
            "dimos/mapping/occupancy/room_segmentation.py",
            "dimos/mapping/occupancy/habitat_area_segmentation.py",
            "dimos/mapping/relocalization/module.py",
            "dimos/robot/unitree/go2/motion_recorder.py",
            "dimos/robot/unitree/pixelgoal/area_captioning.py",
        ]
        for relative in required:
            self.assertTrue((OVERLAY / relative).is_file(), relative)

        map_module = (
            OVERLAY / "dimos/robot/unitree/pixelgoal/pixel_goal_map_module.py"
        ).read_text()
        self.assertIn("render_room_segmentation", map_module)
        self.assertIn("build_slam_area_nodes", map_module)
        self.assertIn("_terminal_2d_slam_map_content", map_module)

        authority = (
            OVERLAY / "dimos/agents/mcp/external_harness_authority_proxy.py"
        ).read_text()
        self.assertIn("Go2GraphMemoryStore", authority)
        self.assertIn("_overlay_graph_badges_on_image_block", authority)

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
