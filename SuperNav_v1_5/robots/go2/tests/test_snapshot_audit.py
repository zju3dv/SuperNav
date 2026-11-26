"""The provenance audit must detect drift outside the overlay manifest too."""

from pathlib import Path
import tempfile
import unittest

from supernav.audit_snapshot import REQUIRED_SKILL_PATHS, REQUIRED_SOURCE_PATHS, audit


OVERLAY = Path(__file__).resolve().parents[1] / "supernav/overlay"


class SnapshotAuditTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.snapshot = Path(tmp.name) / "snapshot"
        self.runtime = Path(tmp.name) / "runtime"
        for root in [self.snapshot, self.runtime]:
            for relative in REQUIRED_SOURCE_PATHS:
                self.write(root / relative, "value = 1\n")
            for relative in REQUIRED_SKILL_PATHS:
                self.write(root / relative, (OVERLAY / relative).read_text())
        self.source = self.snapshot / "dimos/robot/unitree/nomad"
        self.target = self.runtime / "dimos/robot/unitree/pixelgoal"
        for name in [
            "goal.py",
            "pixel_goal_map.py",
            "pixel_goal_map_module.py",
            "area_captioning.py",
            "live_state.py",
            "contracts.py",
            "pixel_goal_support.py",
        ]:
            text = (OVERLAY / "dimos/robot/unitree/pixelgoal" / name).read_text()
            self.write(self.target / name, text)
            # Default endpoint/path adaptations need not be present in the
            # synthetic approved source; identity is also valid after mapping.
            self.write(self.source / name, text)
        support = (self.source / "pixel_goal_support.py").read_text()
        for name in ["nav_module.py", "pixel_goal.py", "pixel_goal_module.py"]:
            self.write(self.source / name, support)
        blueprint = (
            OVERLAY
            / "dimos/robot/unitree/go2/blueprints/pixelgoal/unitree_go2_pixelgoal.py"
        ).read_text()
        self.blueprint = (
            self.runtime
            / "dimos/robot/unitree/go2/blueprints/pixelgoal/unitree_go2_pixelgoal.py"
        )
        self.write(self.blueprint, blueprint)
        source_blueprint = blueprint.replace(
            "unitree_go2_pixelgoal =", "unitree_go2_nomad_edge_mcp ="
        )
        source_blueprint = source_blueprint.replace(
            "PixelGoalMapNavModule.blueprint(camera_info=GO2Connection.camera_info_static),",
            "NomadChunkExecutorModule.blueprint(),\n        NomadNavModule.blueprint(),\n        pixelgoal_blueprint(),",
        )
        self.write(
            self.snapshot
            / "dimos/robot/unitree/go2/blueprints/nomad/unitree_go2_nomad_edge_mcp.py",
            source_blueprint,
        )

    @staticmethod
    def write(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    def test_identical_source_and_audited_backend_pruning_pass(self):
        report = audit(self.snapshot, self.runtime)
        self.assertTrue(report["passed"], report["failures"])
        self.assertTrue(
            all(
                check["expected_sha256"] == check["actual_sha256"]
                for check in report["checks"]
            )
        )

    def test_shared_dependency_drift_is_detected(self):
        relative = "dimos/mapping/voxels.py"
        self.write(self.runtime / relative, "value = 2\n")
        self.assertIn(relative, audit(self.snapshot, self.runtime)["failures"])

    def test_canonical_skill_fallback_cannot_silently_change_prompt(self):
        relative = "skills/go2-graph-memory-hierarchy/SKILL.md"
        (self.runtime / relative).unlink()
        self.assertIn(
            f"missing canonical skill: {relative}",
            audit(self.snapshot, self.runtime)["failures"],
        )

    def test_missing_required_module_is_detected(self):
        relative = "dimos/agents/mcp/mcp_adapter.py"
        (self.runtime / relative).unlink()
        self.assertIn(
            f"missing required source: {relative}",
            audit(self.snapshot, self.runtime)["failures"],
        )

    def test_blueprint_cannot_disable_snapshot_map_persistence(self):
        self.blueprint.write_text(
            self.blueprint.read_text().replace(
                "VoxelGridMapper.blueprint(emit_every=1)",
                "VoxelGridMapper.blueprint(emit_every=1, save_on_stop=False)",
            )
        )
        self.assertIn(
            "pixelgoal/blueprint:composition",
            audit(self.snapshot, self.runtime)["failures"],
        )

    def test_mapped_module_behavior_changes_are_detected(self):
        module = self.target / "pixel_goal_map.py"
        module.write_text(module.read_text() + "\nCHANGED_BEHAVIOR = True\n")
        self.assertIn(
            "pixelgoal/pixel_goal_map.py",
            audit(self.snapshot, self.runtime)["failures"],
        )


if __name__ == "__main__":
    unittest.main()
