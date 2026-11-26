"""Exercise assembly against a tiny real Git repository, without robot imports."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from supernav.prepare import ROOT, prepare, verify_inputs, verify_runtime


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        repo = self.root / "dimos"
        (repo / "dimos/robot").mkdir(parents=True)
        for name in ["get_all_blueprints.py", "test_all_blueprints_generation.py"]:
            shutil.copyfile(ROOT / "dimos/dimos/robot" / name, repo / "dimos/robot" / name)
        (repo / "value.txt").write_text("upstream\n")
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(["git", "-C", str(repo), "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "commit", "-qm", "public fixture"], check=True)
        commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
        sn = self.root / "supernav"
        (sn / "overlay/dimos/robot").mkdir(parents=True)
        (sn / "patches").mkdir()
        for name in ["prepare.py", "registry.py"]:
            shutil.copyfile(ROOT / "supernav" / name, sn / name)
        overlay = sn / "overlay/dimos/robot/new_robot.py"
        overlay.write_text("class NewRobot(Module):\n    pass\n")
        self.patch = sn / "patches/value.patch"
        self.patch.write_text("diff --git a/value.txt b/value.txt\n--- a/value.txt\n+++ b/value.txt\n@@ -1 +1 @@\n-upstream\n+integrated\n")
        self.manifest = {
            "schema_version": 1,
            "upstream_commit": commit,
            "overlay": [{"path": "dimos/robot/new_robot.py", "sha256": hashlib.sha256(overlay.read_bytes()).hexdigest(), "executable": False}],
            "patches": [{"path": "patches/value.patch", "sha256": hashlib.sha256(self.patch.read_bytes()).hexdigest()}],
        }
        self.write_manifest()

    def write_manifest(self):
        (self.root / "supernav/manifest.json").write_text(json.dumps(self.manifest))

    def test_assembly_is_repeatable_and_leaves_submodule_clean(self):
        runtime = prepare(self.root)
        self.assertEqual((runtime / "value.txt").read_text(), "integrated\n")
        self.assertEqual((self.root / "dimos/value.txt").read_text(), "upstream\n")
        self.assertIn('"new-robot"', (runtime / "dimos/robot/all_blueprints.py").read_text())
        (runtime / "extension.so").write_bytes(b"build artifact")
        self.assertEqual(prepare(self.root), runtime)
        self.assertTrue((runtime / "extension.so").exists())
        self.assertFalse(subprocess.check_output(["git", "-C", str(self.root / "dimos"), "status", "--porcelain"], text=True))

    def test_unlisted_or_changed_overlay_is_rejected(self):
        path = self.root / "supernav/overlay/unlisted.py"
        path.write_text("unexpected = True\n")
        with self.assertRaisesRegex(ValueError, "file list"):
            verify_inputs(self.root)
        path.unlink()
        (self.root / "supernav/overlay/dimos/robot/new_robot.py").write_text("modified = True\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            prepare(self.root)
        self.assertFalse((self.root / "runtime/dimos").exists())

    def test_archive_ignores_local_lfs_smudge_filters(self):
        repo = self.root / "dimos"
        git = ["git", "-C", str(repo)]
        # A deterministic stand-in for a populated LFS cache. Export must use
        # the committed bytes regardless of this machine's filter behavior.
        for key, value in {
            "filter.lfs.clean": "cat",
            "filter.lfs.smudge": "sed s/pointer/cached-media/",
            "filter.lfs.process": "",
            "filter.lfs.required": "false",
        }.items():
            subprocess.run([*git, "config", key, value], check=True)
        (repo / ".gitattributes").write_text("media.bin filter=lfs\n")
        (repo / "media.bin").write_text("pointer\n")
        subprocess.run([*git, "add", "."], check=True)
        subprocess.run([*git, "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                        "-c", "commit.gpgsign=false", "commit", "-qm", "LFS fixture"], check=True)
        self.manifest["upstream_commit"] = subprocess.check_output(
            [*git, "rev-parse", "HEAD"], text=True).strip()
        self.write_manifest()
        runtime = prepare(self.root)
        self.assertEqual((runtime / "media.bin").read_text(), "pointer\n")

    def test_wrong_pin_dirty_upstream_and_dirty_runtime_are_rejected(self):
        runtime = prepare(self.root)
        (self.root / "dimos/value.txt").write_text("local edit\n")
        with self.assertRaisesRegex(ValueError, "must be clean"):
            verify_inputs(self.root)
        (self.root / "dimos/value.txt").write_text("upstream\n")
        (runtime / "value.txt").write_text("local runtime edit\n")
        with self.assertRaisesRegex(ValueError, "runtime source modified"):
            verify_runtime(self.root)
        prepare(self.root, rebuild=True)
        self.assertEqual((runtime / "value.txt").read_text(), "integrated\n")
        self.manifest["upstream_commit"] = "0" * 40
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "wrong public"):
            verify_inputs(self.root)

    def test_failed_patch_does_not_destroy_previous_runtime(self):
        runtime = prepare(self.root)
        previous = (runtime / ".supernav-build.json").read_bytes()
        self.patch.write_text(self.patch.read_text().replace("-upstream", "-does-not-match"))
        self.manifest["patches"][0]["sha256"] = hashlib.sha256(self.patch.read_bytes()).hexdigest()
        self.write_manifest()
        with self.assertRaises(subprocess.CalledProcessError):
            prepare(self.root, rebuild=True)
        self.assertEqual((runtime / ".supernav-build.json").read_bytes(), previous)
        self.assertEqual((runtime / "value.txt").read_text(), "integrated\n")


if __name__ == "__main__":
    unittest.main()
