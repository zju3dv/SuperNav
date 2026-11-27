"""Deployment boundary tests: no network, robot, sudo, or dependency installs."""

import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]


class DeploymentTests(unittest.TestCase):
    def run_script(self, name, *args, env=None):
        return subprocess.run(
            [str(ROOT / "deploy/bin" / name), *args],
            cwd=ROOT,
            env={**os.environ, **(env or {})},
            text=True,
            capture_output=True,
            timeout=20,
        )

    def preview(self, *args, rerun="0", **overrides):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "robot.env"
            config.write_text(
                'UNITREE_GO2_IP=192.0.2.10\n'
                'DIMOS_GO2_MOVE_API=joystick\n'
                f'SUPERNAV_RERUN={rerun}\n'
                'UNITREE_AES_128_KEY=DO_NOT_PRINT_TEST_SECRET\n'
            )
            return self.run_script(
                "start-go2-pixel-goal.sh", "--dry-run", *args,
                env={"SUPERNAV_CONFIG": str(config), "SUPERNAV_SECRETS": str(Path(tmp) / "absent"), **overrides},
            )

    def test_preview_is_headless_pinned_and_does_not_leak_secrets(self):
        result = self.preview(DIMOS_REPO="/old/checkout", DIMOS_PYTHON="/old/python")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--viewer none", result.stdout)
        self.assertIn("--no-go2-auto-stand", result.stdout)
        self.assertIn("--stack mapping --control local-planner", result.stdout)
        self.assertIn(str(ROOT / ".venv/bin/python"), result.stdout)
        self.assertIn("Policy: disabled", result.stdout)
        self.assertNotIn("DO_NOT_PRINT_TEST_SECRET", result.stdout + result.stderr)
        self.assertNotIn("/old/", result.stdout)

    def test_1008_and_rerun_are_explicit_options(self):
        result = self.preview("1008", rerun="1")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Move API: 1008", result.stdout)
        self.assertIn("--viewer rerun --rerun-open none --no-rerun-web", result.stdout)

    def test_invalid_arguments_fail_before_launch(self):
        for args in [("bogus",), ("joystick", "1008")]:
            result = self.preview(*args)
            self.assertNotEqual(result.returncode, 0)
        self.assertNotEqual(self.preview(rerun="yes").returncode, 0)

    def test_stage_contains_only_intended_system_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "stage"
            result = self.run_script("install-system.sh", "--stage", str(target))
            self.assertEqual(result.returncode, 0, result.stderr)
            files = sorted(p for p in target.rglob("*") if p.is_file())
            self.assertEqual(len(files), 8)
            self.assertTrue((target / "usr/local/sbin/gopro-usb-reconcile").stat().st_mode & 0o111)
            self.assertFalse((target / "etc/systemd/system/gopro-webcam@.service").exists())
            self.assertNotIn("gopro-webcam-reconcile.service", (target / "etc/udev/rules.d/70-gopro-webcam.rules").read_text())
            self.assertIn("239.255.76.67/32", (target / "etc/systemd/system/supernav-lcm-route.service").read_text())
            second = self.run_script("install-system.sh", "--stage", str(target))
            self.assertNotEqual(second.returncode, 0)

    def test_installers_default_to_plan(self):
        for name in ["install-system.sh", "install-python.sh"]:
            result = self.run_script(name)
            self.assertEqual(result.returncode, 0, result.stderr)
        self.assertNotEqual(self.run_script("install-system.sh", "--stage", "/").returncode, 0)

    def test_placeholder_config_cannot_start_robot(self):
        result = self.run_script(
            "start-go2-pixel-goal.sh",
            env={"SUPERNAV_CONFIG": str(ROOT / "deploy/config/robot.env.example"), "SUPERNAV_SECRETS": "/nonexistent-supernav-test"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("configure UNITREE_GO2_IP", result.stderr)

    def test_move_api_threshold_and_release_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "config"
            config.write_text("DIMOS_GO2_MOVE_API=joystick\nSUPERNAV_RERUN=0\n")
            env = dict(os.environ, SUPERNAV_CONFIG=str(config), SUPERNAV_SECRETS=str(Path(tmp) / "absent"))
            env.pop("DIMOS_ASTAR_MIN_ANGULAR_RPS", None)
            for api, expected in [("joystick", "0.2"), ("1008", "0.5")]:
                result = subprocess.run(
                    ["bash", "-c", 'source "$1"; load_config; set_move_api "$2"; printf "%s\\n%s\\n" "$DIMOS_ASTAR_MIN_ANGULAR_RPS" "$XDG_STATE_HOME"',
                     "test", str(ROOT / "deploy/lib/common.sh"), api],
                    env=env, text=True, capture_output=True, timeout=20,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), [expected, str(ROOT / "runtime/state")])

    def test_old_map_settings_cannot_enable_persistence(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "robot.env"
            config.write_text("DIMOS_VOXEL_MAP_LOAD_ON_START=latest\nDIMOS_VOXEL_MAP_SAVE_ON_STOP=1\n")
            result = subprocess.run(
                ["bash", "-c", 'source "$1"; load_config; printf "%s\\n%s\\n" "$DIMOS_VOXEL_MAP_LOAD_ON_START" "$DIMOS_VOXEL_MAP_SAVE_ON_STOP"',
                 "test", str(ROOT / "deploy/lib/common.sh")],
                env={**os.environ, "SUPERNAV_CONFIG": str(config),
                     "SUPERNAV_SECRETS": str(Path(tmp) / "absent"),
                     "DIMOS_VOXEL_MAP_LOAD_ON_START": "path", "DIMOS_VOXEL_MAP_SAVE_ON_STOP": "1"},
                text=True, capture_output=True, timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), ["none", "0"])

    def test_release_interpreter_overrides_old_shell_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "robot.env"
            config.write_text("DIMOS_VENV=/old/config-venv\nDIMOS_PYTHON=/old/config-python\n")
            result = subprocess.run(
                ["bash", "-c", 'source "$1"; load_config; printf "%s\\n%s\\n" "$DIMOS_VENV" "$DIMOS_PYTHON"',
                 "test", str(ROOT / "deploy/lib/common.sh")],
                env={**os.environ, "SUPERNAV_CONFIG": str(config),
                     "SUPERNAV_SECRETS": str(Path(tmp) / "absent"),
                     "DIMOS_VENV": "/old/shell-venv", "DIMOS_PYTHON": "/old/shell-python"},
                text=True, capture_output=True, timeout=20,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.splitlines(), [
                str(ROOT / ".venv"), str(ROOT / ".venv/bin/python")
            ])

    def test_shutdown_reaps_only_owned_process_groups(self):
        # Run the real launcher with shell-only hardware and source-validation substitutes.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copytree(ROOT / "deploy", root / "deploy")
            repo = root / "runtime/dimos"
            (repo / "scripts").mkdir(parents=True)
            bridge = repo / "scripts/go2_gopro_bridge.sh"
            bridge.write_text('#!/usr/bin/env bash\ntrap "exit 0" TERM INT\necho $$ > "$TEST_BRIDGE_PID"\necho "four YU12 streams ready"\nwhile :; do sleep 0.1; done\n')
            bridge.chmod(0o755)
            common = root / "deploy/lib/common.sh"
            common.write_text(common.read_text() + '\nverify_source() { :; }\nverify_runtime() { :; }\n')
            # Hardware preflight has separate boundary tests; bypass it here.
            (root / "deploy/bin/check.sh").write_text('#!/usr/bin/env bash\nexit 0\n')
            (root / ".venv/bin").mkdir(parents=True)
            python = root / ".venv/bin/python"
            python.write_text('#!/usr/bin/env bash\ntrap "exit 0" TERM INT\necho $$ > "$TEST_ROBOT_PID"\nwhile :; do sleep 0.1; done\n')
            python.chmod(0o755)
            (root / "deploy/config/robot.env").write_text('UNITREE_GO2_IP=192.0.2.10\nDIMOS_GOPRO_SERIALS="A B C D"\nUNITREE_AES_128_KEY=TEST_ONLY\nSUPERNAV_RERUN=0\n')
            bridge_pidfile = root / "bridge.pid"
            robot_pidfile = root / "robot.pid"
            env = {**os.environ, "SUPERNAV_CONFIG": str(root / "deploy/config/robot.env"),
                   "SUPERNAV_SECRETS": str(root / "absent"), "TEST_BRIDGE_PID": str(bridge_pidfile), "TEST_ROBOT_PID": str(robot_pidfile)}
            unrelated = subprocess.Popen(["sleep", "30"])
            launcher = subprocess.Popen([str(root / "deploy/bin/start-go2-pixel-goal.sh")], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            owned = []
            try:
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline and not robot_pidfile.exists() and launcher.poll() is None:
                    time.sleep(0.05)
                self.assertTrue(robot_pidfile.exists(), 'stub robot did not start')
                owned = [int(p.read_text()) for p in (bridge_pidfile, robot_pidfile)]
                launcher.send_signal(signal.SIGTERM)
                out, err = launcher.communicate(timeout=20)
                self.assertEqual(launcher.returncode, 143, out + err)
                for pid in owned:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
                self.assertIsNone(unrelated.poll(), 'launcher killed an unrelated process')
            finally:
                if launcher.poll() is None:
                    launcher.terminate()
                    launcher.communicate(timeout=20)
                for pid in owned:
                    try:
                        os.killpg(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                unrelated.terminate()
                unrelated.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
