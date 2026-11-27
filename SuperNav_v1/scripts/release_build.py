"""Filter private source files before setuptools writes its release manifest."""

from pathlib import Path
import os
import runpy

from setuptools.command.egg_info import egg_info
from setuptools.command.build_py import build_py


_policy = runpy.run_path(str(Path(__file__).with_name("export_release_source.py")))


def _safe_source(name: str) -> bool:
    path = Path(name)
    if _policy["is_private_path"](path.as_posix()):
        return False
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"Release sources must be regular files: {name}")
    if _policy["is_env_example"](name):
        _policy["check_empty_env_example"](name, path.read_bytes())
    return True


def _filter_data_files(distribution) -> None:
    distribution.data_files = [
        (target, [name for name in names if _safe_source(name)])
        for target, names in distribution.data_files or ()
        if not _policy["is_private_path"](target)
    ]


class SafeEggInfo(egg_info):
    def find_sources(self) -> None:
        _filter_data_files(self.distribution)
        super().find_sources()
        safe_files = [name for name in self.filelist.files if _safe_source(name)]
        self.filelist.files = safe_files
        self.write_file("source manifest", str(Path(self.egg_info) / "SOURCES.txt"), "\n".join(safe_files) + "\n")


class SafeBuildPy(build_py):
    def run(self):
        _filter_data_files(self.distribution)
        super().run()
        # install_lib copies the whole build tree, including stale incremental output.
        for directory, dirs, files in os.walk(self.build_lib, followlinks=False):
            for name in dirs + files:
                path = Path(directory) / name
                relative = path.relative_to(self.build_lib).as_posix()
                if path.is_symlink() or _policy["is_private_path"](relative):
                    raise ValueError(f"Private file or link remains in build output: {relative}")
                if path.is_file() and _policy["is_env_example"](relative):
                    _policy["check_empty_env_example"](relative, path.read_bytes())

    def find_data_files(self, package, src_dir):
        return [name for name in super().find_data_files(package, src_dir) if _safe_source(name)]

    def find_package_modules(self, package, package_dir):
        return [item for item in super().find_package_modules(package, package_dir) if _safe_source(item[2])]
