"""Run the pinned upstream's AST registry generator without importing robot code."""

import ast
from pathlib import Path
import re


def regenerate(runtime: Path) -> None:
    # Use upstream's own scanner, including its naming rules and exclusions.
    # Only its stdlib imports, constants and function definitions are evaluated.
    namespace = {"__name__": "supernav_registry", "re": re}
    naming_path = runtime / "dimos/robot/get_all_blueprints.py"
    naming_tree = ast.parse(naming_path.read_text())
    naming = next(n for n in naming_tree.body if isinstance(n, ast.FunctionDef) and n.name == "class_name_to_registry_key")
    exec(compile(ast.Module(body=[naming], type_ignores=[]), str(naming_path), "exec"), namespace)
    scanner_path = runtime / "dimos/robot/test_all_blueprints_generation.py"
    tree = ast.parse(scanner_path.read_text())
    selected = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("dimos"):
            continue
        if isinstance(node, ast.Import) and any(alias.name == "pytest" for alias in node.names):
            continue
        selected.append(node)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(scanner_path), "exec"), namespace)
    blueprints, modules = namespace["_scan_for_blueprints"](runtime / "dimos")
    content = namespace["_generate_all_blueprints_content"](blueprints, modules)
    (runtime / "dimos/robot/all_blueprints.py").write_text(content)
