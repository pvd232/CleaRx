"""Enforce concrete module and callable documentation in orchestration Python."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_orchestration_python_has_docstrings() -> None:
    """Require docstrings on each orchestration module, class, and callable."""
    paths = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "tests/orchestration").glob("*.py"))
    missing: list[str] = []
    for path in paths:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if not ast.get_docstring(tree):
            missing.append(f"{path.relative_to(ROOT)}:<module>")
        for node in ast.walk(tree):
            if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and not ast.get_docstring(node):
                missing.append(f"{path.relative_to(ROOT)}:{node.lineno}:{node.name}")
    assert not missing, "missing docstrings:\n" + "\n".join(missing)
