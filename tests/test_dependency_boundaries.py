"""Enforce the dependency-direction invariants from the project's design notes:

1. proxyml_core.schema / export / _version never import anything outside
   stdlib + numpy — they must stay usable in a REST-only, sklearn-free install.
2. proxyml_core.modeling may additionally import sklearn/scipy (that's the
   whole point of the [modeling] extra), but nothing else new.
"""

import ast
import sys
from pathlib import Path

import proxyml_core

_PURE_MODULES = ["schema.py", "export.py", "_version.py"]
_ALLOWED_PURE_TOP_LEVEL = {"numpy"} | set(sys.stdlib_module_names)
_ALLOWED_MODELING_TOP_LEVEL = _ALLOWED_PURE_TOP_LEVEL | {"sklearn", "scipy"}

_PACKAGE_DIR = Path(proxyml_core.__file__).parent


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:  # skip relative imports (internal to proxyml_core)
                modules.add(node.module.split(".")[0])
    return modules


def test_pure_modules_only_import_stdlib_and_numpy():
    for filename in _PURE_MODULES:
        path = _PACKAGE_DIR / filename
        imported = _top_level_imports(path)
        disallowed = {m for m in imported if m != "proxyml_core"} - _ALLOWED_PURE_TOP_LEVEL
        assert not disallowed, f"{path} imports non-pure module(s): {disallowed}"


def test_modeling_modules_do_not_import_unexpected_dependencies():
    modeling_dir = _PACKAGE_DIR / "modeling"
    for path in modeling_dir.glob("*.py"):
        imported = _top_level_imports(path)
        disallowed = {m for m in imported if m != "proxyml_core"} - _ALLOWED_MODELING_TOP_LEVEL
        assert not disallowed, f"{path} imports unexpected module(s): {disallowed}"


def test_import_proxyml_core_without_sklearn_or_scipy():
    """The pure base must be importable even if sklearn/scipy are absent."""
    import subprocess

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "for m in ('sklearn', 'scipy'):\n"
            "    sys.modules[m] = None\n"  # forces ImportError if anything tries to import them
            "import proxyml_core\n"
            "proxyml_core.FeatureSchema(features=[])\n"
            "print('ok')\n",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
