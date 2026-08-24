"""Conftest for deberta_gliner25 CPU-only tests.

Loads processor.py by file path so tests never import the plugin __init__
(which chain-imports model.py → vllm).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

_PLUGIN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "deberta_gliner25"
_PROCESSOR_PATH = _PLUGIN_DIR / "processor.py"
_SPAN_DIR = Path(__file__).resolve().parents[2] / "plugins" / "deberta_gliner2"

for pkg_name, pkg_path in [
    ("plugins", [str(_PLUGIN_DIR.parent)]),
    ("plugins.deberta_gliner25", [str(_PLUGIN_DIR)]),
    ("plugins.deberta_gliner2", [str(_SPAN_DIR)]),
]:
    if pkg_name not in sys.modules:
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = pkg_path
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg

_span_proc = "plugins.deberta_gliner2.processor"
if _span_proc not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        _span_proc, str(_SPAN_DIR / "processor.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_span_proc] = mod
    spec.loader.exec_module(mod)

_mod_name = "plugins.deberta_gliner25.processor"
if _mod_name not in sys.modules:
    spec = importlib.util.spec_from_file_location(_mod_name, str(_PROCESSOR_PATH))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_mod_name] = mod
    spec.loader.exec_module(mod)
