"""Construct the boundary pooler against local gliner2.layers.

Stubs vLLM (not required for construction) the same way the span pooler
tests do, then loads poolers/gliner25.py by file path so poolers/__init__.py
never imports ColBERT.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_GLINER2_ROOT = Path("/Users/matthewthomas/Documents/GLiNER2")
if _GLINER2_ROOT.exists() and str(_GLINER2_ROOT) not in sys.path:
    sys.path.insert(0, str(_GLINER2_ROOT))

pytest.importorskip("gliner2", reason="boundary pooler construct needs gliner2")

_ROOT = Path(__file__).resolve().parents[2]
_STUBS: dict[str, types.ModuleType] = {}
for pkg_name in [
    "vllm",
    "vllm.config",
    "vllm_factory",
    "vllm_factory.pooling",
    "vllm_factory.pooling.protocol",
    "vllm_factory.pooling.vllm_adapter",
]:
    if pkg_name not in sys.modules:
        mod = types.ModuleType(pkg_name)
        mod.__path__ = []
        _STUBS[pkg_name] = mod
        sys.modules[pkg_name] = mod

if "vllm_factory.pooling.protocol" in _STUBS:
    _STUBS["vllm_factory.pooling.protocol"].PoolerContext = type("PoolerContext", (), {})
    _STUBS["vllm_factory.pooling.protocol"].split_hidden_states = lambda *a, **kw: None

_opt_name = "vllm_factory.optional_deps"
if _opt_name not in sys.modules or _opt_name.split(".")[0] in _STUBS:
    spec = importlib.util.spec_from_file_location(
        _opt_name, _ROOT / "vllm_factory" / "optional_deps.py"
    )
    opt = importlib.util.module_from_spec(spec)
    sys.modules[_opt_name] = opt
    spec.loader.exec_module(opt)

_POOLER_PATH = _ROOT / "poolers" / "gliner25.py"
_spec = importlib.util.spec_from_file_location("gliner25_pooler_test", str(_POOLER_PATH))
_pooler_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pooler_mod)
GLiNER25BoundaryPooler = _pooler_mod.GLiNER25BoundaryPooler


def test_boundary_pooler_constructs_and_exposes_checkpoint_prefixes():
    pooler = GLiNER25BoundaryPooler(hidden_size=32, boundary_head={"dropout": 0.0})
    keys = pooler.state_dict().keys()
    assert any(k.startswith("classifier.") for k in keys)
    assert any(k.startswith("boundary_head.") for k in keys)
    if pooler.enable_records:
        assert any(k.startswith("record_decoder.") for k in keys)
    if pooler.enable_relations:
        assert any(k.startswith("relation_scorer.") for k in keys)
    assert pooler.get_tasks() == {"embed", "classify", "plugin"}


def test_decode_host_borrows_modules_without_module_init() -> None:
    """Warmup used to crash: assigning nn.Modules onto object.__new__ host."""
    pooler = GLiNER25BoundaryPooler(hidden_size=32, boundary_head={"dropout": 0.0})
    host = pooler._decode_host({"text_states": None})
    assert host.boundary_head is pooler.boundary_head
    assert host.classifier is pooler.classifier
    assert host.strict_extraction is True
    assert host._encode_core({})["text_states"] is None
