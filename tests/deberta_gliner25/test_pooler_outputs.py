"""The pooler's output list must stay paired with the scheduled batch.

A coalesced batch holds several callers, so returning the wrong number of
payloads hands one caller another's extraction. Loads poolers/gliner25.py by
file path with vLLM stubbed, like the construct test.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[2]

for _pkg in [
    "vllm",
    "vllm.config",
    "vllm_factory",
    "vllm_factory.pooling",
    "vllm_factory.pooling.protocol",
    "vllm_factory.pooling.vllm_adapter",
]:
    if _pkg not in sys.modules:
        _mod = types.ModuleType(_pkg)
        _mod.__path__ = []
        sys.modules[_pkg] = _mod

_protocol = sys.modules["vllm_factory.pooling.protocol"]
if not hasattr(_protocol, "PoolerContext"):
    _protocol.PoolerContext = type("PoolerContext", (), {})
if not hasattr(_protocol, "split_hidden_states"):
    _protocol.split_hidden_states = lambda *a, **kw: None

_opt = "vllm_factory.optional_deps"
if _opt not in sys.modules or not hasattr(sys.modules[_opt], "require"):
    _spec = importlib.util.spec_from_file_location(
        _opt, _ROOT / "vllm_factory" / "optional_deps.py"
    )
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_opt] = _module
    _spec.loader.exec_module(_module)

_spec = importlib.util.spec_from_file_location(
    "gliner25_pooler_outputs", str(_ROOT / "poolers" / "gliner25.py")
)
pooler_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pooler_mod)


def _decode(payload: torch.Tensor) -> dict:
    """Unpack a payload the way the IO processor does."""
    data = payload.tolist()
    length = int(data[0])
    return json.loads(bytes(int(b) for b in data[1 : length + 1]).decode("utf-8"))


class _Ctx:
    """Minimal PoolerContext stand-in."""

    def __init__(self, seq_lengths: list[int], extra_kwargs: list[dict]) -> None:
        self.seq_lengths = seq_lengths
        self.extra_kwargs = extra_kwargs
        self.prompt_token_ids: list[list[int]] = []


def test_empty_payload_decodes_as_an_empty_result():
    assert _decode(pooler_mod._pack_json({}, torch.device("cpu"))) == {}


def test_empty_payloads_are_one_per_sequence():
    payloads = pooler_mod._empty_payloads(3, torch.device("cpu"))
    assert len(payloads) == 3
    assert all(_decode(p) == {} for p in payloads)


def test_split_failure_still_answers_every_sequence(monkeypatch):
    def _boom(hidden_states, seq_lengths):
        raise RuntimeError("bad lengths")

    monkeypatch.setattr(pooler_mod, "split_hidden_states", _boom)
    pooler = object.__new__(pooler_mod.GLiNER25BoundaryPooler)
    ctx = _Ctx([5, 7, 9], [{"a": 1}, {"b": 2}, {"c": 3}])

    outputs = pooler_mod.GLiNER25BoundaryPooler.forward(pooler, torch.zeros(21, 4), ctx)

    assert len(outputs) == 3
    assert all(_decode(payload) == {} for payload in outputs)


def test_missing_extras_do_not_shift_the_other_results(monkeypatch):
    monkeypatch.setattr(
        pooler_mod,
        "split_hidden_states",
        lambda hidden_states, seq_lengths: [torch.zeros(n, 4) for n in seq_lengths],
    )
    monkeypatch.setattr(
        pooler_mod.GLiNER25BoundaryPooler,
        "_process_one",
        lambda self, token_embs, extra: pooler_mod._pack_json(extra, token_embs.device),
    )
    monkeypatch.setattr(pooler_mod, "_can_batch_compact", lambda extras: False)
    pooler = object.__new__(pooler_mod.GLiNER25BoundaryPooler)
    # vLLM scheduled three sequences but only two carry extras.
    ctx = _Ctx([2, 3, 4], [{"first": True}, {}])

    outputs = pooler_mod.GLiNER25BoundaryPooler.forward(pooler, torch.zeros(9, 4), ctx)

    assert [_decode(payload) for payload in outputs] == [{"first": True}, {}, {}]
