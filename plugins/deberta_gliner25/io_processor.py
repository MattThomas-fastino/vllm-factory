"""IO processor for deberta_gliner25 — same HTTP shape as the span plugin."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Dict

from transformers import AutoTokenizer
from vllm.config import VllmConfig

from plugins.deberta_gliner2.processor import format_results, normalize_gliner2_schema
from plugins.deberta_gliner25.processor import decode_boundary_output, preprocess_boundary
from vllm_factory.io.base import FactoryIOProcessor, PoolingRequestOutput, PromptType, TokensPrompt

logger = logging.getLogger(__name__)

_ADAPTER_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-:/]{1,128}$")


@dataclass
class GLiNER25Input:
    text: str
    schema: Dict = field(default_factory=dict)
    threshold: float = 0.5
    include_confidence: bool = False
    include_spans: bool = False
    truncate_overflow_text: bool = False
    adapter: str | None = None


class DeBERTaGLiNER25IOProcessor(FactoryIOProcessor):
    """Boundary IO processor. HTTP contract matches deberta_gliner2."""

    pooling_task = "plugin"

    def __init__(self, vllm_config: VllmConfig, *args: Any, **kwargs: Any) -> None:
        super().__init__(vllm_config, *args, **kwargs)
        model_id = vllm_config.model_config.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_id, use_fast=True, trust_remote_code=True
        )

    @staticmethod
    def _coerce_bool(value: Any, field_name: str) -> bool:
        if isinstance(value, bool):
            return value
        raise ValueError(f"'{field_name}' must be a boolean")

    @staticmethod
    def _coerce_adapter(value: Any) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("'adapter' must be a string or null")
        stripped = value.strip()
        if not stripped:
            return None
        if not _ADAPTER_NAME_RE.match(stripped):
            raise ValueError(f"'adapter' must match ^[A-Za-z0-9_.\\-:/]{{1,128}}$ — got {value!r}")
        return stripped

    def factory_parse(self, data: Any) -> GLiNER25Input:
        if hasattr(data, "data"):
            data = data.data
        elif isinstance(data, dict) and "data" in data:
            data = data["data"]
        if not isinstance(data, dict):
            raise ValueError("Expected request data dict")
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("'text' is required")
        threshold = data.get("threshold", 0.5)
        try:
            threshold = float(threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("'threshold' must be a number") from exc
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("'threshold' must be between 0 and 1")
        raw_schema = data.get("schema")
        labels = data.get("labels")
        if raw_schema is not None:
            schema = normalize_gliner2_schema(raw_schema)
        elif labels is not None:
            schema = normalize_gliner2_schema({"entities": labels})
        else:
            raise ValueError("Request must include schema or labels")
        return GLiNER25Input(
            text=text,
            schema=schema,
            threshold=threshold,
            include_confidence=self._coerce_bool(
                data.get("include_confidence", False), "include_confidence"
            ),
            include_spans=self._coerce_bool(data.get("include_spans", False), "include_spans"),
            truncate_overflow_text=self._coerce_bool(
                data.get("truncate_overflow_text", False), "truncate_overflow_text"
            ),
            adapter=self._coerce_adapter(data.get("adapter")),
        )

    def factory_pre_process(
        self,
        parsed_input: GLiNER25Input,
        request_id: str | None,
    ) -> PromptType | Sequence[PromptType]:
        started = time.perf_counter()
        result = preprocess_boundary(
            self._tokenizer,
            parsed_input.text,
            parsed_input.schema,
            threshold=parsed_input.threshold,
            include_confidence=parsed_input.include_confidence,
            include_spans=parsed_input.include_spans,
        )
        extra = result["extra_kwargs"]
        postprocess_meta = {
            "schema_dict": parsed_input.schema,
            "threshold": parsed_input.threshold,
            "include_confidence": parsed_input.include_confidence,
            "include_spans": parsed_input.include_spans,
            "adapter": parsed_input.adapter,
            "_observability": {
                "request_id": request_id or "_offline",
                "preprocess_elapsed_ms": (time.perf_counter() - started) * 1000.0,
            },
        }
        self._stash(extra_kwargs=extra, request_id=request_id, meta=postprocess_meta)
        return TokensPrompt(prompt_token_ids=result["input_ids"])

    def factory_post_process(
        self,
        model_output: Sequence[PoolingRequestOutput],
        request_meta: Any,
    ) -> Dict:
        if not model_output or request_meta is None:
            return {}
        raw = model_output[0].outputs.data
        if raw is None:
            return {}
        decoded = decode_boundary_output(raw, request_meta.get("schema_dict") or {})
        formatted = format_results(
            decoded,
            threshold=request_meta.get("threshold", 0.5),
            include_confidence=request_meta.get("include_confidence", False),
            include_spans=request_meta.get("include_spans", False),
        )
        if isinstance(formatted, dict):
            for key in ("entities", "classifications", "structures", "relations"):
                formatted.setdefault(key, {} if key != "structures" else {})
            if request_meta.get("adapter") is not None:
                formatted.setdefault("adapter", request_meta["adapter"])
        return formatted


def get_processor_cls() -> str:
    return "plugins.deberta_gliner25.io_processor.DeBERTaGLiNER25IOProcessor"
