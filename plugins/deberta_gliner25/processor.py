"""Boundary serving processor: GLiNER2 collate + Pioneer response keys."""

from __future__ import annotations

from typing import Any

import torch

from plugins.deberta_gliner2.processor import decode_output
from vllm_factory.optional_deps import require


def _schema_transformer(tokenizer):
    processor_mod = require("gliner2.processor", "gliner2", purpose="GLiNER25 preprocess")
    return processor_mod.SchemaTransformer(tokenizer=tokenizer, token_pooling="first")


def _tensor_to_list(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _layout_payload(layout: Any) -> dict[str, Any]:
    queries = []
    for spec in getattr(layout, "queries", ()):
        queries.append(
            {
                "task_index": getattr(spec, "task_index", 0),
                "role_index": getattr(spec, "role_index", 0),
                "task_type": getattr(spec, "task_type", ""),
                "task_name": getattr(spec, "task_name", ""),
                "role_name": getattr(spec, "role_name", ""),
            }
        )
    return {"queries": queries}


def preprocess_boundary(
    tokenizer,
    text: str,
    schema: dict[str, Any],
    *,
    threshold: float = 0.5,
    include_confidence: bool = False,
    include_spans: bool = False,
) -> dict[str, Any]:
    """Run GLiNER2 boundary collate and stash routing tensors for the pooler."""
    transformer = _schema_transformer(tokenizer)
    batch = transformer.collate_fn_inference(
        [(text, schema)],
        architecture="boundary",
        error_policy="raise",
    )
    input_ids = _tensor_to_list(batch.input_ids[0])
    extra = {
        "input_ids": input_ids,
        "text_word_indices": _tensor_to_list(batch.text_word_indices[0]),
        "text_word_mask": _tensor_to_list(batch.text_word_mask[0]),
        "query_marker_indices": _tensor_to_list(batch.query_marker_indices[0]),
        "query_marker_mask": _tensor_to_list(batch.query_marker_mask[0]),
        "cls_marker_indices": _tensor_to_list(batch.cls_marker_indices[0]),
        "cls_marker_mask": _tensor_to_list(batch.cls_marker_mask[0]),
        "start_mappings": [list(batch.start_mappings[0])],
        "end_mappings": [list(batch.end_mappings[0])],
        "original_texts": [text],
        "original_schemas": [schema],
        "task_types": [list(batch.task_types[0])],
        "schema_counts": [int(batch.schema_counts[0])],
        "schema_tokens_list": [batch.schema_tokens_list[0]],
        "schema_special_indices": [batch.schema_special_indices[0]],
        "text_word_counts": [int(batch.text_word_counts[0])],
        "query_layouts": [_layout_payload(batch.query_layouts[0])],
        "record_specs": getattr(batch, "record_specs", ()),
        "threshold": threshold,
        "include_confidence": include_confidence,
        "include_spans": include_spans,
        "metadata_list": [{}],
    }
    return {
        "input_ids": input_ids,
        "extra_kwargs": extra,
        "schema_dict": schema,
    }


def reshape_boundary_output(sample: dict[str, Any]) -> dict[str, Any]:
    """Map GLiNER2 decode keys onto Pioneer ``entities/classifications/structures/relations``."""
    entities = sample.get("entities", {})
    if isinstance(entities, list) and entities and isinstance(entities[0], dict):
        merged: dict[str, Any] = {}
        for item in entities:
            merged.update(item)
        entities = merged
    classifications: dict[str, Any] = {}
    structures: dict[str, Any] = {}
    relations: dict[str, Any] = {}
    for key, value in sample.items():
        if key == "entities":
            continue
        if _is_relation_payload(value):
            relations[key] = value
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            structures[key] = value
        else:
            classifications[key] = value
    return {
        "entities": entities or {},
        "classifications": classifications,
        "structures": structures,
        "relations": relations,
    }


def _is_relation_payload(value: Any) -> bool:
    if not isinstance(value, list) or not value or not isinstance(value[0], dict):
        return False
    keys = set(value[0])
    return bool(keys & {"head", "tail", "subject", "object", "src", "dst"})


def decode_boundary_output(raw_output, schema: dict[str, Any] | None = None) -> dict[str, Any]:
    """Unpack the JSON byte tensor produced by the boundary pooler."""
    decoded = decode_output(raw_output, schema or {})
    if any(key in decoded for key in ("entities", "classifications", "structures", "relations")):
        if "type" not in decoded:
            return {
                "entities": decoded.get("entities", {}),
                "classifications": decoded.get("classifications", {}),
                "structures": decoded.get("structures", {}),
                "relations": decoded.get("relations", {}),
            }
    return reshape_boundary_output(decoded)
