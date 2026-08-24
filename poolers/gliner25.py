"""GLiNER 2.5 boundary pooler.

Holds real ``gliner2`` head modules (never a copy) under the checkpoint
prefixes ``boundary_head`` / ``record_decoder`` / ``relation_scorer`` /
``classifier``. vLLM hidden states replace the encoder call; decode is
delegated to ``BoundaryExtractor._extract_from_batch``.
"""

from __future__ import annotations

import json
from dataclasses import fields
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn

from vllm_factory.pooling.protocol import PoolerContext, split_hidden_states


def _require(module: str, purpose: str):
    from vllm_factory.optional_deps import require

    return require(module, "gliner2", purpose=purpose)


class GLiNER25BoundaryPooler(nn.Module):
    """Boundary heads + serving split/pack. Does not subclass gliner2 types."""

    def __init__(self, hidden_size: int, boundary_head: dict[str, Any] | None = None):
        super().__init__()
        cfg = dict(boundary_head or {})
        layers = _require("gliner2.layers", "GLiNER25 classifier")
        model_mod = _require("gliner2.models.boundary.model", "GLiNER25 boundary head")
        config_mod = _require("gliner2.configuration", "GLiNER25 boundary settings")
        records_mod = _require("gliner2.models.boundary.records", "GLiNER25 record decoder")
        relations_mod = _require("gliner2.models.boundary.relations", "GLiNER25 relation scorer")

        settings_cls = config_mod.BoundaryHeadSettings
        allowed = {item.name for item in fields(settings_cls)}
        self.boundary_settings = settings_cls(
            **{key: value for key, value in cfg.items() if key in allowed}
        )
        self.hidden_size = hidden_size
        self.enable_records = bool(self.boundary_settings.enable_records)
        self.enable_relations = bool(self.boundary_settings.enable_relations)

        self.classifier = layers.create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size * 2],
            output_dim=1,
            dropout=cfg.get("dropout", 0.1),
            activation="relu",
            add_layer_norm=False,
        )
        self.boundary_head = model_mod.BoundaryHead(
            hidden_size,
            self.boundary_settings,
            query_dim=hidden_size,
            build_candidate_states=self.enable_records,
        )
        if self.enable_records:
            self.record_decoder = records_mod.RecordHead(
                hidden_size,
                self.boundary_settings.record_dim,
                self.boundary_settings.record_instance_queries,
            )
        if self.enable_relations:
            self.relation_pair_generator = relations_mod.TypedRelationPairGenerator(
                relations_mod.RelationProposalSettings(
                    heads_per_relation=self.boundary_settings.relation_heads_per_type,
                    tails_per_relation=self.boundary_settings.relation_tails_per_type,
                    pair_cap=self.boundary_settings.relation_pair_cap,
                    argument_threshold=self.boundary_settings.relation_argument_proposal_threshold,
                )
            )
            self.relation_scorer = relations_mod.SparseRelationScorer(
                hidden_size,
                dropout=self.boundary_settings.dropout,
                relation_query_dim=(
                    2 * hidden_size
                    if self.boundary_settings.directional_relation_states
                    else hidden_size
                ),
                use_biaffine_content=self.boundary_settings.relation_biaffine_content,
            )

    def get_tasks(self) -> set[str]:
        return {"embed", "classify", "plugin"}

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx: PoolerContext,
    ) -> list[torch.Tensor | None]:
        try:
            sequences = split_hidden_states(hidden_states, ctx.seq_lengths)
        except Exception:
            dummy = torch.zeros(4, device=hidden_states.device, dtype=hidden_states.dtype)
            return [dummy]

        outputs: list[torch.Tensor | None] = []
        for i, token_embs in enumerate(sequences):
            extra = ctx.extra_kwargs[i] if i < len(ctx.extra_kwargs) else {}
            if not extra:
                outputs.append(torch.zeros(4, device=token_embs.device, dtype=torch.float32))
                continue
            outputs.append(self._process_one(token_embs, extra))
        return outputs

    def _process_one(self, token_embs: torch.Tensor, extra: dict[str, Any]) -> torch.Tensor:
        batch = _batch_from_extra(extra, token_embs.device)
        core = self._core_from_hidden(token_embs.unsqueeze(0), batch)
        host = self._decode_host(core)
        samples = host._extract_from_batch(
            batch,
            float(extra.get("threshold", 0.5)),
            extra.get("metadata_list") or [{}],
            bool(extra.get("include_confidence", False)),
            bool(extra.get("include_spans", False)),
        )
        sample = samples[0] if samples else {}
        return _pack_json(sample, token_embs.device)

    def _decode_host(self, core: dict[str, Any]) -> Any:
        engine = _require("gliner2.models.boundary.engine", "GLiNER25 decode host")
        # ``object.__new__`` skips ``nn.Module.__init__``, so attribute
        # assignment must not go through ``Module.__setattr__`` (it raises
        # ``cannot assign module before Module.__init__() call``).
        host = object.__new__(engine.BoundaryExtractor)
        for name, value in (
            ("boundary_head", self.boundary_head),
            ("boundary_settings", self.boundary_settings),
            ("classifier", self.classifier),
            ("enable_records", self.enable_records),
            ("enable_relations", self.enable_relations),
            ("record_decoder", getattr(self, "record_decoder", None)),
            ("relation_scorer", getattr(self, "relation_scorer", None)),
            ("relation_pair_generator", getattr(self, "relation_pair_generator", None)),
            ("hidden_size", self.hidden_size),
            ("strict_extraction", True),
            ("_encode_core", lambda _batch: core),
        ):
            object.__setattr__(host, name, value)
        return host

    def _core_from_hidden(
        self, token_embeddings: torch.Tensor, batch: Any
    ) -> dict[str, Any]:
        """Gather word/query/cls states from vLLM hidden states (fast routing)."""
        h = token_embeddings.shape[-1]

        def gather_routed(indices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            safe_idx = indices.clamp(0, token_embeddings.shape[1] - 1)
            states = token_embeddings.gather(1, safe_idx.unsqueeze(-1).expand(-1, -1, h))
            return states * mask.unsqueeze(-1).to(states.dtype)

        text_states = gather_routed(batch.text_word_indices, batch.text_word_mask)
        query_states = gather_routed(batch.query_marker_indices, batch.query_marker_mask)
        text_mask = batch.text_word_mask
        query_mask = batch.query_marker_mask
        cls_states = gather_routed(batch.cls_marker_indices, batch.cls_marker_mask)

        ext_specs: list[list[dict[str, Any]]] = []
        cls_specs: list[list[dict[str, Any]]] = []
        rel_specs: list[list[dict[str, Any]]] = []
        word_offsets: list[int] = []
        relations_mod = _require("gliner2.models.boundary.relations", "relation specs")
        relation_spec_cls = relations_mod.RelationTypeSpec

        for i in range(len(batch)):
            layout = batch.query_layouts[i]
            specs_i = [
                {
                    "group_index": spec.task_index,
                    "field_index": spec.role_index,
                    "task_type": spec.task_type,
                    "task_name": spec.task_name,
                    "field_name": spec.role_name,
                }
                for spec in layout.queries
            ]
            ext_specs.append(specs_i)
            text_len_i = (
                len(batch.start_mappings[i])
                if batch.start_mappings
                else int(batch.text_word_counts[i])
            )
            word_offsets.append(max(int(batch.text_word_counts[i]) - text_len_i, 0))

            cls_i: list[dict[str, Any]] = []
            cls_offset = 0
            for group_index in range(batch.schema_counts[i]):
                if batch.task_types[i][group_index] != "classifications":
                    continue
                choice_count = max(len(batch.schema_special_indices[i][group_index]) - 1, 0)
                if choice_count:
                    schema_tokens = batch.schema_tokens_list[i][group_index]
                    cls_i.append(
                        {
                            "group_index": group_index,
                            "task_name": schema_tokens[2] if len(schema_tokens) > 2 else "",
                            "schema_tokens": schema_tokens,
                            "choice_states": cls_states[i, cls_offset : cls_offset + choice_count],
                            "group_embs": torch.cat(
                                (
                                    cls_states.new_zeros((1, h)),
                                    cls_states[i, cls_offset : cls_offset + choice_count],
                                )
                            ),
                        }
                    )
                cls_offset += choice_count
            cls_specs.append(cls_i)

            rel_i: list[dict[str, Any]] = []
            groups: dict[int, list[int]] = {}
            for query_id, spec in enumerate(specs_i):
                groups.setdefault(spec["group_index"], []).append(query_id)
            for group_index, role_ids_list in groups.items():
                if batch.task_types[i][group_index] != "relations":
                    continue
                role_ids = tuple(role_ids_list)
                if len(role_ids) < 2:
                    continue
                head_id, tail_id = role_ids[:2]
                max_q = query_states.shape[1] - 1
                head_id = min(head_id, max_q)
                tail_id = min(tail_id, max_q)
                role_states = query_states[i, [head_id, tail_id]]
                relation_state = (
                    torch.cat((role_states[0], role_states[1]), dim=-1)
                    if self.boundary_settings.directional_relation_states
                    else role_states.mean(dim=0)
                )
                rel_i.append(
                    {
                        "group_index": group_index,
                        "relation_type": specs_i[head_id]["task_name"],
                        "spec": relation_spec_cls(
                            specs_i[head_id]["task_name"],
                            head_query_ids=(head_id,),
                            tail_query_ids=(tail_id,),
                        ),
                        "query_state": relation_state,
                    }
                )
            rel_specs.append(rel_i)

        return {
            "text_states": text_states,
            "text_mask": text_mask,
            "text_lengths": text_mask.sum(-1).long(),
            "query_states": query_states,
            "query_mask": query_mask,
            "ext_specs": ext_specs,
            "cls_specs": cls_specs,
            "rel_specs": rel_specs,
            "word_offsets": word_offsets,
        }


def _as_tensor(value: Any, device: torch.device, dtype: torch.dtype | None = None) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.to(device)
    else:
        tensor = torch.tensor(value, device=device)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor


def _batch_from_extra(extra: dict[str, Any], device: torch.device) -> SimpleNamespace:
    batch = SimpleNamespace()
    batch.input_ids = _as_tensor(extra["input_ids"], device, torch.long).unsqueeze(0)
    batch.text_word_indices = _as_tensor(extra["text_word_indices"], device, torch.long).unsqueeze(0)
    batch.text_word_mask = _as_tensor(extra["text_word_mask"], device, torch.bool).unsqueeze(0)
    batch.query_marker_indices = _as_tensor(
        extra["query_marker_indices"], device, torch.long
    ).unsqueeze(0)
    batch.query_marker_mask = _as_tensor(
        extra["query_marker_mask"], device, torch.bool
    ).unsqueeze(0)
    batch.cls_marker_indices = _as_tensor(
        extra["cls_marker_indices"], device, torch.long
    ).unsqueeze(0)
    batch.cls_marker_mask = _as_tensor(extra["cls_marker_mask"], device, torch.bool).unsqueeze(0)
    batch.start_mappings = extra.get("start_mappings") or [[]]
    batch.end_mappings = extra.get("end_mappings") or [[]]
    batch.original_texts = extra.get("original_texts") or [""]
    batch.original_schemas = extra.get("original_schemas") or [{}]
    batch.task_types = extra.get("task_types") or [[]]
    batch.schema_counts = extra.get("schema_counts") or [0]
    batch.schema_tokens_list = extra.get("schema_tokens_list") or [[]]
    batch.schema_special_indices = extra.get("schema_special_indices") or [[]]
    batch.text_word_counts = extra.get("text_word_counts") or [0]
    batch.record_specs = extra.get("record_specs") or ()
    batch.query_layouts = [_layout_from_payload(item) for item in extra.get("query_layouts") or [{}]]
    batch.n = 1
    batch.__len__ = lambda: 1  # type: ignore[method-assign]
    return _LenOne(batch)


class _LenOne:
    def __init__(self, inner: SimpleNamespace) -> None:
        self.__dict__.update(inner.__dict__)

    def __len__(self) -> int:
        return 1

    def to(self, device: torch.device) -> _LenOne:
        return self


def _layout_from_payload(payload: Any) -> SimpleNamespace:
    queries = []
    for spec in payload.get("queries") if isinstance(payload, dict) else getattr(payload, "queries", ()):
        if isinstance(spec, dict):
            queries.append(SimpleNamespace(**spec))
        else:
            queries.append(spec)
    return SimpleNamespace(queries=queries)


def _pack_json(sample: dict[str, Any], device: torch.device) -> torch.Tensor:
    payload = json.dumps(sample, default=str).encode("utf-8")
    values = [float(len(payload)), *[float(byte) for byte in payload]]
    return torch.tensor(values, device=device, dtype=torch.float32)
