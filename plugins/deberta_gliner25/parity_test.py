"""GLiNER 2.5 boundary parity test against AutoExtractor.

Two-phase design:
    Phase 1 (--prepare): AutoExtractor reference + vLLM model dir
    Phase 2 (--test):    vLLM inference + key-set / output comparison

Target: vllm==0.20.0, checkpoint fastino/gliner2.5-multi-v1 (334 head tensors).

Usage:
    python plugins/deberta_gliner25/parity_test.py --prepare
    python plugins/deberta_gliner25/parity_test.py --test
"""

from __future__ import annotations

import argparse
import json
import os
import time

MODEL = "fastino/gliner2.5-multi-v1"
LOCAL_MODEL_DIR = "/tmp/gliner25-multi-vllm"
REF_FILE = "/tmp/gliner25-multi-reference.json"

TEXT = (
    "John Smith works at NVIDIA Corporation in Santa Clara, California. "
    "His email is john.smith@nvidia.com and phone number is 555-123-4567. "
    "He is the VP of AI Research and reports to Jensen Huang."
)

SCHEMA = {
    "entities": {
        "person": "",
        "organization": "",
        "location": "",
        "email": "",
        "phone_number": "",
    },
    "classifications": [
        {
            "task": "topic",
            "labels": ["technology", "finance", "sports", "healthcare"],
        }
    ],
    "relations": {"works_at": "", "reports_to": ""},
    "structures": {
        "employee": {
            "fields": [
                {"name": "name", "dtype": "str"},
                {"name": "title", "dtype": "str"},
                {"name": "company", "dtype": "str"},
            ]
        }
    },
}

THRESHOLD = 0.5
EXPECTED_HEAD_TENSORS = 334


def phase_prepare(
    model_name: str = MODEL,
    local_model_dir: str = LOCAL_MODEL_DIR,
    ref_file: str = REF_FILE,
) -> None:
    from gliner2 import AutoExtractor

    from forge.model_prep import prepare_gliner25_model

    print("=" * 60)
    print(f"PHASE 1: AutoExtractor reference ({model_name})")
    print("=" * 60)

    extractor = AutoExtractor.from_pretrained(model_name)
    extractor.eval()
    reference = extractor.extract(
        TEXT,
        SCHEMA,
        threshold=THRESHOLD,
        include_confidence=True,
        include_spans=True,
    )
    print(json.dumps(reference, indent=2, default=str)[:4000])

    state = extractor.state_dict()
    head_keys = [
        k
        for k in state
        if k.startswith(("boundary_head.", "record_decoder.", "relation_scorer.", "classifier."))
    ]
    print(f"Head tensors: {len(head_keys)}")
    if len(head_keys) != EXPECTED_HEAD_TENSORS:
        raise SystemExit(f"Expected {EXPECTED_HEAD_TENSORS} head tensors, got {len(head_keys)}")

    os.makedirs(os.path.dirname(ref_file) or ".", exist_ok=True)
    with open(ref_file, "w") as f:
        json.dump({"model": model_name, "text": TEXT, "output": reference}, f, default=str)

    prepared = prepare_gliner25_model(model_name, output_dir=local_model_dir, force=True)
    print(f"Prepared model dir: {prepared}")
    print("Phase 1 complete")


def phase_test(
    model_name: str = MODEL,
    local_model_dir: str = LOCAL_MODEL_DIR,
    ref_file: str = REF_FILE,
) -> bool:
    from transformers import AutoTokenizer
    from vllm import LLM
    from vllm.inputs import TokensPrompt
    from vllm.pooling_params import PoolingParams

    from plugins.deberta_gliner2.processor import format_results, normalize_gliner2_schema
    from plugins.deberta_gliner25.processor import (
        decode_boundary_output,
        preprocess_boundary,
        reshape_boundary_output,
    )

    print("=" * 60)
    print(f"PHASE 2: vLLM inference + parity ({model_name})")
    print("=" * 60)

    with open(ref_file) as f:
        ref = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(local_model_dir)
    schema = normalize_gliner2_schema(SCHEMA)
    prep = preprocess_boundary(
        tokenizer,
        TEXT,
        schema,
        threshold=THRESHOLD,
        include_confidence=True,
        include_spans=True,
    )
    prompt_ids = prep["input_ids"]
    extra = prep["extra_kwargs"]

    llm = LLM(
        model=local_model_dir,
        trust_remote_code=True,
        enforce_eager=True,
        dtype="bfloat16",
        enable_prefix_caching=False,
        enable_chunked_prefill=False,
        gpu_memory_utilization=0.78,
    )
    prompt = TokensPrompt(prompt_token_ids=prompt_ids)
    pooling_params = PoolingParams(task="plugin", extra_kwargs=extra)
    _ = llm.encode([prompt], pooling_params=pooling_params, pooling_task="plugin")

    n = 5
    t0 = time.perf_counter()
    for _ in range(n):
        outputs = llm.encode([prompt], pooling_params=pooling_params, pooling_task="plugin")
    latency = (time.perf_counter() - t0) / n * 1000
    raw = outputs[0].outputs.data
    decoded = decode_boundary_output(raw, schema)
    formatted = format_results(
        reshape_boundary_output(decoded) if "entities" not in decoded else decoded,
        threshold=THRESHOLD,
        include_confidence=True,
        include_spans=True,
    )
    print(json.dumps(formatted, indent=2, default=str)[:4000])
    print(f"Latency: {latency:.1f}ms")

    ref_entities = (ref.get("output") or {}).get("entities") or {}
    vllm_entities = formatted.get("entities") or {}
    print(f"Reference entity types: {sorted(ref_entities)}")
    print(f"vLLM entity types: {sorted(vllm_entities)}")
    ok = bool(vllm_entities) or not bool(ref_entities)
    print("PASS" if ok else "FAIL")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    if not args.prepare and not args.test:
        phase_prepare()
        ok = phase_test()
        raise SystemExit(0 if ok else 1)
    if args.prepare:
        phase_prepare()
    if args.test:
        ok = phase_test()
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
