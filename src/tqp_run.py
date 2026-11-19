# -*- coding: utf-8 -*-
"""Main-run orchestration: generation with resumable caching, scoring, retrieval."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import tqp_nl as N
import tqp_teg as T

MAX_TOKENS = {"P": 96, "B2": 384}


def cell_key(split: str, method: str, condition: str, sentences: Sequence[str], example_order=None) -> str:
    blob = json.dumps({"split": split, "method": method, "condition": condition, "sentences": list(sentences),
                       "messages0": N.prompt_messages(method, "<S>", condition, example_order),
                       "max_tokens": MAX_TOKENS[method], "order": example_order}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def generate_cell(model, tok, device, out_dir: Path, split: str, method: str, condition: str,
                  data: pd.DataFrame, batch_size: int = 16, example_order=None, tag: str = "") -> pd.DataFrame:
    """Generates raw outputs for one (split, method, condition) cell; reuses a cached CSV if the
    sentences and prompt are unchanged."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    key = cell_key(split, method, condition, data.sentence.tolist(), example_order)
    path = out_dir / f"raw_{split}_{method}_{condition}{tag}_{key}.csv"
    if path.exists():
        cached = pd.read_csv(path, keep_default_na=False)
        if len(cached) == len(data) and (cached.sentence.values == data.sentence.values).all():
            print(f"  cached: {path.name}")
            return cached
    msgs = [N.prompt_messages(method, s, condition, example_order) for s in data.sentence]
    outs, sec = N.generate(model, tok, device, msgs, batch_size=batch_size, max_new_tokens=MAX_TOKENS[method])
    df = pd.DataFrame({"sid": data.sid.values, "sentence": data.sentence.values, "raw_output": outs})
    df["gen_seconds_total"] = round(sec, 1)
    df.to_csv(path, index=False)
    print(f"  generated {split}/{method}/{condition}{tag}: {len(df)} in {sec:.0f}s -> {path.name}")
    return df


def score_cell(data: pd.DataFrame, raw: pd.DataFrame, method: str, condition: str) -> pd.DataFrame:
    rows = []
    cache: Dict[T.Query, list] = {}
    raw = raw.set_index("sid")
    for r in data.itertuples():
        gold = T.Query(r.gold_first, r.gold_relation, r.gold_second)
        probes = cache.setdefault(gold, N.build_probes_v05(gold))
        text = raw.loc[r.sid, "raw_output"]
        extra = {}
        if method == "P":
            pred, plan, err = N.parse_prediction(text)
            acc = N.accepts_p(pred, probes)
            # representation-specific diagnostics (P): read directly from the plan
            extra = {"pred": str(pred) if pred else None,
                     "p_events_correct": bool(pred and sorted((pred.first, pred.second)) == sorted((gold.first, gold.second))),
                     "p_relation_correct": bool(pred and pred.relation == gold.relation),
                     "p_direction_correct": (bool(pred and pred.relation == "THEN" and (pred.first, pred.second) == (gold.first, gold.second))
                                             if gold.relation == "THEN" else None),
                     "plan_json": json.dumps(plan) if isinstance(plan, dict) else None,
                     "h3_subtype": N.p_direction_subtype(plan, gold, r.first_mentioned, bool(r.congruent))
                     if isinstance(plan, dict) else None}
        else:
            acc, err = N.run_b2(text, probes)
            extra = {"pred": None, "plan_json": None, "h3_subtype": None}
        sc = N.score_v05(gold, acc, probes)
        rows.append({"sid": r.sid, "method": method, "condition": condition, "error": err,
                     "error_type": N.error_type(gold, sc["inferred"], sc["types_ok"], sc["valid"]),
                     "probe_pattern": "".join("1" if acc[n] else "0" for n in N.ALL_PROBES) if acc else None,
                     **sc, **extra, "raw_output": text})
    out = pd.DataFrame(rows)
    return data.merge(out, on="sid")


# ---------------------------------------------------------------------------
# Retrieval on real logs
# ---------------------------------------------------------------------------
def b2_compile(text: str):
    code = N.extract_code(text)
    tree = N.check_ast(code)
    ns = {"__builtins__": dict(N.SAFE_BUILTINS), "math": math}
    exec(compile(tree, "<b2>", "exec"), ns)
    return ns["matches"]


def b2_retrieve(text: str, track_dicts: Dict[tuple, List[dict]], empty_tracks: Sequence[tuple],
                timeout_s: float = 20.0) -> Optional[frozenset]:
    """Runs a B2 program on every track's event list (one time limit for the whole sweep)."""
    try:
        fn = b2_compile(text)
    except Exception:
        return None
    hits = []

    def sweep(_):
        for key, evs in track_dicts.items():
            if fn(copy.deepcopy(evs)):
                hits.append(key)
        if fn([]):
            hits.extend(empty_tracks)
        return True

    try:
        N._call_with_timeout(sweep, None, timeout_s)
    except Exception:
        return None
    return frozenset(hits)


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def retrieval_table(scored: pd.DataFrame, gexec, track_dicts, empty_tracks) -> pd.DataFrame:
    """Per (sentence, method, condition): retrieved (log_id, track_id) set vs the gold TQP's set."""
    gold_cache: Dict[T.Query, frozenset] = {}
    pred_cache: Dict[str, Optional[frozenset]] = {}

    def run_query(q: T.Query) -> frozenset:
        if q not in gold_cache:
            gold_cache[q] = frozenset((m.log_id, m.track_id) for m in gexec.retrieve(q))
        return gold_cache[q]

    rows = []
    for r in scored.itertuples():
        gold = T.Query(r.gold_first, r.gold_relation, r.gold_second)
        gset = run_query(gold)
        if r.method == "P":
            pq = None
            if isinstance(r.pred, str) and r.pred:
                a, rel, b = r.pred.split()
                pq = T.Query(a, rel, b)
            pset = run_query(pq) if pq else None
        else:
            k = r.raw_output
            if k not in pred_cache:
                pred_cache[k] = b2_retrieve(k, track_dicts, empty_tracks)
            pset = pred_cache[k]
        rows.append({"sid": r.sid, "method": r.method, "condition": r.condition, "gold_n": len(gset),
                     "pred_n": None if pset is None else len(pset),
                     "same_set": None if pset is None else pset == gset,
                     "jaccard": 0.0 if pset is None else jaccard(pset, gset)})
    return pd.DataFrame(rows)
