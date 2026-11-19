# -*- coding: utf-8 -*-
"""Evaluation data for the main run (v1.0).

- build_dev_set(): the frozen 288-sentence development set (v0.4; 8 templates x 2 verb
  expressions per behaviour). Rerun only as a continuity check.
- build_eval_templates(): held-out template sentences. Connectives and verb expressions
  (three per behaviour) do not occur in the development set or in the prompt examples.
  THEN: 4 connectives in sentence-initial / sentence-internal position -> 7 patterns,
        4 of which keep mention order = chronological order; 12 ordered event pairs x 9 verb combinations.
  WHILE: 3 patterns x 4 ordered event pairs x 9 verb combinations = 108 sentences.
- load_freeform(): hand-written free-form sentences (CSV) with gold annotations.
- Clusters: template sentences that share a template and gold event order (differing only in
  verb expressions) form one cluster; free-form sentences are clustered by gold TQP.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pandas as pd

from tqp_teg import EVENT_TYPES, Query, parse_tqp
import tqp_nl as N

# ---------------------------------------------------------------------------
# Development set (frozen v0.4)
# ---------------------------------------------------------------------------
DEV_PHRASES = {
    "DECELERATE": [("slows down", "slowing down"), ("reduces its speed", "reducing its speed")],
    "TURN": [("turns", "turning"), ("takes a turn", "taking a turn")],
    "ACCELERATE": [("speeds up", "speeding up"), ("accelerates", "accelerating")],
    "STOP": [("stops", "stopping"), ("comes to a stop", "coming to a stop")],
}


def _gold(a: str, b: str, relation: str, order) -> dict:
    slot = {"A": "e1", "B": "e2"}
    return {"events": {"e1": a, "e2": b}, "relation": {"op": relation, "args": [slot[order[0]], slot[order[1]]]}}


def _row(split, tid, connective, cond, sentence, plan, verbs, position):
    q = parse_tqp(plan)
    return {"split": split, "template": tid, "connective": connective, "position": position,
            "order_type": cond, "congruent": cond == "congruent", "sentence": sentence,
            "gold_tqp": json.dumps(plan), "first_mentioned": plan["events"]["e1"],
            "second_mentioned": plan["events"]["e2"],
            "gold_first": q.first, "gold_relation": q.relation, "gold_second": q.second,
            "verbs": verbs, "pair": "+".join(sorted((q.first, q.second))),
            "cluster": f"{split}:{tid}:{q.first}>{q.second}" if q.relation == "THEN"
            else f"{split}:{tid}:{plan['events']['e1']}~{plan['events']['e2']}"}


def build_dev_set() -> pd.DataFrame:
    rows = []

    def fill(pattern, a, b, va, vb):
        A, B = DEV_PHRASES[a][va], DEV_PHRASES[b][vb]
        return pattern.format(A_fin=A[0], A_ger=A[1], B_fin=B[0], B_ger=B[1])

    for tid, pattern, order, cond in N.THEN_TEMPLATES:
        pos = "initial" if tid.endswith("fronted") else "internal"
        for a in EVENT_TYPES:
            for b in EVENT_TYPES:
                if a != b:
                    for va in (0, 1):
                        for vb in (0, 1):
                            rows.append(_row("dev", tid, N.CONNECTIVE[tid], cond, fill(pattern, a, b, va, vb),
                                             _gold(a, b, "THEN", order), f"{va}{vb}", pos))
    for tid, pattern in N.WHILE_TEMPLATES:
        pos = "initial" if tid.endswith("fronted") else "internal"
        for a, b in N.WHILE_PAIRS:
            for va in (0, 1):
                for vb in (0, 1):
                    rows.append(_row("dev", tid, N.CONNECTIVE[tid], "simultaneous", fill(pattern, a, b, va, vb),
                                     _gold(a, b, "WHILE", ("A", "B")), f"{va}{vb}", pos))
    df = pd.DataFrame(rows)
    assert len(df) == 288, len(df)
    return df


# ---------------------------------------------------------------------------
# Held-out evaluation templates
# ---------------------------------------------------------------------------
# (finite, gerund, past participle)
EVAL_PHRASES = {
    "DECELERATE": [("loses speed", "losing speed", "lost speed"),
                   ("eases off", "easing off", "eased off"),
                   ("cuts its speed", "cutting its speed", "cut its speed")],
    "ACCELERATE": [("gains speed", "gaining speed", "gained speed"),
                   ("builds up speed", "building up speed", "built up speed"),
                   ("increases its speed", "increasing its speed", "increased its speed")],
    "STOP": [("comes to rest", "coming to rest", "come to rest"),
             ("comes to a standstill", "coming to a standstill", "come to a standstill"),
             ("becomes stationary", "becoming stationary", "become stationary")],
    "TURN": [("changes direction", "changing direction", "changed direction"),
             ("rounds a corner", "rounding a corner", "rounded a corner"),
             ("swings left", "swinging left", "swung left")],
}
# A = first-mentioned event, B = second-mentioned. order = temporal order of (A, B).
EVAL_THEN_TEMPLATES = [
    ("prior_to", "prior to", "internal", "A vehicle {A_fin} prior to {B_ger}.", ("A", "B"), "congruent"),
    ("prior_to_fronted", "prior to", "initial", "Prior to {A_ger}, a vehicle {B_fin}.", ("B", "A"), "incongruent"),
    ("once_fronted", "once", "initial", "Once a vehicle has {A_pp}, it {B_fin}.", ("A", "B"), "congruent"),
    ("once", "once", "internal", "A vehicle {A_fin} once it has {B_pp}.", ("B", "A"), "incongruent"),
    ("following_fronted", "following", "initial", "Following {A_ger}, a vehicle {B_fin}.", ("A", "B"), "congruent"),
    ("following", "following", "internal", "A vehicle {A_fin} following {B_ger}.", ("B", "A"), "incongruent"),
    ("then", "then", "internal", "A vehicle {A_fin}, then {B_fin}.", ("A", "B"), "congruent"),
]
EVAL_WHILE_TEMPLATES = [
    ("in_the_middle_of", "in the middle of", "internal", "A vehicle {A_fin} in the middle of {B_ger}."),
    ("in_the_middle_of_fronted", "in the middle of", "initial", "In the middle of {A_ger}, a vehicle {B_fin}."),
    ("at_the_same_time_as", "at the same time as", "internal", "A vehicle {A_fin} at the same time as it {B_fin}."),
]
EVAL_WHILE_PAIRS = N.WHILE_PAIRS  # (DECEL, TURN), (TURN, DECEL), (ACCEL, TURN), (TURN, ACCEL)

# Expressions that must not appear in held-out sentences (development set + prompt examples).
DEV_CONNECTIVE_PATTERNS = [r"\bbefore\b", r"\bafter\b", r"\band then\b", r"\bwhile\b", r"\bwhilst\b",
                           r"(?<!same time )\bas\b(?! soon)"]
DEV_VERB_PATTERNS = [r"\bslow(s|ing)? down\b", r"\breduc(es|ing) its speed\b", r"\bturn(s|ing|ed)?\b",
                     r"\btak(es|ing) a turn\b", r"\bspeed(s|ing)? up\b", r"\baccelerat(es|ing|ed)\b",
                     r"\bstop(s|ping|ped)?\b", r"\bcom(es|ing) to a stop\b"]


def _fill_eval(pattern, a, b, va, vb):
    A, B = EVAL_PHRASES[a][va], EVAL_PHRASES[b][vb]
    return pattern.format(A_fin=A[0], A_ger=A[1], A_pp=A[2], B_fin=B[0], B_ger=B[1], B_pp=B[2])


def build_eval_templates() -> pd.DataFrame:
    rows = []
    for tid, conn, pos, pattern, order, cond in EVAL_THEN_TEMPLATES:
        for a in EVENT_TYPES:
            for b in EVENT_TYPES:
                if a != b:
                    for va in range(3):
                        for vb in range(3):
                            rows.append(_row("eval_tpl", tid, conn, cond, _fill_eval(pattern, a, b, va, vb),
                                             _gold(a, b, "THEN", order), f"{va}{vb}", pos))
    for tid, conn, pos, pattern in EVAL_WHILE_TEMPLATES:
        for a, b in EVAL_WHILE_PAIRS:
            for va in range(3):
                for vb in range(3):
                    rows.append(_row("eval_tpl", tid, conn, "simultaneous", _fill_eval(pattern, a, b, va, vb),
                                     _gold(a, b, "WHILE", ("A", "B")), f"{va}{vb}", pos))
    df = pd.DataFrame(rows)
    n_then = sum(1 for t in EVAL_THEN_TEMPLATES if t[5] == "congruent")
    assert n_then == 4 and len(EVAL_THEN_TEMPLATES) == 7
    assert (df.gold_relation == "WHILE").sum() == 108
    return df


# ---------------------------------------------------------------------------
# Free-form sentences
# ---------------------------------------------------------------------------
FREEFORM_COLUMNS = ["id", "sentence", "first_mentioned", "second_mentioned", "relation", "gold_first",
                    "gold_second", "connective", "note"]


def load_freeform(path) -> pd.DataFrame:
    raw = pd.read_csv(path).fillna("")
    missing = [c for c in FREEFORM_COLUMNS if c not in raw.columns]
    assert not missing, f"free-form CSV lacks columns {missing}"
    rows = []
    for r in raw.itertuples():
        rel = r.relation.strip().upper()
        fm, sm = r.first_mentioned.strip().upper(), r.second_mentioned.strip().upper()
        gf, gs = r.gold_first.strip().upper(), r.gold_second.strip().upper()
        assert {fm, sm} == {gf, gs} and fm != sm, f"{r.id}: event types inconsistent"
        if rel == "THEN":
            order = ("A", "B") if gf == fm else ("B", "A")
            cond = "congruent" if order == ("A", "B") else "incongruent"
        else:
            order, cond = ("A", "B"), "simultaneous"
        plan = _gold(fm, sm, rel, order)
        q = parse_tqp(plan)
        assert (q.first, q.second) == (gf, gs) or rel == "WHILE"
        row = _row("eval_free", r.id, r.connective, cond, r.sentence.strip(), plan, "", "free")
        row["cluster"] = "eval_free:" + "_".join(Query(gf, rel, gs).semantic_key())
        row["note"] = r.note
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Checks and hashes
# ---------------------------------------------------------------------------
def leakage_report(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of held-out sentences containing development-set or prompt-example expressions."""
    examples = {s for v in N.VARIANTS.values() for s, _ in v["examples"]}
    bad = []
    for r in df.itertuples():
        s = r.sentence.lower()
        hits = [p for p in DEV_CONNECTIVE_PATTERNS + DEV_VERB_PATTERNS if re.search(p, s)]
        hits += [w for w in N.EXAMPLE_ONLY_WORDS if w in s]
        if r.sentence in examples:
            hits.append("example sentence")
        if hits:
            bad.append({"split": r.split, "sentence": r.sentence, "hits": ", ".join(hits)})
    return pd.DataFrame(bad, columns=["split", "sentence", "hits"])


def frame_hash(df: pd.DataFrame, cols=("sentence", "gold_tqp", "cluster")) -> str:
    h = hashlib.sha256()
    for r in df[list(cols)].itertuples(index=False):
        h.update(("\t".join(map(str, r)) + "\n").encode())
    return h.hexdigest()


def prompts_hash() -> str:
    blob = json.dumps({"variants": {k: VARIANTS_DUMP(k) for k in N.MAIN_CONDITIONS}, "b2": N.B2_SYSTEM},
                      sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def VARIANTS_DUMP(k):
    v = N.VARIANTS[k]
    return {"tqp_system": v["tqp_system"], "examples": v["examples"]}
