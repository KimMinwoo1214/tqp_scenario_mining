# -*- coding: utf-8 -*-
"""Temporal Relation Grounding: TQP (P) vs direct Python generation (B2) with a small LLM.

- build_diagnostic_set(): 72 templated sentences with gold TQPs
  (before / after / and-then / while / as; surface-order congruent, incongruent, simultaneous).
- Prompt variants (same few-shot sentences for P and B2 so the comparison is paired):
    v1_original : the v0.2 prompt (rule: e1 = first-mentioned event), 3 examples
    zero_shot   : revised rules, no examples
    balanced    : revised rules, 4 examples covering before / after / fronted Before / while
                  written with synonym verbs that never appear in the evaluation set
- P  : model writes a TQP JSON; validated by the strict schema, never executed as code.
- B2 : model writes `def matches(events) -> bool`; checked by AST rules and run in a
       restricted namespace with a per-call time limit on behavioural probe tracks.
- Both are scored on the same temporal interpretation (relation, direction, exact) and
  on the same probe tracks (probe_exact).
"""
from __future__ import annotations

import ast
import copy
import json
import math
import re
import signal
import time
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from tqp_teg import EVENT_TYPES, Event, Query, pair_status, parse_tqp, possible_relations

# ---------------------------------------------------------------------------
# Diagnostic sentences
# ---------------------------------------------------------------------------
PHRASES = {
    "DECELERATE": {"fin": "slows down", "ger": "slowing down"},
    "TURN": {"fin": "turns", "ger": "turning"},
    "ACCELERATE": {"fin": "speeds up", "ger": "speeding up"},
    "STOP": {"fin": "stops", "ger": "stopping"},
}
# A = first-mentioned event, B = second-mentioned event.
THEN_TEMPLATES = [
    ("before", "A vehicle {A_fin} before {B_ger}.", ("A", "B"), "congruent"),
    ("after", "A vehicle {A_fin} after {B_ger}.", ("B", "A"), "incongruent"),
    ("before_fronted", "Before {A_ger}, a vehicle {B_fin}.", ("B", "A"), "incongruent"),
    ("after_fronted", "After {A_ger}, a vehicle {B_fin}.", ("A", "B"), "congruent"),
    ("and_then", "A vehicle {A_fin} and then {B_fin}.", ("A", "B"), "congruent"),
]
WHILE_TEMPLATES = [
    ("while", "A vehicle {A_fin} while {B_ger}."),
    ("while_fronted", "While {A_ger}, a vehicle {B_fin}."),
    ("as", "A vehicle {A_fin} as it {B_fin}."),
]
WHILE_PAIRS = [("DECELERATE", "TURN"), ("TURN", "DECELERATE"),
               ("ACCELERATE", "TURN"), ("TURN", "ACCELERATE")]
CONNECTIVE = {"before": "before", "before_fronted": "before", "after": "after",
              "after_fronted": "after", "and_then": "and then", "while": "while",
              "while_fronted": "while", "as": "as"}


def _fill(pattern: str, a: str, b: str) -> str:
    return pattern.format(A_fin=PHRASES[a]["fin"], A_ger=PHRASES[a]["ger"],
                          B_fin=PHRASES[b]["fin"], B_ger=PHRASES[b]["ger"])


def gold_tqp(a: str, b: str, relation: str, order: Tuple[str, str]) -> dict:
    slot = {"A": "e1", "B": "e2"}
    return {"events": {"e1": a, "e2": b},
            "relation": {"op": relation, "args": [slot[order[0]], slot[order[1]]]}}


def build_diagnostic_set() -> pd.DataFrame:
    rows = []
    for tid, pattern, order, cond in THEN_TEMPLATES:
        for a in EVENT_TYPES:
            for b in EVENT_TYPES:
                if a != b:
                    rows.append((tid, cond, _fill(pattern, a, b), gold_tqp(a, b, "THEN", order)))
    for tid, pattern in WHILE_TEMPLATES:
        for a, b in WHILE_PAIRS:
            rows.append((tid, "simultaneous", _fill(pattern, a, b), gold_tqp(a, b, "WHILE", ("A", "B"))))
    out = []
    for tid, cond, sentence, plan in rows:
        q = parse_tqp(plan)
        out.append({"template": tid, "connective": CONNECTIVE[tid], "condition": cond,
                    "sentence": sentence, "gold_tqp": json.dumps(plan), "first_mentioned": plan["events"]["e1"],
                    "gold_first": q.first, "gold_relation": q.relation, "gold_second": q.second})
    df = pd.DataFrame(out)
    leaked = {s for v in VARIANTS.values() for s, _ in v["examples"]} & set(df.sentence)
    assert not leaked, f"few-shot sentences leaked into evaluation set: {leaked}"
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------
TQP_SYSTEM_V1 = """You convert a sentence describing a vehicle's behaviour into a Temporal Query Plan (TQP) in JSON.

Event types (use these exact strings):
- TURN: the vehicle turns / changes heading
- DECELERATE: the vehicle slows down / brakes
- ACCELERATE: the vehicle speeds up
- STOP: the vehicle stops / comes to a stop

Rules:
1. "e1" is the event mentioned FIRST in the sentence, "e2" the event mentioned SECOND.
2. relation "op" is "THEN" if one event happens after the other, "WHILE" if they happen at the same time.
3. relation "args" lists the slots in the order the events ACTUALLY HAPPEN in time (earlier event first). For WHILE use ["e1", "e2"].
4. Output only the JSON object and nothing else."""

TQP_SYSTEM_V2 = """You convert a sentence describing a vehicle's behaviour into a Temporal Query Plan (TQP) in JSON.

Event types (use these exact strings):
- TURN: the vehicle turns / changes heading
- DECELERATE: the vehicle slows down / brakes
- ACCELERATE: the vehicle speeds up
- STOP: the vehicle stops / comes to a stop

Format: {"events": {"e1": TYPE, "e2": TYPE}, "relation": {"op": "THEN" or "WHILE", "args": [slot, slot]}}

Rules:
1. Put the two events of the sentence in the slots "e1" and "e2".
2. "op" is "THEN" if one event happens after the other, "WHILE" if they happen at the same time.
3. "args" lists the slots in the order the events ACTUALLY HAPPEN in time (earlier event first). For WHILE the order does not matter.
4. Output only the JSON object and nothing else."""

B2_SYSTEM = """You write a Python function that decides whether one vehicle's detected events match a sentence.

def matches(events) -> bool
- events: a list of dicts {"type": str, "start": float, "end": float} for ONE vehicle; times are in seconds.
- "type" is one of TURN, DECELERATE (slows down / brakes), ACCELERATE (speeds up), STOP.

Temporal definitions (implement exactly):
- overlap(A, B) = max(0, min(A.end, B.end) - max(A.start, B.start))
- "A then B" (A happens before B): A.start < B.start and overlap(A, B) <= 0.5 and -0.5 <= B.start - A.end <= 5.0
- "A while B" (same time): overlap(A, B) > 0.5 and overlap(A, B) >= 0.8 * min(duration(A), duration(B))

Return True if some pair of events satisfies the sentence, otherwise False.
Do not import anything. Output only the code inside a ```python block."""

FEW_SHOT_V1 = [
    ("A vehicle turns, and afterwards it speeds up.",
     {"events": {"e1": "TURN", "e2": "ACCELERATE"}, "relation": {"op": "THEN", "args": ["e1", "e2"]}}),
    ("A vehicle comes to a stop once it has finished slowing down.",
     {"events": {"e1": "STOP", "e2": "DECELERATE"}, "relation": {"op": "THEN", "args": ["e2", "e1"]}}),
    ("A vehicle slows down during its turn.",
     {"events": {"e1": "DECELERATE", "e2": "TURN"}, "relation": {"op": "WHILE", "args": ["e1", "e2"]}}),
]
FEW_SHOT_BALANCED = [
    ("A vehicle brakes before making a turn.",
     {"events": {"e1": "DECELERATE", "e2": "TURN"}, "relation": {"op": "THEN", "args": ["e1", "e2"]}}),
    ("A vehicle comes to a halt after picking up speed.",
     {"events": {"e1": "STOP", "e2": "ACCELERATE"}, "relation": {"op": "THEN", "args": ["e2", "e1"]}}),
    ("Before coming to a halt, a vehicle makes a turn.",
     {"events": {"e1": "STOP", "e2": "TURN"}, "relation": {"op": "THEN", "args": ["e2", "e1"]}}),
    ("A vehicle picks up speed while making a turn.",
     {"events": {"e1": "ACCELERATE", "e2": "TURN"}, "relation": {"op": "WHILE", "args": ["e1", "e2"]}}),
]
VARIANTS = {
    "v1_original": {"tqp_system": TQP_SYSTEM_V1, "examples": FEW_SHOT_V1},
    "zero_shot": {"tqp_system": TQP_SYSTEM_V2, "examples": []},
    "balanced": {"tqp_system": TQP_SYSTEM_V2, "examples": FEW_SHOT_BALANCED},
}


def b2_reference_code(q: Query) -> str:
    """Correct B2 answer for a few-shot example (same semantics as the executor)."""
    if q.relation == "THEN":
        cond = "a['start'] < b['start'] and overlap <= 0.5 and -0.5 <= b['start'] - a['end'] <= 5.0"
    else:
        cond = "overlap > 0.5 and overlap >= 0.8 * min(a['end'] - a['start'], b['end'] - b['start'])"
    return ("```python\n"
            "def matches(events):\n"
            "    for a in events:\n"
            "        for b in events:\n"
            f"            if a is b or a['type'] != '{q.first}' or b['type'] != '{q.second}':\n"
            "                continue\n"
            "            overlap = max(0.0, min(a['end'], b['end']) - max(a['start'], b['start']))\n"
            f"            if {cond}:\n"
            "                return True\n"
            "    return False\n"
            "```")


def tqp_messages(sentence: str, variant: str) -> List[dict]:
    v = VARIANTS[variant]
    msgs = [{"role": "system", "content": v["tqp_system"]}]
    for s, plan in v["examples"]:
        msgs += [{"role": "user", "content": s}, {"role": "assistant", "content": json.dumps(plan)}]
    return msgs + [{"role": "user", "content": sentence}]


def b2_messages(sentence: str, variant: str) -> List[dict]:
    v = VARIANTS[variant]
    msgs = [{"role": "system", "content": B2_SYSTEM}]
    for s, plan in v["examples"]:
        msgs += [{"role": "user", "content": s},
                 {"role": "assistant", "content": b2_reference_code(parse_tqp(plan))}]
    return msgs + [{"role": "user", "content": sentence}]


# ---------------------------------------------------------------------------
# P: TQP parsing and scoring
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict:
    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object in output")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start:i + 1])
    raise ValueError("unbalanced JSON object in output")


def parse_prediction(text: str) -> Tuple[Optional[Query], Optional[dict], Optional[str]]:
    try:
        plan = extract_json(text)
    except Exception as exc:
        return None, None, f"json: {exc}"
    try:
        return parse_tqp(plan), plan, None
    except Exception as exc:
        return None, plan, f"schema: {exc}"


def interpretation_score(gold: Query, rel: Optional[str], first: Optional[str],
                         second: Optional[str], events_ok: Optional[bool]) -> dict:
    """Shared scoring of a temporal interpretation (relation op + ordered event types)."""
    if rel is None:
        return {"relation_correct": False, "events_correct": bool(events_ok) if events_ok is not None else False,
                "direction_correct": None if gold.relation == "WHILE" else False, "exact": False}
    rel_ok = rel == gold.relation
    direction = None
    if gold.relation == "THEN":
        direction = bool(rel_ok and first == gold.first and second == gold.second)
    if gold.relation == "WHILE":
        exact = rel_ok and sorted((first, second)) == sorted((gold.first, gold.second))
    else:
        exact = bool(direction)
    return {"relation_correct": rel_ok, "events_correct": bool(events_ok),
            "direction_correct": direction, "exact": bool(exact)}


def score_tqp(gold: Query, pred: Optional[Query]) -> dict:
    if pred is None:
        return {"valid": False, **interpretation_score(gold, None, None, None, False)}
    events_ok = sorted((pred.first, pred.second)) == sorted((gold.first, gold.second))
    return {"valid": True, **interpretation_score(gold, pred.relation, pred.first, pred.second, events_ok)}


# ---------------------------------------------------------------------------
# Behavioural probes (shared by P and B2)
# ---------------------------------------------------------------------------
PROBE_NAMES = ("then_xy", "then_yx", "while", "far", "only_x", "only_y", "other")


def build_probes(gold: Query) -> List[Tuple[str, List[Event], List[dict], bool]]:
    """Canonical probe tracks for the gold event types X=gold.first, Y=gold.second."""
    x, y = gold.first, gold.second
    other = next(t for t in EVENT_TYPES if t not in (x, y))
    specs = {
        "then_xy": [(x, 1, 3), (y, 5, 7)],
        "then_yx": [(y, 1, 3), (x, 5, 7)],
        "while": [(x, 2, 6), (y, 2.5, 5.5)],
        "far": [(x, 1, 2), (y, 10, 12)],
        "only_x": [(x, 1, 3)],
        "only_y": [(y, 5, 7)],
        "other": [(other, 1, 3), (other, 5, 7)],
    }
    probes = []
    for name in PROBE_NAMES:
        evs = [Event(f"{name}:{i}", "probe", name, t, float(s), float(e)) for i, (t, s, e) in enumerate(specs[name])]
        dicts = [{"type": e.type, "start": e.start, "end": e.end} for e in evs]
        probes.append((name, evs, dicts, query_accepts(gold, evs)))
    return probes


def query_accepts(q: Query, evs: Sequence[Event]) -> bool:
    return any(pair_status(possible_relations(a, b), q.relation) == "POSITIVE"
               for a in evs for b in evs
               if a.event_id != b.event_id and a.type == q.first and b.type == q.second)


def infer_from_accepts(acc: Dict[str, bool]) -> str:
    key = (acc["then_xy"], acc["then_yx"], acc["while"])
    return {(True, False, False): "THEN_XY", (False, True, False): "THEN_YX",
            (False, False, True): "WHILE"}.get(key, "OTHER")


def probe_score(gold: Query, accepts: Optional[Dict[str, bool]], probes) -> dict:
    if accepts is None:
        return {"probe_exact": False, "probe_pattern": None}
    pattern = "".join("1" if accepts[n] else "0" for n in PROBE_NAMES)
    return {"probe_exact": all(accepts[n] == g for n, _, _, g in probes), "probe_pattern": pattern}


def tqp_probe_accepts(pred: Optional[Query], probes) -> Optional[Dict[str, bool]]:
    if pred is None:
        return None
    return {name: query_accepts(pred, evs) for name, evs, _, _ in probes}


# ---------------------------------------------------------------------------
# B2: code extraction, sandbox, scoring
# ---------------------------------------------------------------------------
import builtins as _builtins

SAFE_BUILTINS = {n: getattr(_builtins, n)
                 for n in ("abs", "min", "max", "len", "range", "enumerate", "zip", "sorted", "sum",
                           "any", "all", "float", "int", "bool", "list", "dict", "set", "tuple",
                           "isinstance", "round", "reversed", "map", "filter", "str",
                           "Exception", "ValueError", "KeyError", "TypeError")}
FORBIDDEN_NAMES = {"eval", "exec", "open", "compile", "__import__", "globals", "locals", "vars",
                   "getattr", "setattr", "delattr", "input", "exit", "quit", "help", "breakpoint",
                   "memoryview", "type", "object", "super", "classmethod", "staticmethod", "property"}


def extract_code(text: str) -> str:
    m = re.search(r"```(?:python|py)?[^\n]*\n(.*?)```", text, re.S)
    if m:
        code = m.group(1)
    else:
        m2 = re.search(r"```(?:python|py)?[^\n]*\n(.*)", text, re.S)   # unterminated fence
        code = m2.group(1) if m2 else text
        i = code.find("def ")
        code = code[i:] if i >= 0 else code
    if "def matches" not in code:
        raise ValueError("no `def matches` in output")
    return code


def check_ast(code: str) -> ast.Module:
    tree = ast.parse(code)
    banned_nodes = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith,
                    ast.AsyncFunctionDef, ast.Await, ast.ClassDef, ast.Delete)
    for node in ast.walk(tree):
        if isinstance(node, banned_nodes):
            raise ValueError(f"disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            raise ValueError(f"disallowed name: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(f"disallowed attribute: {node.attr}")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "matches" for n in tree.body):
        raise ValueError("no top-level function `matches`")
    return tree


class _Timeout(Exception):
    pass


def _call_with_timeout(fn, arg, timeout_s: float):
    if not hasattr(signal, "setitimer"):
        return fn(arg)

    def handler(signum, frame):
        raise _Timeout("time limit exceeded")

    old = signal.signal(signal.SIGALRM, handler)
    signal.setitimer(signal.ITIMER_REAL, timeout_s)
    try:
        return fn(arg)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def run_b2(text: str, probes, timeout_s: float = 1.0) -> Tuple[Optional[Dict[str, bool]], Optional[str]]:
    try:
        code = extract_code(text)
        tree = check_ast(code)
        ns = {"__builtins__": dict(SAFE_BUILTINS), "math": math}
        exec(compile(tree, "<b2>", "exec"), ns)
        fn = ns["matches"]
    except Exception as exc:
        return None, f"compile: {type(exc).__name__}: {exc}"
    accepts = {}
    for name, _, dicts, _ in probes:
        try:
            accepts[name] = bool(_call_with_timeout(fn, copy.deepcopy(dicts), timeout_s))
        except Exception as exc:
            return None, f"runtime({name}): {type(exc).__name__}: {exc}"
    return accepts, None


def score_b2(gold: Query, accepts: Optional[Dict[str, bool]]) -> dict:
    if accepts is None:
        return {"valid": False, **interpretation_score(gold, None, None, None, False)}
    inferred = infer_from_accepts(accepts)
    events_ok = not (accepts["only_x"] or accepts["only_y"] or accepts["other"])
    if inferred == "THEN_XY":
        rel, first, second = "THEN", gold.first, gold.second
    elif inferred == "THEN_YX":
        rel, first, second = "THEN", gold.second, gold.first
    elif inferred == "WHILE":
        rel, first, second = "WHILE", gold.first, gold.second
    else:
        rel, first, second = None, None, None
    out = interpretation_score(gold, rel, first, second, events_ok)
    if rel is None:
        out["relation_correct"] = False
    return {"valid": True, **out}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def mcnemar_exact(a_correct: Sequence[bool], b_correct: Sequence[bool]) -> dict:
    from scipy.stats import binomtest
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"P_only_correct": b, "B2_only_correct": c, "p_value": round(float(p), 4)}


def summarize(df: pd.DataFrame, by) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        d = g.direction_correct.dropna().astype(float)
        k = int(g.exact.sum())
        lo, hi = wilson(k, len(g))
        return pd.Series({"n": len(g), "valid": round(g.valid.mean(), 3),
                          "relation_acc": round(g.relation_correct.mean(), 3),
                          "direction_acc(THEN)": round(d.mean(), 3) if len(d) else float("nan"),
                          "exact_acc": round(k / len(g), 3), "exact_95CI": f"[{lo:.2f}, {hi:.2f}]",
                          "probe_exact": round(g.probe_exact.mean(), 3)})
    return df.groupby(by).apply(agg)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def load_model(model_id: str = "Qwen/Qwen2.5-1.5B-Instruct"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cuda = torch.cuda.is_available()
    dtype = torch.float16 if cuda else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    device = "cuda" if cuda else "cpu"
    return model.to(device).eval(), tok, device


def generate(model, tok, device: str, message_lists: Sequence[List[dict]], *,
             batch_size: int = 16, max_new_tokens: int = 96) -> Tuple[List[str], float]:
    """Greedy decoding. Returns outputs and total seconds."""
    import torch
    prompts = [tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in message_lists]
    outputs: List[str] = []
    t0 = time.perf_counter()
    for i in range(0, len(prompts), batch_size):
        enc = tok(prompts[i:i + batch_size], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id)
        outputs.extend(tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
    return outputs, time.perf_counter() - t0

# ===========================================================================
# Main run (v1.0): frozen prompt conditions, probe set v0.5, scoring, error types
# ===========================================================================
# Counterbalanced: deceleration-turn and acceleration-turn appear with BOTH THEN and WHILE.
FEW_SHOT_COUNTERBALANCED = [
    ("A vehicle brakes while making a turn.",
     {"events": {"e1": "DECELERATE", "e2": "TURN"}, "relation": {"op": "WHILE", "args": ["e1", "e2"]}}),
    ("A vehicle makes a turn after braking.",
     {"events": {"e1": "TURN", "e2": "DECELERATE"}, "relation": {"op": "THEN", "args": ["e2", "e1"]}}),
    ("Before making a turn, a vehicle picks up speed.",
     {"events": {"e1": "TURN", "e2": "ACCELERATE"}, "relation": {"op": "THEN", "args": ["e2", "e1"]}}),
    ("A vehicle picks up speed while making a turn.",
     {"events": {"e1": "ACCELERATE", "e2": "TURN"}, "relation": {"op": "WHILE", "args": ["e1", "e2"]}}),
    ("A vehicle comes to a halt and then picks up speed.",
     {"events": {"e1": "STOP", "e2": "ACCELERATE"}, "relation": {"op": "THEN", "args": ["e1", "e2"]}}),
    ("A vehicle brakes before coming to a halt.",
     {"events": {"e1": "DECELERATE", "e2": "STOP"}, "relation": {"op": "THEN", "args": ["e1", "e2"]}}),
]
VARIANTS["counterbalanced"] = {"tqp_system": TQP_SYSTEM_V2, "examples": FEW_SHOT_COUNTERBALANCED}
MAIN_CONDITIONS = ("zero_shot", "balanced", "counterbalanced")
METHODS = ("P", "B2")
# Words that appear only in the prompt examples; they must not occur in evaluation sentences.
EXAMPLE_ONLY_WORDS = ("brak", "make a turn", "makes a turn", "making a turn", "picks up speed",
                      "picking up speed", "pick up speed", "halt")


def prompt_messages(method: str, sentence: str, condition: str, example_order=None) -> List[dict]:
    """example_order: optional permutation of the condition's examples (shuffled-order runs)."""
    v = VARIANTS[condition]
    examples = v["examples"] if example_order is None else [v["examples"][i] for i in example_order]
    if method == "P":
        msgs = [{"role": "system", "content": v["tqp_system"]}]
        for s, plan in examples:
            msgs += [{"role": "user", "content": s}, {"role": "assistant", "content": json.dumps(plan)}]
    else:
        msgs = [{"role": "system", "content": B2_SYSTEM}]
        for s, plan in examples:
            msgs += [{"role": "user", "content": s},
                     {"role": "assistant", "content": b2_reference_code(parse_tqp(plan))}]
    return msgs + [{"role": "user", "content": sentence}]


# ---- Probe set v0.5: 5 interpretation probes (far from thresholds) + 8 implementation probes
INTERP_PROBES = ("then_xy", "then_yx", "while_same", "x_other", "other_y")
IMPL_PROBES = ("far", "then_gap45", "then_ovl03", "while_unequal", "partial", "only_x", "only_y", "other")
ALL_PROBES = INTERP_PROBES + IMPL_PROBES
PROBE_EXPECT = {
    "THEN": dict(then_xy=True, then_yx=False, while_same=False, x_other=False, other_y=False, far=False,
                 then_gap45=True, then_ovl03=True, while_unequal=False, partial=False,
                 only_x=False, only_y=False, other=False),
    "WHILE": dict(then_xy=False, then_yx=False, while_same=True, x_other=False, other_y=False, far=False,
                  then_gap45=False, then_ovl03=False, while_unequal=True, partial=False,
                  only_x=False, only_y=False, other=False),
}


def _probe_events(name: str, specs) -> Tuple[List[Event], List[dict]]:
    evs = [Event(f"{name}:{i}", "probe", name, t, float(s), float(e)) for i, (t, s, e) in enumerate(specs)]
    return evs, [{"type": e.type, "start": e.start, "end": e.end} for e in evs]


def build_probes_v05(gold: Query):
    """Returns [(name, events, dicts, gold_accepts)] for X = gold.first, Y = gold.second."""
    x, y = gold.first, gold.second
    o = next(t for t in EVENT_TYPES if t not in (x, y))
    tx, ty = ((1, 3), (5, 7)) if gold.relation == "THEN" else ((2, 6), (2, 6))
    specs = {
        "then_xy": [(x, 1, 3), (y, 5, 7)],
        "then_yx": [(y, 1, 3), (x, 5, 7)],
        "while_same": [(x, 2, 6), (y, 2, 6)],
        "x_other": [(x, *tx), (o, *ty)],
        "other_y": [(o, *tx), (y, *ty)],
        "far": [(x, 1, 2), (y, 10, 12)],
        "then_gap45": [(x, 1, 2), (y, 6.5, 8)],
        "then_ovl03": [(x, 1, 3), (y, 2.7, 5)],
        "while_unequal": [(x, 2, 6), (y, 2.5, 5.5)],
        "partial": [(x, 0, 6), (y, 4, 10)],
        "only_x": [(x, 1, 3)],
        "only_y": [(y, 5, 7)],
        "other": [(o, 1, 3), (o, 5, 7)],
    }
    out = []
    for n in ALL_PROBES:
        evs, dicts = _probe_events(n, specs[n])
        out.append((n, evs, dicts, query_accepts(gold, evs)))
    got = {n: g for n, _, _, g in out}
    assert got == PROBE_EXPECT[gold.relation], f"probe design mismatch for {gold}: {got}"
    return out


def infer_v05(acc: Dict[str, bool]) -> str:
    key = (acc["then_xy"], acc["then_yx"], acc["while_same"])
    return {(True, False, False): "THEN_XY", (False, True, False): "THEN_YX",
            (False, False, True): "WHILE"}.get(key, "OTHER")


def accepts_p(pred: Optional[Query], probes) -> Optional[Dict[str, bool]]:
    return None if pred is None else {n: query_accepts(pred, evs) for n, evs, _, _ in probes}


def score_v05(gold: Query, acc: Optional[Dict[str, bool]], probes) -> dict:
    """Interpretation accuracy (5 interpretation probes) and judgment accuracy (all 13)."""
    if acc is None:
        return {"valid": False, "inferred": None, "types_ok": False, "interp_correct": False,
                "judge_correct": False, "impl_error": False, "impl_fail": ""}
    inferred = infer_v05(acc)
    want = "WHILE" if gold.relation == "WHILE" else "THEN_XY"
    types_ok = not (acc["x_other"] or acc["other_y"])
    interp = inferred == want and types_ok
    judge = all(acc[n] == g for n, _, _, g in probes)
    failed = [n for n, _, _, g in probes if n in IMPL_PROBES and acc[n] != g]
    return {"valid": True, "inferred": inferred, "types_ok": types_ok, "interp_correct": bool(interp),
            "judge_correct": bool(judge), "impl_error": bool(interp and not judge), "impl_fail": ";".join(failed)}


def error_type(gold: Query, inferred: Optional[str], types_ok: bool, valid: bool) -> str:
    """Shared interpretation error label (from probes) for both methods."""
    if not valid:
        return "invalid"
    want = "WHILE" if gold.relation == "WHILE" else "THEN_XY"
    if inferred == want and types_ok:
        return "correct"
    if inferred == "OTHER":
        return "unclassified"
    if gold.relation == "THEN" and inferred == "WHILE" or gold.relation == "WHILE" and inferred in ("THEN_XY", "THEN_YX"):
        return "relation_type"
    if gold.relation == "THEN" and inferred == "THEN_YX":
        return "direction" if types_ok else "direction+types"
    return "event_types"


def p_direction_subtype(plan: Optional[dict], gold: Query, first_mentioned: str, congruent: bool) -> Optional[str]:
    """H3: for a P output with the right event types and THEN but the wrong direction (gold THEN).
    args_error: slots follow mention order, args do not follow temporal order.
    double_reversal: slots already follow temporal order and args reverses them again.
    In congruent sentences (mention order = temporal order) the two coincide."""
    try:
        q = parse_tqp(plan)
    except Exception:
        return None
    if gold.relation != "THEN" or q.relation != "THEN" or (q.first, q.second) != (gold.second, gold.first):
        return None
    e1, args = plan["events"]["e1"], plan["relation"]["args"]
    slots_mention = e1 == first_mentioned
    slots_temporal = e1 == gold.first
    reversed_args = args == ["e2", "e1"]
    if congruent:
        return "args_reversed(congruent)" if slots_mention and reversed_args else "slot_reversal(congruent)"
    if slots_mention and not reversed_args:
        return "args_error"
    if slots_temporal and reversed_args:
        return "double_reversal"
    return "other"
