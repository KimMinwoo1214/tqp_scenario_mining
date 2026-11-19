# -*- coding: utf-8 -*-
"""TQP x Temporal Event Graph core engine (v0.3; detector changes documented in DetectorConfig).

Layers implemented here
  - Temporal Query Plan (TQP): strict JSON schema -> Query (no code execution)
  - Temporal Event Graph (TEG): event nodes, HAS_EVENT edges, typed THEN/WHILE edges
  - Query execution: IntervalExecutor (baseline, evaluates intervals at query time)
                     GraphQueryExecutor (looks up precomputed relation edges)
  - AV2 Sensor grounding: annotations.feather + city_SE3_egovehicle.feather per log

Changes from v0.1
  - Boundary clipping is split into clipped_start / clipped_end. A pair is UNCERTAIN
    only when extending the unobserved boundary could change the verdict for the
    queried relation (dense-grid check), instead of whenever any event is clipped.
  - Timing: graph build is timed without tracemalloc; memory is measured separately.
  - Retrieval latency compares "return matching (track, event pair)" for both
    executors, so the graph's precomputation is actually exercised.
  - Event times are relative to the log start (not the track's first observation).
"""
from __future__ import annotations

import re
import time
import tracemalloc
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation

VERSION = "0.4.0"  # frozen for the main run (detector v0.4)
EVENT_TYPES = ("TURN", "DECELERATE", "ACCELERATE", "STOP")
RELATIONS = ("THEN", "WHILE")
LABELS = ("NONE", "THEN", "WHILE", "PARTIAL")
STATUS_ORDER = ("POSITIVE", "UNCERTAIN", "PARTIAL", "NEGATIVE")
VEHICLE_CATEGORIES = {
    "REGULAR_VEHICLE", "LARGE_VEHICLE", "BUS", "BOX_TRUCK",
    "TRUCK", "TRUCK_CAB", "SCHOOL_BUS", "ARTICULATED_BUS",
}
_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RelationConfig:
    then_max_gap_s: float = 5.0          # THEN: start(B) - end(A) <= this
    then_overlap_tolerance_s: float = 0.5  # THEN: allows up to 0.5 s overlap
    while_min_overlap_s: float = 0.5     # WHILE/PARTIAL: overlap must exceed this
    while_min_ratio: float = 0.8         # WHILE: overlap / shorter duration >= this
    # Unobserved boundaries are checked on a grid at the 10 Hz sampling resolution.
    # With >=3 unobserved boundaries in one pair the grid is coarsened (rare case).
    clip_grid_step_s: float = 0.1
    clip_grid_max_points: int = 300_000


@dataclass(frozen=True)
class DetectorConfig:
    """v0.3 defaults. Changes from v0.2:
    - segments shorter than min_segment_s are skipped (short tracks gave edge artefacts)
    - derivative-based events (DECELERATE/ACCELERATE/TURN) ignore edge_trim_s at both
      segment edges; an event reaching the trimmed edge is marked clipped
    - same-type runs separated by <= merge_gap_s are merged (fragmentation fix)
    - acceleration events: candidate run |a| >= accel_mask_threshold; accepted if
      strong (peak |a| >= accel_peak_threshold, |dv| >= min_speed_change, duration >= 0.5 s)
      or gentle (duration >= gentle_min_duration_s and |dv| >= gentle_min_speed_change)
    """
    window: int = 11                 # Savitzky-Golay window at 10 Hz (1.1 s)
    polyorder: int = 2
    max_gap_s: float = 0.25          # split track at missing frames
    min_segment_s: float = 3.0
    min_displacement_m: float = 3.0
    edge_trim_s: float = 0.5
    merge_gap_s: float = 1.0
    accel_mask_threshold: float = 0.5    # m/s^2
    accel_peak_threshold: float = 1.0
    min_accel_duration_s: float = 0.5
    min_speed_change: float = 1.5        # m/s
    gentle_min_duration_s: float = 2.0
    gentle_min_speed_change: float = 2.0
    stop_speed: float = 0.5              # m/s
    min_stop_duration_s: float = 1.0
    turn_rate_deg_s: float = 5.0
    min_turn_deg: float = 45.0
    min_turn_travel_m: float = 5.0
    min_turn_speed: float = 1.0


# Reproduces the v0.2 detector exactly (used for before/after comparison).
DETECTOR_V02 = DetectorConfig(min_segment_s=1.0, edge_trim_s=0.0, merge_gap_s=0.0,
                              accel_mask_threshold=1.0, accel_peak_threshold=1.0,
                              gentle_min_duration_s=float("inf"))

DEFAULT_REL = RelationConfig()
DEFAULT_DET = DetectorConfig()


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Event:
    event_id: str
    log_id: str
    track_id: str
    type: str
    start: float
    end: float
    clipped_start: bool = False   # true start may be earlier (observation began mid-event)
    clipped_end: bool = False     # true end may be later (observation ended mid-event)

    def __post_init__(self) -> None:
        if self.type not in EVENT_TYPES:
            raise ValueError(f"Unknown event type: {self.type}")
        if not (np.isfinite(self.start) and np.isfinite(self.end) and self.end > self.start):
            raise ValueError("Event must have a finite positive duration")

    @property
    def track_key(self) -> Tuple[str, str]:
        return (self.log_id, self.track_id)

    @property
    def clipped(self) -> bool:
        return self.clipped_start or self.clipped_end


@dataclass(frozen=True)
class Query:
    first: str      # temporally first argument (for WHILE: order irrelevant)
    relation: str
    second: str

    def __post_init__(self) -> None:
        if self.first not in EVENT_TYPES or self.second not in EVENT_TYPES:
            raise ValueError("Unsupported event type")
        if self.relation not in RELATIONS:
            raise ValueError("Unsupported relation")

    def __str__(self) -> str:
        return f"{self.first} {self.relation} {self.second}"

    def semantic_key(self) -> tuple:
        if self.relation == "WHILE":
            return ("WHILE",) + tuple(sorted((self.first, self.second)))
        return ("THEN", self.first, self.second)

    def to_tqp(self) -> dict:
        return {"events": {"e1": self.first, "e2": self.second},
                "relation": {"op": self.relation, "args": ["e1", "e2"]}}


@dataclass(frozen=True, order=True)
class Match:
    log_id: str
    track_id: str
    first_event_id: str
    first_type: str
    first_start: float
    first_end: float
    second_event_id: str
    second_type: str
    second_start: float
    second_end: float

    @staticmethod
    def of(a: Event, b: Event) -> "Match":
        return Match(a.log_id, a.track_id, a.event_id, a.type, a.start, a.end,
                     b.event_id, b.type, b.start, b.end)


def parse_tqp(plan: dict) -> Query:
    """Validate a TQP dict and return the temporally ordered Query.

    Schema: {"events": {"e1": TYPE, "e2": TYPE},
             "relation": {"op": "THEN"|"WHILE", "args": [slot, slot]}}
    args lists the slots in temporal order (args[0] happens first for THEN).
    """
    if not isinstance(plan, dict) or set(plan) != {"events", "relation"}:
        raise ValueError("TQP must have exactly the keys 'events' and 'relation'")
    events, relation = plan["events"], plan["relation"]
    if not isinstance(events, dict) or set(events) != {"e1", "e2"}:
        raise ValueError("Expected event slots e1 and e2")
    for slot, typ in events.items():
        if typ not in EVENT_TYPES:
            raise ValueError(f"Unsupported event type in {slot}: {typ!r}")
    if not isinstance(relation, dict) or set(relation) != {"op", "args"}:
        raise ValueError("Expected relation keys op and args")
    if relation["op"] not in RELATIONS:
        raise ValueError(f"Unsupported relation op: {relation['op']!r}")
    args = relation["args"]
    if not isinstance(args, list) or sorted(args) != ["e1", "e2"]:
        raise ValueError("Relation args must contain e1 and e2 exactly once")
    return Query(events[args[0]], relation["op"], events[args[1]])


# ---------------------------------------------------------------------------
# Interval relation semantics (shared by both executors)
# ---------------------------------------------------------------------------
def _label_codes(sa, ea, sb, eb, cfg: RelationConfig):
    """Vectorised relation label for ordered pair (A, B). 0 NONE 1 THEN 2 WHILE 3 PARTIAL."""
    ov = np.maximum(0.0, np.minimum(ea, eb) - np.maximum(sa, sb))
    shorter = np.minimum(ea - sa, eb - sb)
    overlapping = ov > cfg.while_min_overlap_s
    ratio = ov / np.where(shorter > 0, shorter, 1.0)
    is_while = overlapping & (ratio >= cfg.while_min_ratio - 1e-12)
    is_partial = overlapping & ~is_while
    gap = sb - ea
    is_then = ((~overlapping) & (sa < sb)
               & (gap >= -cfg.then_overlap_tolerance_s - 1e-12)
               & (gap <= cfg.then_max_gap_s + 1e-12))
    code = np.where(is_then, 1, 0)
    code = np.where(is_while, 2, code)
    code = np.where(is_partial, 3, code)
    return code


def possible_relations(a: Event, b: Event, cfg: RelationConfig = DEFAULT_REL) -> FrozenSet[str]:
    """Set of labels the ordered pair (a, b) can take over all completions of
    clipped (unobserved) boundaries. Unclipped pairs return a singleton."""
    if a.event_id == b.event_id or a.track_key != b.track_key:
        return frozenset({"NONE"})
    free = (a.clipped_start, a.clipped_end, b.clipped_start, b.clipped_end)
    k = sum(free)
    if k == 0:
        code = int(_label_codes(a.start, a.end, b.start, b.end, cfg))
        return frozenset({LABELS[code]})
    # Beyond this extension every comparison in _label_codes is saturated.
    horizon = (max(a.end, b.end) - min(a.start, b.start)
               + cfg.then_max_gap_s + cfg.then_overlap_tolerance_s + 1.0)
    n = int(np.ceil(horizon / cfg.clip_grid_step_s)) + 1
    n = max(2, min(n, int(cfg.clip_grid_max_points ** (1.0 / k))))
    ext = np.linspace(0.0, horizon, n)
    axis_i = 0

    def axis(base: float, sign: float, is_free: bool):
        nonlocal axis_i
        if not is_free:
            return np.float64(base)
        shape = [1] * k
        shape[axis_i] = n
        axis_i += 1
        return base + sign * ext.reshape(shape)

    sa = axis(a.start, -1.0, a.clipped_start)
    ea = axis(a.end, +1.0, a.clipped_end)
    sb = axis(b.start, -1.0, b.clipped_start)
    eb = axis(b.end, +1.0, b.clipped_end)
    codes = np.unique(_label_codes(sa, ea, sb, eb, cfg))
    return frozenset(LABELS[int(c)] for c in codes)


def pair_status(possible: FrozenSet[str], relation: str) -> str:
    """POSITIVE: relation holds for every completion. UNCERTAIN: holds for some.
    PARTIAL: insufficient overlap (ambiguous for both operators). Else NEGATIVE."""
    if possible == frozenset({relation}):
        return "POSITIVE"
    if relation in possible:
        return "UNCERTAIN"
    if "PARTIAL" in possible:
        return "PARTIAL"
    return "NEGATIVE"


def stronger(s1: str, s2: str) -> str:
    return s1 if STATUS_ORDER.index(s1) <= STATUS_ORDER.index(s2) else s2


# ---------------------------------------------------------------------------
# Executor 1: interval baseline
# ---------------------------------------------------------------------------
class IntervalExecutor:
    """Evaluates interval pairs at query time; only events are grouped by track/type."""

    def __init__(self, events: Sequence[Event], track_universe: Iterable[Tuple[str, str]] = (),
                 cfg: RelationConfig = DEFAULT_REL):
        self.cfg = cfg
        grouped: Dict[Tuple[str, str], Dict[str, List[Event]]] = defaultdict(lambda: defaultdict(list))
        for e in events:
            grouped[e.track_key][e.type].append(e)
        self.by_track = {k: dict(v) for k, v in grouped.items()}
        self.keys = set(map(tuple, track_universe)) | set(self.by_track)

    def _pairs(self, q: Query):
        for key, types in self.by_track.items():
            for a in types.get(q.first, ()):
                for b in types.get(q.second, ()):
                    if a.event_id != b.event_id:
                        yield key, a, b

    def retrieve(self, q: Query) -> List[Match]:
        return sorted(Match.of(a, b) for _, a, b in self._pairs(q)
                      if pair_status(possible_relations(a, b, self.cfg), q.relation) == "POSITIVE")

    def statuses(self, q: Query) -> Dict[Tuple[str, str], str]:
        out = {k: "NEGATIVE" for k in self.keys}
        for key, a, b in self._pairs(q):
            out[key] = stronger(out[key], pair_status(possible_relations(a, b, self.cfg), q.relation))
        return out


# ---------------------------------------------------------------------------
# Temporal Event Graph + executor 2
# ---------------------------------------------------------------------------
class TemporalEventGraph:
    """Nodes: tracks and events. Edges: HAS_EVENT (track->event) and typed relation
    edges THEN/WHILE (event->event) that hold for every boundary completion.
    Ambiguous pairs are indexed separately so they are never returned as matches
    and never counted as negatives."""

    def __init__(self, events: Sequence[Event], track_universe: Iterable[Tuple[str, str]] = (),
                 cfg: RelationConfig = DEFAULT_REL):
        self.cfg = cfg
        self.event_nodes: Dict[str, Event] = {}
        self.has_event: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for e in events:
            if e.event_id in self.event_nodes:
                raise ValueError(f"Duplicate event_id {e.event_id}")
            self.event_nodes[e.event_id] = e
            self.has_event[e.track_key].append(e.event_id)
        self.track_nodes = set(map(tuple, track_universe)) | set(self.has_event)
        self.relation_edges: List[Tuple[str, str, str]] = []
        self.edge_index: Dict[tuple, List[Tuple[str, str]]] = defaultdict(list)
        self.ambiguous_index: Dict[tuple, Dict[Tuple[str, str], str]] = defaultdict(dict)
        self.pairs_evaluated = 0
        self.clip_aware_pairs = 0
        for key, ids in self.has_event.items():
            evs = [self.event_nodes[i] for i in ids]
            for a in evs:
                for b in evs:
                    if a.event_id == b.event_id:
                        continue
                    self.pairs_evaluated += 1
                    self.clip_aware_pairs += int(a.clipped or b.clipped)
                    poss = possible_relations(a, b, cfg)
                    for rel in RELATIONS:
                        st = pair_status(poss, rel)
                        index_key = (rel, a.type, b.type)
                        if st == "POSITIVE":
                            self.relation_edges.append((a.event_id, rel, b.event_id))
                            self.edge_index[index_key].append((a.event_id, b.event_id))
                        elif st in ("UNCERTAIN", "PARTIAL"):
                            bucket = self.ambiguous_index[index_key]
                            bucket[key] = stronger(bucket.get(key, "NEGATIVE"), st)
        self.edge_index = dict(self.edge_index)
        self.ambiguous_index = dict(self.ambiguous_index)

    def summary(self) -> dict:
        return {
            "track_nodes": len(self.track_nodes),
            "event_nodes": len(self.event_nodes),
            "has_event_edges": sum(len(v) for v in self.has_event.values()),
            "relation_edges": len(self.relation_edges),
            "relation_edges_by_type": dict(Counter(r for _, r, _ in self.relation_edges)),
            "relation_edges_by_pattern": {f"{k[1]} {k[0]} {k[2]}": len(v)
                                          for k, v in sorted(self.edge_index.items())},
            "ordered_event_pairs_evaluated": self.pairs_evaluated,
            "pairs_with_clipped_boundary": self.clip_aware_pairs,
            "note": "WHILE is symmetric, so each WHILE relation is stored as two directed edges.",
        }

    def edges_frame(self) -> pd.DataFrame:
        rows = []
        for s, r, d in self.relation_edges:
            a, b = self.event_nodes[s], self.event_nodes[d]
            rows.append({"source_event_id": s, "relation": r, "target_event_id": d,
                         "log_id": a.log_id, "track_id": a.track_id,
                         "source_type": a.type, "target_type": b.type,
                         "source_start": a.start, "source_end": a.end,
                         "target_start": b.start, "target_end": b.end})
        return pd.DataFrame(rows, columns=["source_event_id", "relation", "target_event_id", "log_id",
                                           "track_id", "source_type", "target_type", "source_start",
                                           "source_end", "target_start", "target_end"])


class GraphQueryExecutor:
    """Translates a TQP Query into a typed edge-pattern lookup on the graph."""

    def __init__(self, graph: TemporalEventGraph):
        self.g = graph

    @staticmethod
    def pattern(q: Query) -> str:
        return f"(:Event {{type:'{q.first}'}})-[:{q.relation}]->(:Event {{type:'{q.second}'}})"

    def retrieve(self, q: Query) -> List[Match]:
        pairs = self.g.edge_index.get((q.relation, q.first, q.second), ())
        nodes = self.g.event_nodes
        return sorted(Match.of(nodes[s], nodes[d]) for s, d in pairs)

    def statuses(self, q: Query) -> Dict[Tuple[str, str], str]:
        idx = (q.relation, q.first, q.second)
        out = {k: "NEGATIVE" for k in self.g.track_nodes}
        out.update(self.g.ambiguous_index.get(idx, {}))
        for s, _ in self.g.edge_index.get(idx, ()):
            out[self.g.event_nodes[s].track_key] = "POSITIVE"
        return out


def all_queries() -> List[Query]:
    return [Query(x, r, y) for x in EVENT_TYPES for r in RELATIONS for y in EVENT_TYPES]


def verify_and_benchmark(events: Sequence[Event], universe: Iterable[Tuple[str, str]],
                         queries: Sequence[Query], cfg: RelationConfig = DEFAULT_REL, *,
                         repeat: int = 30, build_repeats: int = 3,
                         time_budget_s: float = 0.5) -> Tuple[dict, TemporalEventGraph]:
    universe = set(map(tuple, universe))
    build_ms, prep_ms = [], []
    graph = None
    for _ in range(build_repeats):
        t0 = time.perf_counter()
        graph = TemporalEventGraph(events, universe, cfg)
        build_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        interval = IntervalExecutor(events, universe, cfg)
        prep_ms.append((time.perf_counter() - t0) * 1000)
    tracemalloc.start()
    probe = TemporalEventGraph(events, universe, cfg)
    retained, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del probe
    gexec = GraphQueryExecutor(graph)

    per_query = []
    for q in queries:
        si, sg = interval.statuses(q), gexec.statuses(q)
        ri, rg = interval.retrieve(q), gexec.retrieve(q)
        mism = [[*k, si[k], sg.get(k)] for k in si if si[k] != sg.get(k)]
        per_query.append({
            "query": str(q), "graph_pattern": gexec.pattern(q),
            "status_equal": si == sg, "retrieval_equal": ri == rg,
            "matches": len(rg), "positive_tracks": sum(v == "POSITIVE" for v in sg.values()),
            "uncertain_tracks": sum(v == "UNCERTAIN" for v in sg.values()),
            "partial_tracks": sum(v == "PARTIAL" for v in sg.values()),
            "negative_tracks": sum(v == "NEGATIVE" for v in sg.values()),
            "mismatches": mism[:10],
        })

    def latency(fn) -> List[float]:
        """Mean ms per call; at most `repeat` calls or ~time_budget_s per query (>=1 call)."""
        out = []
        for q in queries:
            n, t0 = 0, time.perf_counter()
            while n < repeat:
                fn(q)
                n += 1
                if time.perf_counter() - t0 > time_budget_s:
                    break
            out.append((time.perf_counter() - t0) * 1000 / n)
        return out

    lat_i, lat_g = latency(interval.retrieve), latency(gexec.retrieve)
    for row, li, lg in zip(per_query, lat_i, lat_g):
        row["interval_ms"], row["graph_ms"] = round(li, 5), round(lg, 5)
    build = float(np.median(build_ms))
    saved = float(np.mean(lat_i) - np.mean(lat_g))
    report = {
        "version": VERSION,
        "relation_config": asdict(cfg),
        "graph": graph.summary(),
        "graph_build_ms_median": round(build, 3),
        "interval_prep_ms_median": round(float(np.median(prep_ms)), 3),
        "graph_retained_memory_kib": round(retained / 1024, 1),
        "graph_build_peak_memory_kib": round(peak / 1024, 1),
        "retrieval_latency_interval_ms_mean": round(float(np.mean(lat_i)), 5),
        "retrieval_latency_graph_ms_mean": round(float(np.mean(lat_g)), 5),
        "break_even_query_count": (int(np.ceil(build / saved)) if saved > 0 else None),
        "latency_repeat_max": repeat,
        "latency_time_budget_s_per_query": time_budget_s,
        "query_count": len(queries),
        "all_status_equal": all(r["status_equal"] for r in per_query),
        "all_retrieval_equal": all(r["retrieval_equal"] for r in per_query),
        "per_query": per_query,
    }
    return report, graph


# ---------------------------------------------------------------------------
# Event detection from trajectories
# ---------------------------------------------------------------------------
def contiguous_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    padded = np.r_[False, np.asarray(mask, dtype=bool), False].astype(int)
    d = np.diff(padded)
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _extract_events_v03(t, x, y, log_id: str, track_id: str,
                   dcfg: DetectorConfig = DEFAULT_DET) -> Tuple[List[Event], List[dict]]:
    """Rule-based pilot detector for ~10 Hz trajectories (times in seconds, log-relative).
    Returns events and per-segment signals for visual verification."""
    t, x, y = (np.asarray(v, float) for v in (t, x, y))
    events: List[Event] = []
    segments: List[dict] = []
    if len(t) < dcfg.window:
        return events, segments
    order = np.argsort(t)
    t, x, y = t[order], x[order], y[order]
    cuts = np.r_[0, np.flatnonzero(np.diff(t) > dcfg.max_gap_s) + 1, len(t)]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        tt, xx, yy = t[lo:hi], x[lo:hi], y[lo:hi]
        if len(tt) < dcfg.window or np.any(np.diff(tt) <= 0):
            continue
        if tt[-1] - tt[0] < dcfg.min_segment_s - 1e-9:
            continue
        dt = float(np.median(np.diff(tt)))
        if not 0.075 <= dt <= 0.125:
            continue
        if np.max(np.hypot(xx - xx[0], yy - yy[0])) < dcfg.min_displacement_m:
            continue
        w, p = dcfg.window, dcfg.polyorder
        vx = savgol_filter(xx, w, p, deriv=1, delta=dt)
        vy = savgol_filter(yy, w, p, deriv=1, delta=dt)
        speed = np.hypot(vx, vy)
        accel = savgol_filter(speed, w, p, deriv=1, delta=dt)
        heading = np.unwrap(np.arctan2(vy, vx))
        yaw_rate = savgol_filter(heading, w, p, deriv=1, delta=dt)
        travel = np.r_[0.0, np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))]
        segments.append({"t": tt, "x": xx, "y": yy, "speed": speed,
                         "accel": accel, "yaw_rate_deg_s": np.rad2deg(yaw_rate)})
        n = len(tt)
        n_trim = int(round(dcfg.edge_trim_s / dt))
        n_merge = int(round(dcfg.merge_gap_s / dt))
        rate = np.deg2rad(dcfg.turn_rate_deg_s)
        # (kind, raw mask, derivative-based?)
        candidates = [("DECELERATE", accel <= -dcfg.accel_mask_threshold, True),
                      ("ACCELERATE", accel >= dcfg.accel_mask_threshold, True),
                      ("STOP", speed < dcfg.stop_speed, False),
                      ("TURN", yaw_rate >= rate, True),
                      ("TURN", yaw_rate <= -rate, True)]
        for kind, raw_mask, derivative in candidates:
            mask = close_gaps(raw_mask, n_merge)
            lo_edge, hi_edge = 0, n
            if derivative and n_trim > 0:
                mask[:n_trim] = False
                mask[max(n - n_trim, 0):] = False
                lo_edge, hi_edge = n_trim, n - n_trim
            for a, b in contiguous_runs(mask):
                duration = (b - a) * dt
                if kind == "STOP":
                    valid = duration >= dcfg.min_stop_duration_s - 1e-9
                elif kind in ("DECELERATE", "ACCELERATE"):
                    sign = -1.0 if kind == "DECELERATE" else 1.0
                    dv = sign * (speed[b - 1] - speed[a])
                    peak = float(np.max(sign * accel[a:b]))
                    strong = (duration >= dcfg.min_accel_duration_s - 1e-9
                              and peak >= dcfg.accel_peak_threshold - 1e-12
                              and dv >= dcfg.min_speed_change)
                    gentle = (duration >= dcfg.gentle_min_duration_s - 1e-9
                              and dv >= dcfg.gentle_min_speed_change)
                    valid = strong or gentle
                else:
                    angle = abs(_trapz(yaw_rate[a:b], dx=dt))
                    valid = (angle >= np.deg2rad(dcfg.min_turn_deg)
                             and travel[b - 1] - travel[a] >= dcfg.min_turn_travel_m
                             and float(np.median(speed[a:b])) >= dcfg.min_turn_speed)
                if valid:
                    events.append(Event(f"{log_id}:{track_id}:{len(events)}", log_id, track_id, kind,
                                        float(tt[a]), float(tt[b - 1] + dt),
                                        clipped_start=(a <= lo_edge), clipped_end=(b >= hi_edge)))
    return events, segments


def close_gaps(mask: np.ndarray, max_gap_n: int) -> np.ndarray:
    """Fill interior False runs of length <= max_gap_n (merges fragmented events)."""
    out = np.asarray(mask, dtype=bool).copy()
    if max_gap_n <= 0:
        return out
    for a, b in contiguous_runs(~out):
        if a > 0 and b < len(out) and (b - a) <= max_gap_n:
            out[a:b] = True
    return out


def detector_quality(events: Sequence[Event], universe: Iterable[Tuple[str, str]]) -> dict:
    """Automatic sanity indicators for a detector run (not a substitute for manual checks)."""
    by_track: Dict[Tuple[str, str], List[Event]] = defaultdict(list)
    for e in events:
        by_track[e.track_key].append(e)
    same_type_gaps = []
    for evs in by_track.values():
        for typ in EVENT_TYPES:
            seq = sorted((e for e in evs if e.type == typ), key=lambda e: e.start)
            same_type_gaps += [b.start - a.end for a, b in zip(seq, seq[1:])]
    gaps = np.asarray(same_type_gaps, float)
    universe = set(map(tuple, universe))
    counts = Counter(e.type for e in events)
    return {
        "moving_tracks": len(universe),
        "tracks_with_events": len(by_track),
        "events": len(events),
        "events_by_type": {t: int(counts.get(t, 0)) for t in EVENT_TYPES},
        "clipped_event_rate": round(float(np.mean([e.clipped for e in events])), 3) if events else None,
        "consecutive_same_type_pairs": int(len(gaps)),
        "same_type_gap_le_1s": int(np.sum(gaps <= 1.0)) if len(gaps) else 0,
        "median_duration_s_by_type": {t: round(float(np.median([e.end - e.start for e in events if e.type == t])), 2)
                                      for t in EVENT_TYPES if counts.get(t)},
    }




# ---------------------------------------------------------------------------
# Detector v0.4 (frozen for the main run; developed on the 150 AV2 val logs)
# ---------------------------------------------------------------------------
from scipy.signal import savgol_filter  # noqa: E402

@dataclass(frozen=True)
class DetectorConfigV04(DetectorConfig):
    """v0.3 + (1) 반대 방향 사건을 건너뛰는 병합 금지 (2) 흔들림 제거 (3) 저속 회전 제외"""
    jitter_window_s: float = 1.5       # 사건 앞뒤 1.5초를 확인
    jitter_return_frac: float = 0.3    # 그 안에 속도가 반대쪽 끝값의 30% 이내로 되돌아오면 흔들림
    use_jitter_filter: bool = True
    turn_mask_min_speed: float = 1.5   # m/s 미만에서는 요 레이트를 회전으로 보지 않음


DET_V04 = DetectorConfigV04()
DET_V04_NOJITTER = DetectorConfigV04(use_jitter_filter=False)


def close_gaps_guarded(mask, max_gap_n, block):
    """길이 max_gap_n 이하의 내부 빈칸을 메우되, 빈칸 안에 반대 방향 사건(block)이 있으면 메우지 않음."""
    out = np.asarray(mask, dtype=bool).copy()
    if max_gap_n <= 0:
        return out
    for a, b in contiguous_runs(~out):
        if a > 0 and b < len(out) and (b - a) <= max_gap_n and not np.any(block[a:b]):
            out[a:b] = True
    return out


def is_jitter(kind, speed, a, b, n_win, frac):
    vs, ve = float(speed[a]), float(speed[b - 1])
    d = abs(ve - vs)
    if d <= 0:
        return True
    pre = speed[max(0, a - n_win):a]
    post = speed[b:b + n_win]
    if kind == "DECELERATE":   # 직전에 이미 낮았거나(위로 튄 값) 직후 다시 높아짐(아래로 꺼진 값)
        return bool((len(pre) and pre.min() <= ve + frac * d) or (len(post) and post.max() >= vs - frac * d))
    return bool((len(pre) and pre.max() >= ve - frac * d) or (len(post) and post.min() <= vs + frac * d))


_extract_v03 = _extract_events_v03


def extract_events_v04(t, x, y, log_id, track_id, dcfg=DEFAULT_DET):
    if not isinstance(dcfg, DetectorConfigV04):
        return _extract_v03(t, x, y, log_id, track_id, dcfg)
    t, x, y = (np.asarray(v, float) for v in (t, x, y))
    events_, segments = [], []
    if len(t) < dcfg.window:
        return events_, segments
    o = np.argsort(t)
    t, x, y = t[o], x[o], y[o]
    cuts = np.r_[0, np.flatnonzero(np.diff(t) > dcfg.max_gap_s) + 1, len(t)]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        tt, xx, yy = t[lo:hi], x[lo:hi], y[lo:hi]
        if len(tt) < dcfg.window or np.any(np.diff(tt) <= 0):
            continue
        if tt[-1] - tt[0] < dcfg.min_segment_s - 1e-9:
            continue
        dt = float(np.median(np.diff(tt)))
        if not 0.075 <= dt <= 0.125:
            continue
        if np.max(np.hypot(xx - xx[0], yy - yy[0])) < dcfg.min_displacement_m:
            continue
        w, p = dcfg.window, dcfg.polyorder
        vx = savgol_filter(xx, w, p, deriv=1, delta=dt)
        vy = savgol_filter(yy, w, p, deriv=1, delta=dt)
        speed = np.hypot(vx, vy)
        accel = savgol_filter(speed, w, p, deriv=1, delta=dt)
        yaw_rate = savgol_filter(np.unwrap(np.arctan2(vy, vx)), w, p, deriv=1, delta=dt)
        travel = np.r_[0.0, np.cumsum(np.hypot(np.diff(xx), np.diff(yy)))]
        segments.append({"t": tt, "x": xx, "y": yy, "speed": speed, "accel": accel,
                         "yaw_rate_deg_s": np.rad2deg(yaw_rate)})
        n = len(tt)
        n_trim = int(round(dcfg.edge_trim_s / dt))
        n_merge = int(round(dcfg.merge_gap_s / dt))
        n_win = int(round(dcfg.jitter_window_s / dt))
        rate = np.deg2rad(dcfg.turn_rate_deg_s)
        moving = speed >= dcfg.turn_mask_min_speed
        dec = accel <= -dcfg.accel_mask_threshold
        acc = accel >= dcfg.accel_mask_threshold
        left = (yaw_rate >= rate) & moving
        right = (yaw_rate <= -rate) & moving
        candidates = [("DECELERATE", dec, acc, True), ("ACCELERATE", acc, dec, True),
                      ("STOP", speed < dcfg.stop_speed, np.zeros(n, bool), False),
                      ("TURN", left, right, True), ("TURN", right, left, True)]
        for kind, raw, block, derivative in candidates:
            mask = close_gaps_guarded(raw, n_merge, block)
            lo_e, hi_e = 0, n
            if derivative and n_trim > 0:
                mask[:n_trim] = False
                mask[max(n - n_trim, 0):] = False
                lo_e, hi_e = n_trim, n - n_trim
            for a, b in contiguous_runs(mask):
                dur = (b - a) * dt
                if kind == "STOP":
                    valid = dur >= dcfg.min_stop_duration_s - 1e-9
                elif kind in ("DECELERATE", "ACCELERATE"):
                    sign = -1.0 if kind == "DECELERATE" else 1.0
                    dv = sign * (speed[b - 1] - speed[a])
                    peak = float(np.max(sign * accel[a:b]))
                    strong = (dur >= dcfg.min_accel_duration_s - 1e-9
                              and peak >= dcfg.accel_peak_threshold - 1e-12 and dv >= dcfg.min_speed_change)
                    gentle = dur >= dcfg.gentle_min_duration_s - 1e-9 and dv >= dcfg.gentle_min_speed_change
                    valid = strong or gentle
                    if valid and dcfg.use_jitter_filter:
                        valid = not is_jitter(kind, speed, a, b, n_win, dcfg.jitter_return_frac)
                else:
                    angle = abs(_trapz(yaw_rate[a:b], dx=dt))
                    valid = (angle >= np.deg2rad(dcfg.min_turn_deg)
                             and travel[b - 1] - travel[a] >= dcfg.min_turn_travel_m
                             and float(np.median(speed[a:b])) >= dcfg.min_turn_speed)
                if valid:
                    events_.append(Event(f"{log_id}:{track_id}:{len(events_)}", log_id, track_id, kind,
                                           float(tt[a]), float(tt[b - 1] + dt),
                                           clipped_start=(a <= lo_e), clipped_end=(b >= hi_e)))
    return events_, segments



extract_events = extract_events_v04
DEFAULT_DET = DET_V04


# ---------------------------------------------------------------------------
# AV2 Sensor log loading
# ---------------------------------------------------------------------------
ANNOTATION = "annotations.feather"
POSE = "city_SE3_egovehicle.feather"
SENSOR_COLUMNS = {
    ANNOTATION: ["timestamp_ns", "track_uuid", "category", "tx_m", "ty_m", "tz_m"],
    POSE: ["timestamp_ns", "qw", "qx", "qy", "qz", "tx_m", "ty_m", "tz_m"],
}
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I)


def load_av2_log(folder: Path, *, pose_tolerance_ms: float = 15.0,
                 dcfg: Optional[DetectorConfig] = None):
    """Annotation cuboid centres (ego frame) -> city frame via nearest ego pose,
    then rule-based event detection. Time origin = first annotation of the log."""
    dcfg = DEFAULT_DET if dcfg is None else dcfg
    folder = Path(folder)
    ann = pd.read_feather(folder / ANNOTATION, columns=SENSOR_COLUMNS[ANNOTATION])
    pose = pd.read_feather(folder / POSE, columns=SENSOR_COLUMNS[POSE])
    t0_ns = int(ann.timestamp_ns.min())
    log_duration_s = (int(ann.timestamp_ns.max()) - t0_ns) / 1e9
    ann = ann[ann.category.isin(VEHICLE_CATEGORIES)].copy()
    ann = ann.sort_values(["timestamp_ns", "track_uuid"]).drop_duplicates(["track_uuid", "timestamp_ns"])
    pose = pose.sort_values("timestamp_ns").drop_duplicates("timestamp_ns")
    pose = pose.rename(columns={c: f"pose_{c}" for c in pose.columns})
    joined = pd.merge_asof(ann.sort_values("timestamp_ns"), pose,
                           left_on="timestamp_ns", right_on="pose_timestamp_ns",
                           direction="nearest", tolerance=int(pose_tolerance_ms * 1e6))
    unmatched = int(joined.pose_timestamp_ns.isna().sum())
    joined = joined.dropna(subset=["pose_timestamp_ns"]).copy()
    if len(ann) and unmatched / len(ann) > 0.05:
        raise ValueError(f"{folder.name}: >5% annotations lack an ego pose within tolerance")
    log_id = folder.name
    events: List[Event] = []
    universe = set()
    signals: Dict[Tuple[str, str], List[dict]] = {}
    max_dt_ms = 0.0
    if len(joined):
        ego = joined[["tx_m", "ty_m", "tz_m"]].to_numpy(float)
        quat = joined[["pose_qx", "pose_qy", "pose_qz", "pose_qw"]].to_numpy(float)
        trans = joined[["pose_tx_m", "pose_ty_m", "pose_tz_m"]].to_numpy(float)
        city = Rotation.from_quat(quat).apply(ego) + trans
        joined["cx"], joined["cy"] = city[:, 0], city[:, 1]
        max_dt_ms = float(np.max(np.abs(joined.timestamp_ns - joined.pose_timestamp_ns)) / 1e6)
        for track_id, df in joined.groupby("track_uuid", sort=True):
            tt = (df.timestamp_ns.to_numpy(np.int64) - t0_ns) / 1e9
            xx, yy = df.cx.to_numpy(), df.cy.to_numpy()
            if len(tt) < dcfg.window or np.ptp(tt) < dcfg.min_segment_s:
                continue
            if np.max(np.hypot(xx - xx[0], yy - yy[0])) < dcfg.min_displacement_m:
                continue
            key = (log_id, str(track_id))
            evs, segs = extract_events(tt, xx, yy, log_id, str(track_id), dcfg)
            if not segs:
                continue
            universe.add(key)
            signals[key] = segs
            events.extend(evs)
    info = {"log_id": log_id, "log_duration_s": round(log_duration_s, 2),
            "vehicle_annotation_rows": int(len(ann)),
            "pose_match_rate": round(1 - unmatched / max(len(ann), 1), 6),
            "max_pose_time_difference_ms": round(max_dt_ms, 3),
            "vehicle_tracks": int(ann.track_uuid.nunique()),
            "moving_tracks_evaluated": len(universe),
            "events": len(events),
            "events_by_type": dict(Counter(e.type for e in events))}
    return events, info, universe, signals


def valid_pair(folder: Path) -> bool:
    import pyarrow.feather as feather
    try:
        for name, cols in SENSOR_COLUMNS.items():
            p = Path(folder) / name
            if not p.is_file() or p.stat().st_size < 128:
                return False
            tab = feather.read_table(str(p), columns=cols)
            if tab.num_rows == 0:
                return False
        return True
    except Exception:
        return False


def _s3_client():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    return boto3.client("s3", config=Config(signature_version=UNSIGNED, region_name="us-east-1",
                                            connect_timeout=15, read_timeout=90,
                                            retries={"max_attempts": 3},
                                            max_pool_connections=32))


def list_s3_logs(split: str) -> List[str]:
    client = _s3_client()
    prefix = f"datasets/av2/sensor/{split}/"
    ids, token = [], None
    while True:
        kw = dict(Bucket="argoverse", Prefix=prefix, Delimiter="/", MaxKeys=1000)
        if token:
            kw["ContinuationToken"] = token
        resp = client.list_objects_v2(**kw)
        for cp in resp.get("CommonPrefixes", []):
            name = cp["Prefix"].rstrip("/").split("/")[-1]
            if UUID_RE.match(name):
                ids.append(name)
        token = resp.get("NextContinuationToken")
        if not token:
            break
    return sorted(set(ids))


def ensure_av2_logs(cache_root: Path, split: str = "val", max_logs: int = 150,
                    workers: int = 16, log_ids: Optional[Sequence[str]] = None):
    """Reuse cached log folders; download only the two Feather files per missing log
    from the official public AV2 bucket. Never falls back to synthetic data."""
    cache_dir = Path(cache_root) / "s3" / split
    cache_dir.mkdir(parents=True, exist_ok=True)
    note = None
    if log_ids:
        ids = list(log_ids)
    else:
        try:
            ids = list_s3_logs(split)
        except Exception as exc:  # offline: use whatever valid cache exists
            note = f"S3 listing failed ({type(exc).__name__}: {exc}); using cache only"
            ids = sorted(p.name for p in cache_dir.iterdir() if p.is_dir() and valid_pair(p))
    ids = ids[:max_logs]
    client = _s3_client()

    def fetch(log_id: str):
        folder = cache_dir / log_id
        if valid_pair(folder):
            return log_id, folder, "cache", None
        folder.mkdir(parents=True, exist_ok=True)
        try:
            for name in (ANNOTATION, POSE):
                client.download_file("argoverse", f"datasets/av2/sensor/{split}/{log_id}/{name}",
                                     str(folder / name))
            return (log_id, folder, "s3", None) if valid_pair(folder) else (log_id, None, "invalid", "bad files")
        except Exception as exc:
            return log_id, None, "error", f"{type(exc).__name__}: {exc}"

    results = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for r in pool.map(fetch, ids):
            results.append(r)
    folders = [f for _, f, _, _ in results if f is not None]
    if not folders:
        raise RuntimeError("No valid AV2 Sensor log pairs available (network and cache both failed).")
    info = {"split": split, "requested": len(ids), "ready": len(folders),
            "from_cache": sum(r[2] == "cache" for r in results),
            "downloaded": sum(r[2] == "s3" for r in results),
            "failed": [(r[0], r[3]) for r in results if r[1] is None],
            "note": note}
    return sorted(folders), info


# ---------------------------------------------------------------------------
# Visual verification helpers
# ---------------------------------------------------------------------------
TYPE_COLORS = {"TURN": "#1f77b4", "DECELERATE": "#d62728", "ACCELERATE": "#2ca02c", "STOP": "#7f7f7f"}


def plot_track(key, events: Sequence[Event], segments: Sequence[dict], ax_row, title_extra: str = ""):
    ax_xy, ax_v, ax_w = ax_row
    for seg in segments:
        ax_xy.plot(seg["x"], seg["y"], color="#bbbbbb", lw=1.5)
        ax_v.plot(seg["t"], seg["speed"], color="black", lw=1)
        ax_w.plot(seg["t"], seg["yaw_rate_deg_s"], color="black", lw=1)
        ax_xy.plot(seg["x"][0], seg["y"][0], "o", color="black", ms=4)
    for e in events:
        c = TYPE_COLORS[e.type]
        target = ax_w if e.type == "TURN" else ax_v
        target.axvspan(e.start, e.end, color=c, alpha=0.25)
        for seg in segments:
            m = (seg["t"] >= e.start) & (seg["t"] < e.end)
            if m.any():
                ax_xy.plot(seg["x"][m], seg["y"][m], color=c, lw=3)
    ax_xy.set_aspect("equal", adjustable="datalim")
    ax_xy.set_title(f"{key[0][:8]} / {key[1][:8]} {title_extra}", fontsize=9)
    ax_v.set_ylabel("speed (m/s)")
    ax_w.set_ylabel("yaw rate (deg/s)")
    ax_v.set_xlabel("time since log start (s)")
    ax_w.set_xlabel("time since log start (s)")
    labels = ", ".join(f"{e.type}[{e.start:.1f}-{e.end:.1f}{'*' if e.clipped else ''}]" for e in events)
    ax_v.set_title(labels, fontsize=7)