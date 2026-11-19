"""Run: python -m pytest -q test_tqp_pipeline.py"""
import itertools
import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

import tqp_teg as T
import tqp_nl as N


def ev(name, kind, s, e, cs=False, ce=False, track="trk"):
    return T.Event(name, "log", track, kind, float(s), float(e), cs, ce)


def rel(a, b):
    return T.possible_relations(a, b)


# ---------------------------------------------------------------- relation semantics
def test_unclipped_semantics_match_v01():
    a, b = ev("a", "DECELERATE", 1, 3), ev("b", "TURN", 5, 7)
    assert rel(a, b) == {"THEN"} and rel(b, a) == {"NONE"}
    assert rel(ev("c", "TURN", 1, 3), ev("d", "TURN", 2.5, 5)) == {"THEN"}       # 0.5 s overlap tolerated
    assert rel(ev("c", "TURN", 1, 3), ev("d", "TURN", 8, 10)) == {"THEN"}        # gap exactly 5 s
    assert rel(ev("c", "TURN", 1, 3), ev("d", "TURN", 8.01, 10)) == {"NONE"}
    w1, w2 = ev("w1", "DECELERATE", 1, 4), ev("w2", "TURN", 1.5, 3.5)
    assert rel(w1, w2) == {"WHILE"} and rel(w2, w1) == {"WHILE"}
    assert rel(w1, ev("p", "TURN", 3, 6)) == {"PARTIAL"}


def test_real_case_clipped_turn_then_decelerate_is_positive():
    """Track 8a04afdc in the first AV2 run: TURN clipped at start, DECELERATE later."""
    turn = ev("turn", "TURN", 0.0, 3.4, cs=True)
    dec = ev("dec", "DECELERATE", 6.9, 8.0)
    acc = ev("acc", "ACCELERATE", 1.1, 2.6)
    assert T.pair_status(rel(turn, dec), "THEN") == "POSITIVE"
    assert T.pair_status(rel(dec, turn), "THEN") == "NEGATIVE"
    assert T.pair_status(rel(acc, turn), "WHILE") == "POSITIVE"
    assert T.pair_status(rel(turn, acc), "THEN") == "NEGATIVE"


def test_clipped_end_that_matters_is_uncertain():
    dec = ev("dec", "DECELERATE", 3.0, 4.5, ce=True)   # may continue past observation
    turn = ev("turn", "TURN", 6.0, 8.0)
    poss = rel(dec, turn)
    assert "THEN" in poss and len(poss) > 1
    assert T.pair_status(poss, "THEN") == "UNCERTAIN"


def test_grid_is_sound_against_random_completions():
    rng = np.random.default_rng(7)
    misses = 0
    for _ in range(300):
        flags = rng.random(4) < 0.4
        s1, s2 = rng.uniform(0, 12, 2)
        a = ev("a", "TURN", s1, s1 + rng.uniform(0.3, 4), flags[0], flags[1])
        b = ev("b", "DECELERATE", s2, s2 + rng.uniform(0.3, 4), flags[2], flags[3])
        poss = rel(a, b)
        for _ in range(60):
            ext = rng.exponential(3.0, 4) * flags
            code = int(T._label_codes(a.start - ext[0], a.end + ext[1],
                                      b.start - ext[2], b.end + ext[3], T.DEFAULT_REL))
            misses += T.LABELS[code] not in poss
    assert misses == 0


# ---------------------------------------------------------------- executors and graph
def random_events(seed, n_tracks=15, per_track=10):
    rng = np.random.default_rng(seed)
    out = []
    for t in range(n_tracks):
        for i in range(per_track):
            s = float(rng.uniform(0, 15))
            d = float(rng.uniform(0.3, 4))
            out.append(T.Event(f"e{t}_{i}", "L", str(t), str(rng.choice(T.EVENT_TYPES)), s, s + d,
                               bool(rng.random() < 0.15), bool(rng.random() < 0.15)))
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_graph_equals_interval_statuses_and_matches(seed):
    events = random_events(seed)
    universe = {("L", str(t)) for t in range(20)}           # includes tracks without events
    for cfg in (T.RelationConfig(), T.RelationConfig(then_max_gap_s=2.0, while_min_ratio=0.5)):
        g = T.TemporalEventGraph(events, universe, cfg)
        ge, ie = T.GraphQueryExecutor(g), T.IntervalExecutor(events, universe, cfg)
        for q in T.all_queries():
            assert ie.statuses(q) == ge.statuses(q), q
            assert ie.retrieve(q) == ge.retrieve(q), q


def test_verify_and_benchmark_report():
    events = random_events(5)
    rep, g = T.verify_and_benchmark(events, set(), T.all_queries(), repeat=2, build_repeats=1)
    assert rep["all_status_equal"] and rep["all_retrieval_equal"]
    assert rep["query_count"] == 32 and rep["graph"]["event_nodes"] == len(events)
    assert set(g.edges_frame().columns) >= {"source_event_id", "relation", "target_event_id"}


def test_parse_tqp_reversal_and_errors():
    plan = {"events": {"e1": "TURN", "e2": "DECELERATE"}, "relation": {"op": "THEN", "args": ["e2", "e1"]}}
    assert T.parse_tqp(plan) == T.Query("DECELERATE", "THEN", "TURN")
    for bad in ({"events": {"e1": "TURN", "e2": "TURN"}, "relation": {"op": "THEN", "args": ["e1", "e1"]}},
                {"events": {"e1": "TURN", "e2": "LANE_CHANGE"}, "relation": {"op": "THEN", "args": ["e1", "e2"]}},
                {"events": {"e1": "TURN", "e2": "STOP"}, "relation": {"op": "BEFORE", "args": ["e1", "e2"]}},
                {"events": {"e1": "TURN", "e2": "STOP"}, "relation": {"op": "THEN", "args": ["e1", "e2"]}, "x": 1}):
        with pytest.raises(ValueError):
            T.parse_tqp(bad)


# ---------------------------------------------------------------- detector and AV2 loader
def synthetic_track(label, duration=12.0, dt=0.1, t_offset=0.0):
    t = np.round(np.arange(0, duration + 1e-9, dt), 5)
    speed, yaw = np.full(t.shape, 10.0), np.zeros_like(t)
    decel, turn = {"before": ((1, 3), (5, 8)), "after": ((5, 7), (1, 4)),
                   "while": ((3.5, 5.7), (3, 6)), "constant": (None, None)}[label]
    for i in range(1, len(t)):
        speed[i] = max(3.0, speed[i - 1] - 3.0 * dt) if decel and decel[0] <= t[i] < decel[1] else speed[i - 1]
        yaw[i] = yaw[i - 1] + (np.deg2rad(20) * dt if turn and turn[0] <= t[i] < turn[1] else 0.0)
    x = np.r_[0., np.cumsum((speed * np.cos(yaw))[:-1] * dt)]
    y = np.r_[0., np.cumsum((speed * np.sin(yaw))[:-1] * dt)]
    return t + t_offset, x, y


def test_detector_relations_on_synthetic_motion():
    events, universe = [], set()
    for label in ("before", "after", "while", "constant"):
        evs, segs = T.extract_events(*synthetic_track(label), "m", label)
        events += evs
        universe.add(("m", label))
        assert segs
    ge = T.GraphQueryExecutor(T.TemporalEventGraph(events, universe))
    st = ge.statuses(T.Query("DECELERATE", "THEN", "TURN"))
    assert st[("m", "before")] == "POSITIVE"
    assert st[("m", "after")] == "NEGATIVE"
    assert st[("m", "constant")] == "NEGATIVE"
    assert ge.statuses(T.Query("TURN", "THEN", "DECELERATE"))[("m", "after")] == "POSITIVE"
    assert ge.statuses(T.Query("DECELERATE", "WHILE", "TURN"))[("m", "while")] == "POSITIVE"


def test_av2_loader_transform_and_log_relative_time(tmp_path, monkeypatch):
    log = tmp_path / "test_log"
    log.mkdir()
    t, x, y = synthetic_track("before", t_offset=0.0)
    log_t0 = 315968867659956000
    track_offset_ns = int(2e9)                     # track appears 2 s after log start
    ns = (t * 1e9).astype(np.int64) + log_t0 + track_offset_ns
    rot = Rotation.from_euler("z", 90, degrees=True)
    shift = np.array([500., 700., 1.])
    ego = rot.inv().apply(np.column_stack((x, y, np.zeros_like(x))) - shift)
    first_row = pd.DataFrame({"timestamp_ns": [log_t0], "track_uuid": ["ped"], "category": ["PEDESTRIAN"],
                              "tx_m": [0.0], "ty_m": [0.0], "tz_m": [0.0]})
    ann = pd.concat([first_row, pd.DataFrame({"timestamp_ns": ns, "track_uuid": "car", "category": "REGULAR_VEHICLE",
                                              "tx_m": ego[:, 0], "ty_m": ego[:, 1], "tz_m": ego[:, 2]})])
    q = rot.as_quat()
    pose_ns = np.r_[log_t0, ns]
    pose = pd.DataFrame({"timestamp_ns": pose_ns, "qw": q[3], "qx": q[0], "qy": q[1], "qz": q[2],
                         "tx_m": shift[0], "ty_m": shift[1], "tz_m": shift[2]})
    monkeypatch.setattr(pd, "read_feather",
                        lambda p, columns: (ann if str(p).endswith(T.ANNOTATION) else pose)[columns].copy())
    events, info, universe, signals = T.load_av2_log(log)
    assert info["pose_match_rate"] == 1.0 and ("test_log", "car") in universe
    assert {e.type for e in events} == {"TURN", "DECELERATE"}
    assert min(e.start for e in events) >= 2.0     # log-relative time base
    ge = T.GraphQueryExecutor(T.TemporalEventGraph(events, universe))
    assert ge.statuses(T.Query("DECELERATE", "THEN", "TURN"))[("test_log", "car")] == "POSITIVE"


def test_ensure_logs_uses_cache_when_offline(tmp_path, monkeypatch):
    lid = "01bb304d-7bd8-35f8-bbef-7086b688e35e"
    (tmp_path / "s3" / "val" / lid).mkdir(parents=True)

    def offline(split):
        raise ConnectionError("offline")

    monkeypatch.setattr(T, "list_s3_logs", offline)
    monkeypatch.setattr(T, "valid_pair", lambda folder: True)
    monkeypatch.setattr(T, "_s3_client", lambda: None)
    folders, info = T.ensure_av2_logs(tmp_path, "val", max_logs=5)
    assert [f.name for f in folders] == [lid] and info["from_cache"] == 1 and info["note"]


# ---------------------------------------------------------------- detector v0.3 changes
def test_close_gaps_merges_only_interior_short_gaps():
    m = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0], bool)
    out = T.close_gaps(m, 2)
    assert out.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0]
    assert T.close_gaps(m, 0).tolist() == m.tolist()


def test_fragmented_acceleration_is_merged():
    dt = 0.1
    t = np.round(np.arange(0, 12 + 1e-9, dt), 5)
    speed = np.full(t.shape, 3.0)
    for i in range(1, len(t)):
        a = 2.0 if (3 <= t[i] < 5) or (5.9 <= t[i] < 7.5) else 0.0     # 0.9 s pause in between
        speed[i] = speed[i - 1] + a * dt
    x = np.r_[0., np.cumsum(speed[:-1] * dt)]
    y = np.zeros_like(x)
    ev_v2, _ = T.extract_events(t, x, y, "L", "trk", T.DETECTOR_V02)
    ev_v3, _ = T.extract_events(t, x, y, "L", "trk")
    assert sum(e.type == "ACCELERATE" for e in ev_v2) == 2
    assert sum(e.type == "ACCELERATE" for e in ev_v3) == 1


def test_short_segment_and_edge_trim():
    t, x, y = synthetic_track("before", duration=2.0)
    assert T.extract_events(t, x, y, "L", "short")[0] == []          # < 3 s segment skipped
    t, x, y = synthetic_track("before")
    evs, _ = T.extract_events(t, x, y, "L", "trk")
    assert all(e.start >= 0.5 - 1e-9 for e in evs if e.type != "STOP")


def test_gentle_deceleration_detected_only_in_v03():
    dt = 0.1
    t = np.round(np.arange(0, 12 + 1e-9, dt), 5)
    speed = np.where(t < 3, 8.0, np.where(t < 8, 8.0 - 0.8 * (t - 3), 4.0))   # 0.8 m/s^2 for 5 s
    x = np.r_[0., np.cumsum(speed[:-1] * dt)]
    y = np.zeros_like(x)
    assert not any(e.type == "DECELERATE" for e in T.extract_events(t, x, y, "L", "g", T.DETECTOR_V02)[0])
    assert any(e.type == "DECELERATE" for e in T.extract_events(t, x, y, "L", "g")[0])


def test_detector_quality_counts():
    evs = [ev("a", "STOP", 1, 2), ev("b", "STOP", 2.5, 4), ev("c", "TURN", 1, 5)]
    q = T.detector_quality(evs, {("log", "trk")})
    assert q["consecutive_same_type_pairs"] == 1 and q["same_type_gap_le_1s"] == 1


# ---------------------------------------------------------------- NL layer: P (TQP) and B2
def test_diagnostic_set_gold_and_no_leakage():
    df = N.build_diagnostic_set()
    assert len(df) == 72
    row = df[df.sentence == "A vehicle turns after slowing down."].iloc[0]
    assert (row.gold_first, row.gold_relation, row.gold_second) == ("DECELERATE", "THEN", "TURN")
    assert row.first_mentioned == "TURN" and row.connective == "after"
    for _, r in df.iterrows():
        q = T.parse_tqp(json.loads(r.gold_tqp))
        assert N.score_tqp(q, q)["exact"]


def test_tqp_parsing_and_scoring():
    gold = T.Query("DECELERATE", "THEN", "TURN")
    good = '```json\n{"events": {"e1": "TURN", "e2": "DECELERATE"}, "relation": {"op": "THEN", "args": ["e2", "e1"]}}\n```'
    q, _, err = N.parse_prediction(good)
    assert err is None and N.score_tqp(gold, q)["exact"]
    q, _, _ = N.parse_prediction('{"events": {"e1": "TURN", "e2": "DECELERATE"}, "relation": {"op": "THEN", "args": ["e1", "e2"]}}')
    s = N.score_tqp(gold, q)
    assert s["relation_correct"] and s["events_correct"] and not s["direction_correct"] and not s["exact"]
    q, _, err = N.parse_prediction("THEN")
    assert q is None and err.startswith("json")
    assert N.score_tqp(T.Query("DECELERATE", "WHILE", "TURN"), T.Query("TURN", "WHILE", "DECELERATE"))["exact"]


def test_probes_gold_labels():
    gold = T.Query("DECELERATE", "THEN", "TURN")
    labels = {n: g for n, _, _, g in N.build_probes(gold)}
    assert labels == {"then_xy": True, "then_yx": False, "while": False, "far": False,
                      "only_x": False, "only_y": False, "other": False}
    wl = {n: g for n, _, _, g in N.build_probes(T.Query("TURN", "WHILE", "ACCELERATE"))}
    assert wl["while"] and not wl["then_xy"] and not wl["then_yx"]


@pytest.mark.parametrize("q", [T.Query("DECELERATE", "THEN", "TURN"), T.Query("TURN", "WHILE", "ACCELERATE")])
def test_b2_reference_code_is_correct_on_probes(q):
    probes = N.build_probes(q)
    accepts, err = N.run_b2(N.b2_reference_code(q), probes)
    assert err is None
    assert N.probe_score(q, accepts, probes)["probe_exact"]
    assert N.score_b2(q, accepts)["exact"]
    tqp_acc = N.tqp_probe_accepts(q, probes)
    assert N.probe_score(q, tqp_acc, probes)["probe_exact"]


def test_b2_reversed_code_scored_as_direction_error():
    gold = T.Query("DECELERATE", "THEN", "TURN")
    wrong = N.b2_reference_code(T.Query("TURN", "THEN", "DECELERATE"))
    accepts, err = N.run_b2(wrong, N.build_probes(gold))
    s = N.score_b2(gold, accepts)
    assert err is None and s["relation_correct"] and not s["direction_correct"] and not s["exact"]


def test_b2_sandbox_rejects_unsafe_and_broken_code():
    probes = N.build_probes(T.Query("DECELERATE", "THEN", "TURN"))
    for bad in ("```python\nimport os\ndef matches(events):\n    return True\n```",
                "```python\ndef matches(events):\n    return open('x')\n```",
                "```python\ndef matches(events):\n    return events.__class__\n```",
                "```python\ndef matches(events):\n    while True:\n        pass\n```",
                "```python\ndef matches(events):\n    return events[0]['type'] ==\n```",
                "no code here"):
        accepts, err = N.run_b2(bad, probes, timeout_s=0.2)
        assert accepts is None and err, bad


def test_messages_and_stats():
    for v in N.VARIANTS:
        m1, m2 = N.tqp_messages("A vehicle stops after turning.", v), N.b2_messages("A vehicle stops after turning.", v)
        assert len(m1) == len(m2) == 2 + 2 * len(N.VARIANTS[v]["examples"])
    lo, hi = N.wilson(5, 12)
    assert 0.18 < lo < 0.2 and 0.67 < hi < 0.69
    r = N.mcnemar_exact([True] * 10 + [False] * 2, [False] * 10 + [True] * 2)
    assert r["P_only_correct"] == 10 and r["B2_only_correct"] == 2 and r["p_value"] < 0.05