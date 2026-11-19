# -*- coding: utf-8 -*-
"""Prespecified analyses for the main run (HW1 Section 5).

Unit of inference: sentence clusters (template x gold event order; free-form: gold TQP).
Clusters are relation-pure, so bootstrap resampling is stratified by relation.

- balanced_accuracy: Acc_bal = (A_THEN + A_WHILE) / 2
- paired_effect: D_c = Acc_bal(P, c) - Acc_bal(B2, c) with a 95% cluster-bootstrap interval
  (10,000 resamples) and a cluster-level sign-flip permutation test (10,000 permutations)
- holm: Holm adjustment of the three per-condition p-values (H1)
- gee_interaction: logistic GEE, correct ~ method * condition + relation, groups = clusters,
  exchangeable working correlation, weights 1 / (2 n_relation); Wald test of the interaction (H2)
- proportion_ci: cluster-bootstrap interval of a proportion among selected rows (H3, H3b)
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np
import pandas as pd

B_DEFAULT = 10_000
SEED = 2026


def balanced_accuracy(df: pd.DataFrame, col: str = "interp_correct") -> float:
    accs = [df.loc[df.gold_relation == r, col].mean() for r in ("THEN", "WHILE") if (df.gold_relation == r).any()]
    return float(np.mean(accs)) if accs else float("nan")


def _cluster_table(df: pd.DataFrame, a_mask, b_mask, col: str) -> pd.DataFrame:
    """Per cluster: relation, n sentences, sum correct for A and for B (paired on sentence id)."""
    a = df[a_mask].set_index("sid")[[col, "cluster", "gold_relation"]]
    b = df[b_mask].set_index("sid")[col].reindex(a.index)
    assert b.notna().all(), "unpaired sentences between the two cells"
    t = pd.DataFrame({"cluster": a.cluster, "relation": a.gold_relation,
                      "a": a[col].astype(float), "b": b.astype(float)})
    return t.groupby(["relation", "cluster"]).agg(n=("a", "size"), a=("a", "sum"), b=("b", "sum")).reset_index()


def _bal_from_sums(ct_list):
    """ct_list: list of (n_total, a_sum, b_sum) per relation present."""
    accs_a = [a / n for n, a, _ in ct_list]
    accs_b = [b / n for n, _, b in ct_list]
    return np.mean(accs_a, axis=0), np.mean(accs_b, axis=0)


def paired_effect(df: pd.DataFrame, a_mask, b_mask, col: str = "interp_correct",
                  B: int = B_DEFAULT, seed: int = SEED) -> dict:
    ct = _cluster_table(df, a_mask, b_mask, col)
    rng = np.random.default_rng(seed)
    groups = [g for _, g in ct.groupby("relation")]
    # point estimate
    sums = [(g.n.sum(), g.a.sum(), g.b.sum()) for g in groups]
    acc_a, acc_b = _bal_from_sums(sums)
    d = float(acc_a - acc_b)
    # stratified cluster bootstrap
    boot_a, boot_b = [], []
    for g in groups:
        n, a, b = g.n.to_numpy(float), g.a.to_numpy(float), g.b.to_numpy(float)
        idx = rng.integers(0, len(g), size=(B, len(g)))
        nn = n[idx].sum(1)
        boot_a.append(a[idx].sum(1) / nn)
        boot_b.append(b[idx].sum(1) / nn)
    bd = np.mean(boot_a, axis=0) - np.mean(boot_b, axis=0)
    lo, hi = np.percentile(bd, [2.5, 97.5])
    # sign-flip permutation on cluster-level paired differences
    stat = np.zeros(B)
    for g in groups:
        dc = (g.a - g.b).to_numpy(float) / g.n.sum()
        signs = rng.choice([-1.0, 1.0], size=(B, len(g)))
        stat += (signs * dc).sum(1) / len(groups)
    p = (1 + np.sum(np.abs(stat) >= abs(d) - 1e-12)) / (B + 1)
    return {"acc_a": float(acc_a), "acc_b": float(acc_b), "D": d, "ci_low": float(lo), "ci_high": float(hi),
            "p_perm": float(p), "n_sentences": int(ct.n.sum()), "n_clusters": int(len(ct)),
            "n_clusters_by_relation": {r: int(len(g)) for r, g in ct.groupby("relation")}}


def holm(pvals: Sequence[float]) -> np.ndarray:
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def accuracy_ci(df: pd.DataFrame, mask, col: str = "interp_correct", balanced: bool = True,
                B: int = B_DEFAULT, seed: int = SEED) -> dict:
    sub = df[mask]
    ct = sub.groupby(["gold_relation", "cluster"])[col].agg(["size", "sum"]).reset_index()
    rng = np.random.default_rng(seed)
    if balanced:
        point = balanced_accuracy(sub, col)
        boots = []
        for _, g in ct.groupby("gold_relation"):
            n, k = g["size"].to_numpy(float), g["sum"].to_numpy(float)
            idx = rng.integers(0, len(g), size=(B, len(g)))
            boots.append(k[idx].sum(1) / n[idx].sum(1))
        bs = np.mean(boots, axis=0)
    else:
        point = float(sub[col].mean())
        n, k = ct["size"].to_numpy(float), ct["sum"].to_numpy(float)
        idx = rng.integers(0, len(ct), size=(B, len(ct)))
        bs = k[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"acc": float(point), "ci_low": float(lo), "ci_high": float(hi), "n": int(len(sub))}


def proportion_ci(df: pd.DataFrame, denom_mask, num_col_or_mask, B: int = B_DEFAULT, seed: int = SEED) -> dict:
    """Share of rows satisfying num among rows satisfying denom; cluster bootstrap (not stratified)."""
    sub = df[denom_mask].copy()
    if isinstance(num_col_or_mask, str):
        sub["_num"] = sub[num_col_or_mask].astype(float)
    else:
        sub["_num"] = num_col_or_mask[denom_mask].astype(float).to_numpy()
    if len(sub) == 0:
        return {"share": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "k": 0, "n": 0}
    ct = sub.groupby("cluster")["_num"].agg(["size", "sum"])
    n, k = ct["size"].to_numpy(float), ct["sum"].to_numpy(float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ct), size=(B, len(ct)))
    bs = k[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return {"share": float(k.sum() / n.sum()), "ci_low": float(lo), "ci_high": float(hi),
            "k": int(k.sum()), "n": int(n.sum())}


def gee_interaction(df: pd.DataFrame, col: str = "interp_correct") -> dict:
    """Logistic GEE with method x condition interaction; Wald test of the interaction terms."""
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = df.copy()
    d["y"] = d[col].astype(float)
    n_rel = d.drop_duplicates("sid").gold_relation.value_counts()
    d["w"] = d.gold_relation.map(lambda r: 1.0 / (2 * n_rel[r]))
    d["w"] = d["w"] / d["w"].mean()           # rescale; GEE estimates are invariant to a constant factor
    d = d.sort_values(["cluster", "sid", "method", "condition"])
    model = smf.gee("y ~ C(method, Treatment('B2')) * C(condition, Treatment('zero_shot')) + C(gold_relation)",
                    groups="cluster", data=d, family=sm.families.Binomial(),
                    cov_struct=sm.cov_struct.Exchangeable(), weights=d["w"])
    res = model.fit(maxiter=200)
    names = list(res.params.index)
    inter = [i for i, n in enumerate(names) if ":" in n]
    R = np.zeros((len(inter), len(names)))
    for r, i in enumerate(inter):
        R[r, i] = 1.0
    wt = res.wald_test(R, scalar=True)
    return {"wald_chi2": float(np.squeeze(wt.statistic)), "df": int(len(inter)), "p_value": float(wt.pvalue),
            "params": res.params.round(4).to_dict(), "bse": res.bse.round(4).to_dict(),
            "n_obs": int(res.nobs), "n_clusters": int(d.cluster.nunique()),
            "dep_param": float(res.model.cov_struct.dep_params)}
