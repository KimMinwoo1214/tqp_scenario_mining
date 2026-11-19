# Temporal Query Plans for Natural-Language Driving Scenario Mining

Code, data, and results for a CSE4057 (Natural Language Processing, Hanyang University, Fall 2026) course project:
**"Temporal Query Plans for Natural-Language Driving Scenario Mining: Separating Temporal Interpretation from Judgment Implementation."**

A small language model (Qwen2.5-1.5B-Instruct) reads a driving-scenario query with a temporal connective
(e.g., *"A vehicle loses speed prior to rounding a corner"*) and either

- **P (TQP):** writes a Temporal Query Plan (strict JSON: two event types and a `THEN`/`WHILE` relation) that a fixed executor evaluates, or
- **B2 (direct Python):** writes a `matches(events)` function that implements the judgment itself.

Both are scored with the same probe set, and the plans are executed on a Temporal Event Graph built from Argoverse 2 (AV2) Sensor logs.

## Repository layout

```
src/                 Experiment package (frozen for the main run)
  tqp_teg.py           TQP schema/parser, Temporal Event Graph, graph and interval executors,
                       AV2 log loading, behavior detector v0.4
  tqp_nl.py            Prompts (zero-shot / balanced / counterbalanced) for P and B2, generation,
                       B2 sandbox, probe set v0.5, scoring, error labels
  tqp_eval.py          Development set (288), held-out templates (864), free-form loader, leakage check
  tqp_run.py           Resumable generation, scoring, retrieval on real logs
  tqp_stats.py         Relation-balanced accuracy, cluster bootstrap, sign-flip permutation, Holm, GEE
tests/               Engine unit tests (pilot)
data/                dev_set_288.csv, eval_templates_864.csv, eval_freeform_50.csv, freeform.csv (source)
notebooks/
  pilot/             Proposal-stage pilots (HW1)
    01_temporal_event_frequency.ipynb   Behavior / relation frequencies in 150 AV2 val logs
    05_TQP_TEG_AV2_v03_v04_Colab.ipynb  Detector v0.3 -> v0.4, development set, P vs. B2, prompt variants
    06_TQP_v05_checks.ipynb             Probe set v0.5 design and checks
    early/                              First TEG pilot, first TQP+TEG integration, four-layer end-to-end check
  main/              Main run (HW2)
    TQP_main_v1.ipynb                   Generation, scoring, H1/H2/H3/H3b, retrieval, manual-check sheets
    TQP_shuffle_v1.ipynb                Exploratory example-order sensitivity
results/
  pilot/             Pilot reports and summaries (10/14 to 10/17)
  main/              report_main.json (all statistics), scored_all.csv (per sentence x method x condition),
                     retrieval.csv, manifest.json (data/prompt hashes), train_log_ids.json,
                     shuffle/, manual_checks/ (detector precision 80 events, B2 inspection 40 programs)
figures/             Accuracy, accuracy by relation, error-type shares
```

## Version history

| Stage | Date (2026) | What changed |
|---|---|---|
| Frequency analysis | Oct 13 | Behavior and relation counts in AV2 val to choose behaviors and connectives |
| TEG pilot | Oct 14 | First event detector and event graph |
| Four-layer check | Oct 14 | Sentence -> TQP -> graph -> retrieval end to end |
| v0.3 / v0.4 | Oct 15-16 | Detector v0.4 (jitter filter, guarded gap closing), 288-sentence development set, P vs. B2, prompt variants |
| v0.5 checks | Oct 17 | Probe set v0.5 (5 interpretation + 8 implementation probes) |
| Main run v1.0 | Nov 17-18 | Held-out evaluation set (914 sentences), three prompt conditions, prespecified statistics, 150 random AV2 train logs, manual checks, example-order sensitivity |

## Main results (evaluation set, 914 sentences, 107 clusters)

Relation-balanced interpretation accuracy, `D = P - B2` with 95% cluster-bootstrap intervals:

| Condition | P | B2 | D [95% CI] | Holm-adjusted permutation p |
|---|---|---|---|---|
| Zero-shot | 0.003 | 0.014 | -0.011 [-0.022, -0.002] | 0.078 |
| Balanced | 0.599 | 0.456 | +0.144 [0.049, 0.238] | 0.078 |
| Counterbalanced | 0.431 | 0.333 | +0.099 [0.043, 0.150] | 0.198 |

- Method x condition interaction (logistic GEE): Wald chi^2 = 9.10, df = 2, p = 0.011.
- In incongruent sentences, every P direction error with correct event types was an `args` error or a double reversal.
- Interpretation errors change the retrieved (log, track) set almost completely (Jaccard about 0.01 to 0.02).
- Detector precision on 80 sampled events: 80/80. Probe scoring agreed with manual inspection on 39/40 B2 programs.
- Shuffling the example order alone moves D between about -0.01 and +0.21 (exploratory).

See `results/main/report_main.json` for all numbers.

## Reproducing the main run

1. Copy `src/*.py` and `data/freeform.csv` into Google Drive as `MyDrive/NLP/tqp_repo/` (with `freeform.csv` under `data/`), and put the notebooks there as well.
2. Open `notebooks/main/TQP_main_v1.ipynb` in Google Colab with a T4 GPU and run all cells.
   - Results are written to `MyDrive/NLP/AV2_TQP_TEG_Results/main_v1/`.
   - AV2 annotation files are downloaded from the public AV2 S3 bucket into `MyDrive/NLP/AV2_TEG_Pilot_Data/`.
   - Generation is cached per (split, method, condition), so an interrupted run resumes.
3. Optional: `notebooks/main/TQP_shuffle_v1.ipynb`.

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-1.5B-Instruct`, fp16, greedy decoding, batch 16, max new tokens P 96 / B2 384 |
| Seed | 2026 |
| Bootstrap / permutations | 10,000 / 10,000 |
| Runtime (Colab T4) | about 60-70 min generation, about 10 min retrieval |

Paths are set in the first notebook cell; change `REPO`, `OUT`, and `CACHE_ROOT` to run elsewhere.

## Data

Sentences in `data/` are template-generated or hand-written for this project. AV2 Sensor data are **not** included;
the notebooks download annotations directly from the official source. AV2 is released under CC BY-NC-SA 4.0.

## Notes

- The free-form sentences were drafted with an AI assistant and reviewed by the author; AI use is disclosed in the paper's AI Usage appendix.
- This is a course project, not a maintained library.
