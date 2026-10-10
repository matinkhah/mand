# MAND: Modal-Aware Neural Distillation

Code for the paper **"Modal-Aware Neural Distillation: Teaching Compact Edge Intrusion Detectors the Extremes of Host History"** (S. Mojtaba Matinkhah, Yazd University).

## In 60 seconds

**Problem.** A small intrusion detector on an IoT gateway is usually trained to imitate a large cloud teacher one flow at a time. That ignores what the same host did a moment ago.

**Idea.** For each flow, look at the teacher's attack probability over the host's recent flows and ask the student to also reproduce the **minimum** ("were *all* recent flows suspicious?") and the **maximum** ("was *at least one*?"). These are two extra loss terms on top of ordinary distillation.

**Why it is cheap.** The targets come from the teacher's outputs, so the deployed student is the same plain per-flow MLP as in standard distillation. It needs no graph, no buffer and no extra input at inference.

**What it buys (macro-F1, mean over 10 teacher seeds):**

| Method | MQTT-IoT-IDS2020 | TON_IoT | IoT-23 (primary) |
|---|---|---|---|
| Teacher (context-aware) | 0.965 | 0.951 | 0.938 |
| Standard KD | 0.941 | 0.934 | 0.915 |
| iid KD | 0.939 | 0.932 | 0.913 |
| **MAND** | **0.949** | **0.942** | **0.924** |

- Median paired gain over standard KD is +0.0074 / +0.0079 / +0.0087 macro-F1, with 95% block-bootstrap intervals above zero on all three datasets.
- MAND closes 33% / 47% / 39% of the remaining student-teacher gap.
- False-negative rate drops, for example 0.089 to 0.078 on IoT-23.
- Edge cost is unchanged: about 1.28x10^4 parameters and 1.31 ms per flow on a Raspberry Pi 5. Training takes about 5% longer.

## What the evidence does and does not show

The gains are real but modest (roughly 0.7 to 1.0 macro-F1 points). Please read them with these points in mind.

- **Controls.** MAND beats a shuffled-window control (same host, wrong flows) on all three datasets, and beats per-flow squared-error and window-mean controls on IoT-23 and TON_IoT. On MQTT-IoT-IDS2020 the gain over the window-mean control has a 95% interval that just touches zero, so we do not count it as a win.
- **Not new information.** Every target is a function of the teacher's outputs. MAND works by reshaping the loss, not by adding data.
- **Not the best possible accuracy.** A student that consumes window-aggregated inputs scores 0.002 to 0.004 higher but needs window features at inference and costs about 1.8x the latency and 1.9x the parameters.
- **Applicability depends on the data.** MAND needs a teacher clearly better than the student, and hosts whose windows hold several flows and sometimes both classes. A different selection of IoT-23 captures failed our screen (62.7% of attack records lacked a duration, and the teacher left no headroom), so MAND had nothing to add there. IoT-23 results depend on which scenarios are used.
- **Scope.** Binary labels, same-host windows, three datasets, ten teacher seeds.

## Reproduce the results

Everything runs from one resumable script, `mand_reproduce.py`. Finished runs are cached and never recomputed, and the script prints `ALL STAGES COMPLETE` when done.

**Colab:** upload `mand_reproduce.py`, set Runtime to GPU, then run

```python
%run mand_reproduce.py
```

**Profiles** (set `PROFILE` at the top of the script):

- `"paper"`: the full experimental setting. This needs substantial GPU time; the earlier README estimated about 800 GPU-hours across the three datasets.
- `"smoke"`: a tiny pipeline test (2 epochs, 2 teacher seeds). It only checks that your environment works and is **not** evidence for any claim.

### Data

Datasets are not bundled. Download them from their public sources.

- **IoT-23** (`conn.log.labeled`). The paper uses these captures: `CTU-IoT-Malware-Capture-7-1`, `-9-1`, `-43-1`, `-48-1`, `-60-1`, and `CTU-Honeypot-Capture-4-1`, `-5-1`. The paper's appendix gives the preprocessing details.
- **MQTT-IoT-IDS2020** and **TON_IoT** need a preprocessed CSV with columns `cap, host, ts, duration, label` plus numeric features. Place the files and set `RUN_DATASETS`.
- Edge-IIoTset was only a reserve candidate (0.4% mixed windows) and is not used.

### What the script writes

| File | Content |
|---|---|
| `manifest.json` | Paper constants, seeds, hyperparameter grids, library versions, GPU info, SHA-256 of the script and data files |
| `results.jsonl` | One line per run: dataset, split, method, seeds, hyperparameters, macro-F1, FNR, fidelity, training time |
| `teacher_*.npz` | Cross-fitted teacher logits (float32) |
| `report.md` / `report.json` | Verification report: applicability checks, result tables next to the paper's numbers, statistical tests, ablations, cost, PASS/FAIL checks |

### Seeds

- Teacher seeds 11 to 20; paired student seeds 301 to 320.
- Tuning students 401 to 405, with reserved teachers 1 and 2.
- Host-held-out teachers 101 to 110 and students 201 to 220.
- Seeds for bootstrap, minimum-detectable-effect simulation, hyperparameter search, host split and calibration are recorded in the `ASSUMED` section of `manifest.json`.

## Built-in checks

The script verifies the following and aborts if any check fails.

- The exact signed-rank distribution matches SciPy.
- The soft min/max operators satisfy their stated limits.
- Every training flow is an anchor exactly once per epoch.
- Loss weights sum to one.
- Each closure contains all window members.
- No window crosses a train/validation/test split, and guard gaps are at least the window length.
- Cross-fit folds do not leak across boundaries.
- Determinism is checked and reported per device.

## Method in brief

- **Window.** The most recent same-host flows ending at most Delta seconds before the current flow (causal, computable online).
- **Targets.** Soft minimum and soft maximum (log-mean-exp, which is non-expansive) of the teacher's attack probability over the window.
- **Loss.** Cross-entropy + KD + a ramped squared error on the two targets, with the temperature annealed from 1 toward a small value.
- **Honest targets.** Teacher outputs used for training are cross-fitted over contiguous time folds, so no training target comes from a teacher that saw that flow.
- **Statistics.** Ten teacher seeds are the unit of analysis, with exact one-sided Wilcoxon tests and paired block-bootstrap intervals.

## Citation

The manuscript is being prepared for the *IEEE Internet of Things Journal*. Please update this entry when it is published.

```bibtex
@unpublished{matinkhah2026mand,
  title  = {Modal-Aware Neural Distillation: Teaching Compact Edge Intrusion Detectors the Extremes of Host History},
  author = {Matinkhah, S. Mojtaba},
  year   = {2026},
  note   = {Manuscript in preparation, IEEE Internet of Things Journal}
}
```

## License

Apache-2.0

## Contact

Questions about reproduction: please open an issue. Corresponding author: S. Mojtaba Matinkhah, matinkhah@yazd.ac.ir.
