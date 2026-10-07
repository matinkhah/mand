# Reproducing MAND (resumable harness)

`mand_reproduce.py` is one self-contained file. In Colab: upload it, set Runtime -> GPU, then `%run mand_reproduce.py`.
Re-run the same cell after every disconnect: finished runs are cached in `DRIVE_DIR/run_<profile>_<fingerprint>/results.jsonl`
and are never recomputed. It prints `ALL STAGES COMPLETE` when done.

## What the script writes (commit these to the public repository)
| file | content |
|---|---|
| `manifest.json` | every paper constant, seed, grid; every ASSUMED parameter; library versions; GPU; script SHA-256; SHA-256 of every data file |
| `results.jsonl` | one line per run: key = dataset / split / variant / teacher type / method / teacher seed / student seed / full hyperparameters; macro-F1 (val, test), FNR, fidelity, packed test predictions, test probabilities, training seconds |
| `teacher_*.npz` | cross-fitted teacher logits (float32) per teacher seed |
| `report.md`, `report.json` | PC1-PC4, tuned hyperparameters, main table next to the paper's numbers, HC/MDE, C1-C4, C2, AF, TOST, held-out conjunct, ablations, R1-R4, cost, PASS/FAIL against the paper |

## Seeds (all from the paper)
teacher 11-20, students 301-320 (teacher 11+i with 301+2i, 302+2i); tuning students 401-405 with reserved teachers 1 (seeds 401-403) and 2 (404-405);
held-out teachers 101-110, students 201-220. Seeds the paper does not give (bootstrap 2024, MDE Monte-Carlo 7, search RNG 1000+k, held-out host split 12345,
calibration student 7, noise 555+teacher seed) are in `ASSUMED` and must be disclosed.

## Profiles
* `PROFILE = "paper"`: every constant of the paper. Extrapolating from Colab T4 timings (about 25-50 s per student at 20 epochs), expect roughly 70-80 GPU-hours
  per dataset, i.e. many resumed sessions. The paper itself budgets about 800 GPU-hours over three datasets.
* `PROFILE = "smoke"`: tiny pipeline test (2 epochs, 2 teacher seeds). Not evidence about any claim.

## Built-in checks (run automatically; the script stops if any fails)
exact signed-rank CDF against scipy; soft-operator limits; every training flow is an anchor exactly once per epoch; weights 1/n_f sum to 1;
closures contain every window member; no window crosses a split; guard gaps >= Delta; cross-fit exclusion; same seed gives identical weights (determinism check is
reported per device; bitwise repeatability on GPU is not guaranteed by PyTorch and is reported, not assumed).

## What a reviewer must know
1. The paper does not say which IoT-23 scenarios were used. `CAPTURES` is a placeholder. With the public captures tried so far, PC1 fails (missing durations 22-99.95%),
   and the paper's rule is then "no confirmatory claim"; the script stops for that dataset unless `ALLOW_PC1_FAIL=True` (labelled EXPLORATORY).
2. MQTT-IoT-IDS2020 / TON_IoT / Edge-IIoTset: the paper's preprocessing is not specified precisely enough to rebuild. Supply a canonical CSV
   (columns `cap, host, ts, duration, label` + numeric features) and set `RUN_DATASETS`.
3. Not implemented: ablation 7 (kNN frame), Raspberry-Pi latency (Colab-CPU latency is reported and labelled), calibration/KL/FPR extras, dense/sparse recall,
   campaign-held-out and temporal-shift splits, Edge-IIoTset backup.
