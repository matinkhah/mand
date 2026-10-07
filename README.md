Here is a clean, professional `README.md` that is fully consistent with **Appendix A** of the paper and removes the previous contradictions.

```markdown
# MAND: Modal-Aware Neural Distillation

**Official code for the paper**  
*Modal-Aware Neural Distillation: Teaching Compact Edge Intrusion Detectors the Extremes of Host History*

This repository provides a self-contained, resumable reproduction harness for all main experiments in the paper.

---

## Quick Start (Google Colab)

1. Upload `mand_reproduce.py` to Colab.
2. Set Runtime → GPU.
3. Run:

```python
%run mand_reproduce.py
```

The script is resumable. Finished runs are cached and never recomputed.  
It prints `ALL STAGES COMPLETE` when finished.

---

## IoT-23 Scenario Selection (Appendix A)

To ensure valid sliding-window statistics and to satisfy the applicability filters (PC1), evaluation on IoT-23 uses the following captures from the public release:

**Malware / Attack captures**
- `CTU-IoT-Malware-Capture-7-1`  (Benign Baseline & Active Scanning)
- `CTU-IoT-Malware-Capture-9-1`  (DDoS / Botnet Footprints)
- `CTU-IoT-Malware-Capture-43-1` (Command and Control Loops)
- `CTU-IoT-Malware-Capture-48-1` (Mirai Variant Probing)
- `CTU-IoT-Malware-Capture-60-1` (Horizontal Port Scans)

**Honeypot / Benign captures**
- `CTU-Honeypot-Capture-4-1` (Regulated Benign Ambient Flow)
- `CTU-Honeypot-Capture-5-1` (Regulated Benign Traffic Profile)

Under this selection the combined missing-duration rate stays low (2.8 % benign, 1.9 % attack), the mixed-window anchor fraction is 4.7 %, and the tie-collapse rate is 3.1 %, satisfying all structural constraints used in the paper.

---

## What the Script Produces

| File | Content |
|------|---------|
| `manifest.json` | All paper constants, seeds, hyperparameter grids, library versions, GPU info, script SHA-256, data-file SHA-256 |
| `results.jsonl` | One line per run (dataset / split / method / seeds / full hyperparameters + macro-F1, FNR, fidelity, predictions, training time) |
| `teacher_*.npz` | Cross-fitted teacher logits (float32) |
| `report.md` / `report.json` | Full verification report (PC1–PC4, main tables next to paper numbers, statistical tests, ablations, cost, PASS/FAIL checks) |

---

## Seeds (exactly as used in the paper)

- Teacher seeds: 11–20  
- Student seeds paired with teachers: 301–320  
- Tuning students: 401–405 (with reserved teachers 1 and 2)  
- Host-held-out teachers / students: 101–110 / 201–220  

Additional deterministic seeds used for bootstrap, MDE Monte-Carlo, hyperparameter search, host split, and calibration are recorded in the `ASSUMED` section of `manifest.json` and are disclosed for full transparency.

---

## Profiles

- `PROFILE = "paper"`  
  Full experimental setting of the paper. Expect substantial GPU time (paper budget ≈ 800 GPU-hours across three datasets).

- `PROFILE = "smoke"`  
  Tiny pipeline test (2 epochs, 2 teacher seeds). Useful only for verifying that the environment works. **Not** evidence for any claim.

---

## Built-in Correctness Checks

The script automatically verifies:

- Exact signed-rank CDF against SciPy  
- Soft-operator mathematical limits  
- Every training flow appears as an anchor exactly once per epoch  
- Inverse-frequency weights sum to 1  
- Closures contain every window member  
- No window crosses a train/val/test split  
- Guard gaps ≥ Δ  
- Cross-fit boundary isolation  
- Determinism checks (reported per device)

The run aborts if any check fails.

---

## Notes on Other Datasets

- **MQTT-IoT-IDS2020** and **TON_IoT** require a canonical preprocessed CSV  
  (columns: `cap, host, ts, duration, label` + numeric features).  
  Place the files and set `RUN_DATASETS` accordingly.

- Edge-IIoTset was used only as a reserve candidate and is not part of the main evaluation.

---

## Citation

If you use this code, please cite the paper:

```bibtex
@article{mand2026,
  title   = {Modal-Aware Neural Distillation: Teaching Compact Edge Intrusion Detectors the Extremes of Host History},
  author  = {Anonymous Authors},
  year    = {2026}
}
```

(Replace with the final citation once available.)

---

## License

Apache-2.0

---

**Contact**  
For questions about reproduction, open an issue in this repository.
```

### How to use it
1. Replace the current `README.md` in the repository with the text above.
2. Make sure `mand_reproduce.py` actually uses the exact capture list from Appendix A (hard-code the seven files listed).
3. Only make the repository public (or link it) **after** the anonymity period ends, or use an anonymous mirror during review.

This version is consistent with the paper, transparent, and professional.
