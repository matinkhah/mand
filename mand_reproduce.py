#@title MAND — full reproducibility harness (single resumable Colab script)
# =============================================================================
# Reproduces / validates MAND_revised (IoT-23 confirmatory chain + optional development datasets).
#
# HONEST SCOPE (printed again at run time):
#   * The paper's own budget is ~8,840 student runs + 312 teacher trainings (~800 GPU-hours).
#     No single Colab session can do that. Every run is cached in DRIVE_DIR, every stage is
#     resumable: just re-run the cell until it prints "ALL STAGES COMPLETE".
#   * Everything the paper specifies is in PAPER (constants, grids, seeds, thresholds).
#   * Everything the paper does NOT specify is in ASSUMED and is written to manifest.json together with
#     library versions, GPU, script hash and data-file hashes. Reviewers should read that ledger.
#   * Not implemented (listed in COVERAGE): ablation 7 (kNN frame), Raspberry-Pi latency,
#     calibration/KL/FPR extras, secondary splits other than host-held-out.
#   * MQTT-IoT-IDS2020 / TON_IoT / Edge-IIoTset: the paper does not specify their preprocessing precisely
#     enough to rebuild it. Provide a canonical CSV (see load_generic) and the harness treats it identically.
# =============================================================================
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # determinism; must precede torch import
import sys, re, time, math, json, base64, hashlib, subprocess, warnings, platform, itertools, random
from types import SimpleNamespace
import numpy as np
import pandas as pd
import requests
try:
    import numba
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numba"]); import numba
from numba import njit
from scipy.stats import wilcoxon, spearmanr
warnings.filterwarnings("ignore")

# =============================== RUN CONTROL =================================
PROFILE = "paper"            # "paper" = every constant of the paper; "smoke" = tiny pipeline test (NOT evidence)
RUN_DATASETS = ["IoT-23"]    # add "MQTT-IoT-IDS2020", "TON_IoT" after providing their canonical CSVs below
DATASET_SPECS = {
    "IoT-23":           dict(kind="iot23", role="confirmatory", level=0.05),
    "MQTT-IoT-IDS2020": dict(kind="csv", path="/content/mqtt_canonical.csv", role="development", level=0.025),
    "TON_IoT":          dict(kind="csv", path="/content/ton_canonical.csv",  role="development", level=0.025),
}
MAX_HOURS = 11.0             # stop gracefully before Colab's session limit; re-run to resume
ALLOW_PC1_FAIL = False       # paper: a PC1 failure => no confirmatory claim. True = continue, labelled EXPLORATORY
DRIVE_DIR = "/content/drive/MyDrive/mand_repro" if os.path.isdir("/content") else "./mand_repro"
# IoT-23 scenarios: NOT specified in the paper -> must be filled by the authors (see ledger). Placeholder below.
CAPTURES = ["CTU-IoT-Malware-Capture-34-1", "CTU-IoT-Malware-Capture-3-1", "CTU-IoT-Malware-Capture-1-1",
            "CTU-IoT-Malware-Capture-8-1", "CTU-IoT-Malware-Capture-42-1", "CTU-IoT-Malware-Capture-20-1",
            "CTU-IoT-Malware-Capture-21-1", "CTU-IoT-Malware-Capture-33-1",
            "CTU-Honeypot-Capture-4-1", "CTU-Honeypot-Capture-5-1"]
MAX_LINES_PER_CAPTURE = 300_000        # ASSUMED (paper gives no cap); None = whole file
BASE_URL = ("https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/"
            "IndividualScenarios/{}/bro/conn.log.labeled")
MAX_ATTACK_SHARE = None                # paper has no such rule (my earlier script had one; not used here)
TREAT_MISSING_AS_ZERO = False          # paper rule: '-' duration => excluded from windows

# =============================== PAPER CONSTANTS =============================
PAPER = dict(
    m=4, c=8, batch_chunks=32, E=60, ramp_end=30, tau0=1.0, alpha=1.0,
    delta_grid=[30, 60, 120, 300], Q_default=5, Q_alt=3, trunc_thresh=0.05, min_mixed_for_Q=20,
    sesoi_f1=0.005, sesoi_mj=0.01, min_support=50,
    teacher_seeds=list(range(11, 21)), student_seeds=list(range(301, 321)),   # teacher 11+i <-> students 301+2i,302+2i
    tuning_seeds=[401, 402, 403, 404, 405], reserved_teachers=[1, 2],          # 3 seeds with teacher 1, 2 with teacher 2
    heldout_teacher_seeds=list(range(101, 111)), heldout_student_seeds=list(range(201, 221)),
    p_levels=dict(confirmatory=0.05, development=0.025),
    grids=dict(lr=[1e-3, 3e-3], beta=[0.5, 1.0], theta=[2.0, 4.0], tau1=[0.02, 0.05, 0.1], kappa=[2.0, 4.0],
               mult=[0.1, 0.3, 1.0, 3.0, 10.0], lam=[1.0, 3.0, 10.0]),
    tune_base=30, tune_per_h=10,
    n_h={"B1": 0, "B2": 0, "B2i": 0, "B8": 0, "B3": 1, "B4": 1, "B9": 1, "X0": 2, "X1": 2, "B7": 2,
         "MAND": 3, "B6": 3, "X2": 3},
    calib=dict(lr=1e-3, beta=1.0, theta=2.0, kappa=2.0, tau1=0.05),
    grid_shift_lo=0.05, grid_shift_hi=20.0,
    pc1=dict(frac_ge3=0.5, mixed_min=0.02, missing_max=0.05, tie_max=0.20, min_attack=50),
    pc2_spread=0.05, pc3_retain=0.20, mj_prev_min=0.005, label_purity_trigger=0.95,
    block_lens=[300.0, 60.0, 30.0], blocks_min=200, blocks_low=100, ci_level=0.95,
    hc=dict(eta_min=0.005, mde_max=0.005, sd_inflate=1.5, power=0.8),
    tost_margin=0.005, student_width=64, teacher_width=512, student_hidden_layers=2,
    rkd_triplets=256, rkd_weights=(1.0, 2.0),
    b10_rf=dict(trees=[100, 300], depth=[20, None]), b10_xgb=dict(n=[100, 300], depth=[4, 8], lr=[0.1, 0.3]),
    robust=dict(noise=[0.05, 0.1, 0.2], f1_margin=0.005, mj_margin=0.02, p_event=0.10, p_type=1 / 3, benign_insert=0.10),
    split=(0.6, 0.2, 0.2),
)
# ---- ASSUMED: NOT specified in the paper. Every item is a free parameter that must be disclosed. -------------
ASSUMED = dict(
    teacher_optimizer="Adam", teacher_lr=1e-3, teacher_epochs=8, teacher_batch=1024, teacher_loss="cross-entropy, unweighted",
    teacher_init="PyTorch default (seeded by torch.manual_seed(seed))", teacher_seed_rule="deployed=100*s+99; fold q=100*s+q",
    student_optimizer="Adam (default betas/eps, no weight decay)", student_init="PyTorch default",
    weighted_loss_normalisation="sum_f w_f*l_f / sum_f w_f over closure (w_f=1/n_f)",
    b2i_batch="256 flows (=c*32), one pass per epoch, no closures",
    feature_encoding="log1p numerics + one-hot proto/service/conn_state + log1p history-letter counts; train mean/std, clip +-10 (paper d=132; this d differs)",
    iot23_scenarios="NOT SPECIFIED IN PAPER (CAPTURES list above is a placeholder)", iot23_line_cap=MAX_LINES_PER_CAPTURE,
    missing_duration_rule="paper rule (excluded from windows) unless TREAT_MISSING_AS_ZERO",
    tuning_search_rng="numpy default_rng(1000+index of method); sampling without replacement",
    tuning_tie_break="higher mean val macro-F1; ties -> earlier candidate",
    grid_shift_rule="mult grid shifted by one position (x sqrt(10)-ish list step) once; best of union kept",
    calibration_seed=7, calibration_teacher="reserved teacher 1", calibration_batches=8,
    b4_temperature="softmax at T=1", b7_entropy_temperature="teacher softmax at T=1, normalised by log K",
    b7_rescale="multiplier (1+lam_e*h_f) divided by its closure mean", b9_q_temperature="teacher tempered at theta before reweighting",
    b6_eps_rule="eps=(1-mean max prob of kappa-tempered OOF teacher on train)/(1-1/K)",
    b3_huber="smooth_l1 (delta=1) over all pairs incl. diagonal; angle on random triplets",
    added_term_ramp="gamma(e)=gamma_max*rho(e) used for every added-term method (MOD, B3, B4, B7)",
    bootstrap_replicates=1000, bootstrap_seed=2024, mde_mc_reps=2000, mde_seed=7,
    heldout_split="20% of host keys (seeded hash, seed 12345) held out of train/val; test = their test-period flows",
    r2_r4_perturbation="R2: members dropped / delayed beyond anchor end time removed / duplicated(no effect on min/max), no refill; "
                       "R3: one attack flow from another test host added to benign windows; R4: benign flow inserted w.p. 0.1 per window slot in attack windows; "
                       "perturbation applied to training-frame windows of every flow, MJ over F u F^ev formulas",
    mj_formulas="Box,Dia,Box2,Dia2,DiaBox,BoxDia,Box3,Dia3,Dia&~Box (atk atom), hard ops, threshold 0.5",
    b10_inputs="per-flow standardized features, label only (no teacher)", b8_aggregation="[x_w, mean_R x, max_R x]",
    holdout_hc="same HC procedure on held-out validation blocks", cpu_latency="Colab CPU 1 thread, NOT Raspberry Pi 5",
    cross_fit_excluded_flows="flows of other folds within Delta of fold-q time interval (per capture) excluded from teacher q training",
)
if PROFILE == "smoke":      # tiny pipeline test; deliberately NOT the paper's numbers
    PAPER.update(E=2, ramp_end=1, teacher_seeds=[11, 12], student_seeds=[301, 302, 303, 304],
                 tuning_seeds=[401, 402], reserved_teachers=[1, 2], tune_base=1, tune_per_h=0, teacher_width=32,
                 heldout_teacher_seeds=[101], heldout_student_seeds=[201, 202])
    ASSUMED.update(teacher_epochs=1, bootstrap_replicates=20, mde_mc_reps=100)
    MAX_LINES_PER_CAPTURE = 40_000
PROF = {"MAX_LINES": MAX_LINES_PER_CAPTURE or 10**9}
M, CLEN, NCHUNK = PAPER["m"], PAPER["c"], PAPER["batch_chunks"]
DELTA_GRID = PAPER["delta_grid"]
WORKDIR = os.path.join(DRIVE_DIR, "data"); os.makedirs(WORKDIR, exist_ok=True)
T_START = time.time()

def log(*a): print(*a, flush=True)
def hours(): return (time.time() - T_START) / 3600
class OutOfTime(Exception): pass
def check_time():
    if hours() > MAX_HOURS: raise OutOfTime()

def b64(a):  return base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
def unb64(s, dt): return np.frombuffer(base64.b64decode(s), dtype=dt)
def sha(path, n=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(n)
            if not b: break
            h.update(b)
    return h.hexdigest()

# =============================== RESUMABLE STORE =============================
def config_fingerprint():
    blob = json.dumps(dict(PAPER=PAPER, ASSUMED={k: str(v) for k, v in ASSUMED.items()}, PROFILE=PROFILE,
                           captures=CAPTURES, ds=RUN_DATASETS, miss0=TREAT_MISSING_AS_ZERO), sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:10]
FPRINT = config_fingerprint()
RUN_DIR = os.path.join(DRIVE_DIR, f"run_{PROFILE}_{FPRINT}"); os.makedirs(RUN_DIR, exist_ok=True)

class Store:
    """append-only jsonl; key -> value; resume = skip keys already present."""
    def __init__(self, path):
        self.path, self.d = path, {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f:
                    try: r = json.loads(line); self.d[r["k"]] = r["v"]
                    except Exception: pass
    def has(self, k): return k in self.d
    def get(self, k): return self.d[k]
    def put(self, k, v):
        self.d[k] = v
        with open(self.path, "a") as f: f.write(json.dumps({"k": k, "v": v}) + "\n")
STORE = Store(os.path.join(RUN_DIR, "results.jsonl"))

def write_manifest(extra=None):
    try: import torch; tv, gpu = torch.__version__, (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    except Exception: tv, gpu = "n/a", "n/a"
    try: script_hash = sha(os.path.abspath(sys.argv[0])) if os.path.exists(sys.argv[0]) else "interactive"
    except Exception: script_hash = "interactive"
    man = dict(profile=PROFILE, fingerprint=FPRINT, python=sys.version, platform=platform.platform(), torch=tv, gpu=gpu,
               numpy=np.__version__, pandas=pd.__version__, script_sha256=script_hash, PAPER=PAPER,
               ASSUMED={k: str(v) for k, v in ASSUMED.items()}, run_datasets=RUN_DATASETS, captures=CAPTURES,
               data_files={}, extra=extra or {})
    for fn in sorted(os.listdir(WORKDIR)):
        if fn.endswith(".log") or fn.endswith(".csv"): man["data_files"][fn] = sha(os.path.join(WORKDIR, fn))
    with open(os.path.join(RUN_DIR, "manifest.json"), "w") as f: json.dump(man, f, indent=1, default=str)
    return man

COVERAGE = """COVERAGE
  implemented : PC1-PC4, cross-fitted CXT/PFT teachers, B1-B4, B6-B10, X0-X2, MAND, tuning protocol (30+10*n_h random search,
                5 tuning seeds over 2 reserved teachers, grid-shift repeat), HC/MDE, C1-C4, C2 (Holm), AF (Holm), TOST,
                host-held-out conjunct, ablations 1,2,3,4,6,8, robustness R1-R4 (Holm), cost (params/FLOPs/CPU latency),
                comparison against the paper's reported numbers, resumable cache + manifest.
  NOT implemented: ablation 7 (kNN frame), Raspberry-Pi latency, calibration/KL/FPR/ECE extras, dense/sparse recall,
                campaign-held-out and temporal-shift splits, Edge-IIoTset backup PC1 (needs its raw CSVs + exact mapping).
"""
# =============================== DATA ========================================
ZCOLS = ["ts", "uid", "orig_h", "orig_p", "resp_h", "resp_p", "proto", "service", "duration",
         "orig_bytes", "resp_bytes", "conn_state", "local_orig", "local_resp", "missed", "history",
         "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "tail"]

def fetch_capture(cap, max_lines):
    """Download (at most max_lines lines of) conn.log.labeled; cache in WORKDIR."""
    path = f"{WORKDIR}/{cap}.{max_lines}.log"
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "r", errors="ignore") as f:
            head = f.read(2000).lower()
        if "<html" in head or "<!doctype" in head:
            os.remove(path)                      # corrupt cache from a wrong URL
        else:
            return path
    url = BASE_URL.format(cap)
    last = None
    for verify in (True, False):                 # fall back to insecure if cert fails
        try:
            r = requests.get(url, stream=True, timeout=90, verify=verify)
            r.raise_for_status()
            n = 0
            with open(path, "wb") as f:
                for line in r.iter_lines(chunk_size=1 << 20):
                    f.write(line + b"\n"); n += 1
                    if n >= max_lines:
                        break
            r.close()
            with open(path, "r", errors="ignore") as f:
                if "<html" in f.read(2000).lower():
                    raise RuntimeError("got an HTML page instead of a Zeek log")
            return path
        except Exception as e:
            last = e
            if os.path.exists(path): os.remove(path)
    log(f"  !! skipping {cap}: {last}")
    return None

def _read_tab(path):
    with open(path, "r", errors="ignore") as f:
        first = next((l for l in f if not l.startswith("#") and l.strip()), "")
    ncol = len(first.rstrip("\n").split("\t"))
    extra = [f"x{i}" for i in range(max(ncol - 20, 1))]          # tail may be 1 col (space-joined) or several
    names = ZCOLS[:20] + extra
    df = pd.read_csv(path, sep="\t", comment="#", header=None, names=names, dtype=str, index_col=False,
                     keep_default_na=False, na_filter=False, on_bad_lines="skip")
    if len(extra) > 1:
        log(f"  {os.path.basename(path)}: {ncol} tab columns (label/detailed-label are separate columns)")
    df["tail"] = df[extra].astype(str).agg(" ".join, axis=1) if len(extra) > 1 else df[extra[0]].astype(str)
    return df[df["tail"].str.strip().str.len() > 0]

def _read_ws(path):
    """fallback: split every data line on any whitespace (first 20 fields + joined tail)."""
    rows = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("#") or not line.strip(): continue
            t = line.split()
            if len(t) >= 21: rows.append(t[:20] + [" ".join(t[20:])])
    return pd.DataFrame(rows, columns=ZCOLS)

def parse_conn(path):
    df = _read_tab(path)
    if len(df) == 0:
        with open(path, "r", errors="ignore") as f:
            ex = [l.rstrip("\n")[:200] for l in f if not l.startswith("#")][:2]
        log(f"  tab parse gave 0 rows for {os.path.basename(path)}; sample data lines: {ex} -> whitespace fallback")
        df = _read_ws(path)
        if len(df) == 0:
            with open(path, "r", errors="ignore") as f:
                log("  first raw lines:", [l[:160] for l in f.readlines()[:6]])
    tail = df["tail"]
    lab = np.where(tail.str.contains("malicious", case=False), 1, np.where(tail.str.contains("benign", case=False), 0, -1))
    if (lab >= 0).sum() == 0:
        log(f"  !! no Malicious/Benign label found in {os.path.basename(path)}; most common tail values: "
            f"{tail.value_counts().head(5).to_dict()}")
    df = df[lab >= 0].copy(); df["y"] = lab[lab >= 0]
    df["ts"] = pd.to_numeric(df["ts"], errors="coerce")
    dur = pd.to_numeric(df["duration"].replace("-", np.nan), errors="coerce")
    df["missing"] = dur.isna().values
    df["valid"] = (~df["missing"]) | TREAT_MISSING_AS_ZERO   # False -> singleton window (paper rule)
    df["sigma"] = df["ts"] + dur.fillna(0.0)             # flow END time
    for c in ["orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes"]:
        df[c] = pd.to_numeric(df[c].replace("-", np.nan), errors="coerce").fillna(0.0)
    df["duration"] = dur.fillna(0.0)
    df["host"] = df["orig_h"]
    df = df.dropna(subset=["ts"])
    return df[["ts", "sigma", "valid", "missing", "y", "host", "proto", "service", "conn_state", "history",
               "duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
               "orig_ip_bytes", "resp_ip_bytes"]].reset_index(drop=True)

def load_all():
    frames = []
    for ci, cap in enumerate(CAPTURES):
        p = fetch_capture(cap, PROF["MAX_LINES"])
        if p is None: continue
        d = parse_conn(p); d["cap"] = ci
        if len(d) == 0:
            log(f"  {cap}: 0 usable flows (see message above)"); continue
        if MAX_ATTACK_SHARE is not None and d.y.mean() > MAX_ATTACK_SHARE:
            log(f"  {cap}: {len(d):,} flows, attack share {d.y.mean():.1%} > {MAX_ATTACK_SHARE:.0%} -> DROPPED (near-pure attack)")
            continue
        log(f"  {cap}: {len(d):,} flows, attack={int(d.y.sum()):,}, missing-duration={d.missing.mean():.3%}")
        frames.append(d)
    if not frames:
        raise RuntimeError("No capture could be downloaded. Check the network / URLs.")
    D = pd.concat(frames, ignore_index=True)
    D["host_id"] = pd.factorize(D["cap"].astype(str) + "|" + D["host"])[0]
    return D

def build_features(D):
    cols, names = [], []
    for c in ["duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts", "orig_ip_bytes", "resp_ip_bytes"]:
        cols.append(np.log1p(D[c].values.astype(np.float64).clip(min=0))); names.append(c)
    for c in ["proto", "service", "conn_state"]:
        v = D[c].values
        for k in sorted(set(v)):
            cols.append((v == k).astype(np.float64)); names.append(f"{c}={k}")
    codes, uniq = pd.factorize(D["history"])
    letters = sorted(set("".join(uniq)) - {"-"})
    cnt = np.array([[u.count(l) for l in letters] for u in uniq], dtype=np.float64)
    hist = np.log1p(cnt[codes])
    for j, l in enumerate(letters):
        cols.append(hist[:, j]); names.append(f"hist={l}")
    return np.stack(cols, 1).astype(np.float32), names

# ============================ WINDOWS (numba) ================================
@njit(cache=False)
def _windows(sig, grp, valid, delta, m, K):
    """R(w) history per paper Eq.(1): same group, strictly earlier end time, within delta,
    whole tie-groups, add while total<=m; an overflowing group is added only if total<=2m.
    Arrays must be sorted by (grp, sig). Returns hist (N,K) of row indices (-1 padded)
    and collapse flag (first tie-group already overflows 2m)."""
    N = sig.shape[0]
    hist = np.full((N, K), -1, np.int32)
    collapse = np.zeros(N, np.bool_)
    vp = np.empty(N, np.int64)
    gs = np.empty(N, np.int64)
    i0 = 0
    while i0 < N:
        i1 = i0
        while i1 < N and grp[i1] == grp[i0]:
            i1 += 1
        nv = 0
        for p in range(i0, i1):
            if valid[p]:
                vp[nv] = p; nv += 1
        for t in range(nv):
            if t > 0 and sig[vp[t]] == sig[vp[t - 1]]:
                gs[t] = gs[t - 1]
            else:
                gs[t] = t
        for t in range(nv):
            s_t = sig[vp[t]]
            k = gs[t]
            total = 0
            cnt = 0
            while k > 0:
                if s_t - sig[vp[k - 1]] > delta:
                    break
                g = gs[k - 1]
                size = k - g
                if total + size <= m:
                    for q in range(k - 1, g - 1, -1):
                        hist[vp[t], cnt] = vp[q]; cnt += 1
                    total += size
                    k = g
                elif total + size <= 2 * m:
                    for q in range(k - 1, g - 1, -1):
                        hist[vp[t], cnt] = vp[q]; cnt += 1
                    total += size
                    break
                else:
                    if total == 0:
                        collapse[vp[t]] = True
                    break
        i0 = i1
    return hist, collapse

def make_NB(sig, grp, valid, delta, m):
    hist, coll = _windows(sig, grp.astype(np.int64), valid, float(delta), int(m), int(2 * m))
    NB = np.concatenate([np.arange(len(sig), dtype=np.int64)[:, None], hist.astype(np.int64)], 1)
    return NB, coll

def make_split(sig, cap, delta):
    """60/20/20 per capture by end time, guard gap `delta` before val and before test.
    0 train, 1 val, 2 test, -1 dropped (guard)."""
    split = np.full(len(sig), -1, np.int8)
    for c in np.unique(cap):
        idx = np.where(cap == c)[0]
        s = np.sort(sig[idx]); n = len(s)
        tv, tt = s[int(n * 0.6)], s[min(int(n * 0.8), n - 1)]
        sc = sig[idx]; sp = np.full(len(idx), -1, np.int8)
        sp[sc < tv - delta] = 0
        sp[(sc >= tv) & (sc < tt - delta)] = 1
        sp[sc >= tt] = 2
        split[idx] = sp
    return split

def assign_folds(sig, cap, split, Q):
    fold = np.full(len(sig), -1, np.int64)
    for c in np.unique(cap):
        idx = np.where((cap == c) & (split == 0))[0]
        if len(idx) == 0: continue
        r = np.argsort(np.argsort(sig[idx], kind="stable"), kind="stable")
        fold[idx] = (r * Q) // len(idx)
    return fold

# ================================= PC1 =======================================
def pc1(D):
    sig, cap, host, valid, y = (D[k].values for k in ["sigma", "cap", "host_id", "valid", "y"])
    log("\n[PC1] dataset audit / applicability gates")
    chosen = None
    for delta in DELTA_GRID:
        split = make_split(sig, cap, delta)
        idx = np.where(split == 0)[0]
        o = np.lexsort((sig[idx], host[idx])); idx = idx[o]
        NB, coll = make_NB(sig[idx], host[idx], valid[idx], delta, M)
        size = (NB >= 0).sum(1)
        yl = y[idx]
        f_all, f_atk = (size >= 3).mean(), (size[yl == 1] >= 3).mean() if (yl == 1).any() else 0.0
        log(f"  Delta={delta:>3}s: frac(|R|>=3) all={f_all:.3f} attack={f_atk:.3f}")
        if f_all >= 0.5 and f_atk >= 0.5:
            chosen = delta; break
    out = dict(delta=chosen, passed=False, gates={})
    if chosen is None:
        log("  (i) no qualifying Delta -> PC1 FAIL"); out["gates"]["(i) qualifying delta"] = False
        return out
    split = make_split(sig, cap, chosen)
    idx = np.where(split == 0)[0]; o = np.lexsort((sig[idx], host[idx])); idx = idx[o]
    NB, coll = make_NB(sig[idx], host[idx], valid[idx], chosen, M)
    yl = y[idx]
    ymin, ymax = yl.copy(), yl.copy()
    for k in range(1, NB.shape[1]):
        col = NB[:, k]; msk = col >= 0; yv = yl[np.clip(col, 0, None)]
        ymin = np.where(msk, np.minimum(ymin, yv), ymin); ymax = np.where(msk, np.maximum(ymax, yv), ymax)
    mixed = ymin != ymax
    mixed_frac = mixed.mean()
    miss_atk = D["missing"].values[idx][yl == 1].mean() if (yl == 1).any() else 1.0
    tie_atk = coll[yl == 1].mean() if (yl == 1).any() else 1.0
    n_atk = {s: int(((split == s) & (y == 1)).sum()) for s in (0, 1, 2)}
    g = {"(i) qualifying Delta": True,
         "(ii) mixed-window fraction >= 2%": mixed_frac >= 0.02,
         "(iii) missing-duration attack <= 5%": miss_atk <= 0.05,
         "(iv) tie-collapse (attack) <= 20%": tie_atk <= 0.20,
         "(v) >=50 attack flows in train/val/test": all(v >= 50 for v in n_atk.values())}
    log(f"  selected Delta = {chosen}s | mixed-window frac={mixed_frac:.3%} | missing-dur(attack)={miss_atk:.2%} | "
        f"tie-collapse(attack)={tie_atk:.2%} | attack flows tr/va/te={n_atk}")
    # Q rule
    Q = 5
    fold5 = assign_folds(sig, cap, split, 5)[idx]
    n_mixed = int(mixed.sum())
    if n_mixed >= 20:
        trunc = np.zeros(len(idx), bool)
        for k in range(1, NB.shape[1]):
            col = NB[:, k]; msk = col >= 0
            trunc |= msk & (fold5[np.clip(col, 0, None)] != fold5)
        tf = trunc[mixed].mean()
        if tf > 0.05: Q = 3
        log(f"  fold-boundary truncation among mixed anchors (Q=5): {tf:.3%} -> Q={Q}")
    else:
        log(f"  only {n_mixed} mixed anchors (<20): truncation fraction unstable, Q=5 kept")
    # label purity
    h = host[idx]; uh, inv = np.unique(h, return_inverse=True)
    mean_lab = np.bincount(inv, weights=yl) / np.bincount(inv)
    purity = (np.maximum(mean_lab, 1 - mean_lab)[inv] >= 0.95).mean()
    log(f"  label purity (train records in >=95%-pure hosts) = {purity:.1%}"
        + ("  -> host-held-out conjunct would be required (NOT run here)" if purity > 0.95 else ""))
    for k, v in g.items(): log(f"   {'PASS' if v else 'FAIL'}  {k}")
    out.update(passed=all(g.values()), gates=g, Q=Q, mixed_frac=mixed_frac, purity=purity)
    return out

# ============================= FINAL DATASET =================================
def build_dataset(D, Xraw, delta, Q, heldout=False):
    sig, cap, host, valid, y = (D[k].values for k in ["sigma", "cap", "host_id", "valid", "y"])
    split = make_split(sig, cap, delta)
    if heldout:        # ASSUMED construction: 20% of host keys never seen in train/val; test = their test-period flows
        hosts = np.unique(host); rg = np.random.default_rng(12345)
        held = np.isin(host, rg.choice(hosts, size=max(1, int(0.2 * len(hosts))), replace=False))
        split[held & (split < 2)] = -1; split[~held & (split == 2)] = -1
    fold = assign_folds(sig, cap, split, Q)
    seg = np.where(split == 0, fold, np.where(split == 1, Q, Q + 1))
    grp = host.astype(np.int64) * (Q + 2) + seg
    idx = np.where(split >= 0)[0]
    idx = idx[np.lexsort((sig[idx], grp[idx]))]
    S = SimpleNamespace()
    S.delta, S.Q = float(delta), Q
    S.sig, S.cap, S.host, S.valid, S.y = sig[idx], cap[idx], host[idx], valid[idx], y[idx].astype(np.int64)
    S.split, S.fold, S.grp = split[idx], fold[idx], grp[idx]
    Xs = Xraw[idx].astype(np.float64)
    tr = S.split == 0
    mu, sd = Xs[tr].mean(0), Xs[tr].std(0) + 1e-6
    S.X = np.clip((Xs - mu) / sd, -10, 10).astype(np.float32)
    S.N, S.d = len(idx), S.X.shape[1]
    S.NB, S.coll = make_NB(S.sig, S.grp, S.valid, delta, M)
    # group bounds (inclusive) for the X2 shuffle control
    b = np.r_[0, np.flatnonzero(np.diff(S.grp)) + 1, S.N]
    S.gstart = np.repeat(b[:-1], np.diff(b)); S.gend = np.repeat(b[1:] - 1, np.diff(b))
    S.tr_idx, S.va_idx, S.te_idx = (np.where(S.split == k)[0] for k in (0, 1, 2))
    # fold time intervals per capture (for teacher guard exclusion)
    S.fold_lo = {}; S.fold_hi = {}
    for c in np.unique(S.cap):
        for q in range(Q):
            m_ = (S.cap == c) & (S.split == 0) & (S.fold == q)
            if m_.any(): S.fold_lo[(c, q)] = S.sig[m_].min(); S.fold_hi[(c, q)] = S.sig[m_].max()
    return S

def frames_of(S):
    d, m = S.delta, M
    return {"(D/2,m)": (d / 2, m), "(2D,m)": (2 * d, m), "(D,2m)": (d, 2 * m), "(D,m/2)": (d, max(m // 2, 1))}

def changed_rows(nb_base, nb_new):
    K = max(nb_base.shape[1], nb_new.shape[1])
    pad = lambda a: np.concatenate([a, np.full((len(a), K - a.shape[1]), -1, a.dtype)], 1) if a.shape[1] < K else a
    a, b = np.sort(pad(nb_base), 1), np.sort(pad(nb_new), 1)
    return (a != b).any(1)

# ---- hard modal operators for MJ (numpy) ----
def _hbox(a, nb): return np.where(nb >= 0, a[np.clip(nb, 0, None)], np.inf).min(1)
def _hdia(a, nb): return np.where(nb >= 0, a[np.clip(nb, 0, None)], -np.inf).max(1)
def hard_formulas(a, nb):
    b, d = _hbox(a, nb), _hdia(a, nb)
    b2, d2 = _hbox(b, nb), _hdia(d, nb)
    return {"Box": b, "Dia": d, "Box2": b2, "Dia2": d2, "DiaBox": _hdia(b, nb), "BoxDia": _hbox(d, nb),
            "Box3": _hbox(b2, nb), "Dia3": _hdia(d2, nb), "Dia&~Box": np.minimum(d, 1 - b)}

def compact(S, idx, nb):
    """restrict a global window matrix to a split (windows never leave the split)."""
    loc = np.full(S.N, -1, np.int64); loc[idx] = np.arange(len(idx))
    sub = nb[idx]
    return np.where(sub >= 0, loc[np.clip(sub, 0, None)], -1)


def load_generic(path):
    """Canonical CSV for MQTT-IoT-IDS2020 / TON_IoT / Edge-IIoTset: columns cap,host,ts,duration,label + numeric feature columns.
    duration blank/NaN => missing (excluded from windows, kept for scoring). label: 0 benign / 1 attack."""
    df = pd.read_csv(path)
    for c in ["cap", "host", "ts", "duration", "label"]:
        assert c in df.columns, f"canonical CSV needs column '{c}'"
    dur = pd.to_numeric(df["duration"], errors="coerce"); ts = pd.to_numeric(df["ts"], errors="coerce")
    D = pd.DataFrame(dict(ts=ts, sigma=ts + dur.fillna(0.0), missing=dur.isna().values, y=df["label"].astype(int).values,
                          cap=pd.factorize(df["cap"])[0], host=df["host"].astype(str)))
    D["valid"] = (~D["missing"]) | TREAT_MISSING_AS_ZERO
    D["host_id"] = pd.factorize(D["cap"].astype(str) + "|" + D["host"])[0]
    feats = df.drop(columns=["cap", "host", "ts", "duration", "label"]).select_dtypes("number")
    D = D.dropna(subset=["ts"]).reset_index(drop=True)
    return D, feats.fillna(0.0).values[D.index].astype(np.float32), list(feats.columns)

def load_dataset(name):
    spec = DATASET_SPECS[name]
    if spec["kind"] == "iot23":
        D = load_all(); X, names = build_features(D); return D, X, names
    return load_generic(spec["path"])

# ============================================================================
# ============================ TORCH CORE =====================================
# ============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
TW = PAPER["teacher_width"]

class Student(nn.Module):               # d_in-64-64-2
    def __init__(s, d):
        super().__init__(); w = PAPER["student_width"]
        s.net = nn.Sequential(nn.Linear(d, w), nn.ReLU(), nn.Linear(w, w), nn.ReLU(), nn.Linear(w, 2))
    def forward(s, x): return s.net(x)

class PFT(nn.Module):                   # 4-layer residual MLP, width 512 (per-flow teacher)
    def __init__(s, d, w=None):
        super().__init__(); w = w or TW
        s.enc = nn.Linear(d, w)
        s.blocks = nn.ModuleList([nn.Sequential(nn.Linear(w, w), nn.ReLU()) for _ in range(4)])
        s.out = nn.Linear(w, 2)
    def forward(s, x):
        h = F.relu(s.enc(x))
        for b in s.blocks: h = h + b(h)
        return s.out(h)

class CXT(nn.Module):                   # 2 message-passing layers over R, concat(mean,max)
    def __init__(s, d, w=None):
        super().__init__(); w = w or TW
        s.enc = nn.Linear(d, w); s.l1 = nn.Linear(3 * w, w); s.l2 = nn.Linear(3 * w, w); s.out = nn.Linear(w, 2)
    @staticmethod
    def agg(h_self, h_all, nbl):
        mask = nbl >= 0
        g = h_all[nbl.clamp_min(0)]
        mean = (g * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        mx = g.masked_fill(~mask.unsqueeze(-1), -1e9).max(1).values
        return torch.cat([h_self, mean, mx], 1)

def teacher_logits(model, X, NB, idx):
    if isinstance(model, PFT): return model(X[idx])
    nb1 = NB[idx]; S1 = torch.unique(nb1[nb1 >= 0])
    nb2 = NB[S1]; S2 = torch.unique(nb2[nb2 >= 0])
    loc = torch.full((X.shape[0],), -1, dtype=torch.long, device=X.device); loc[S2] = torch.arange(len(S2), device=X.device)
    h0 = F.relu(model.enc(X[S2]))
    nbl2 = torch.where(nb2 >= 0, loc[nb2.clamp_min(0)], torch.full_like(nb2, -1))
    h1 = F.relu(model.l1(CXT.agg(h0[loc[S1]], h0, nbl2)))
    loc2 = torch.full((X.shape[0],), -1, dtype=torch.long, device=X.device); loc2[S1] = torch.arange(len(S1), device=X.device)
    nbl1 = torch.where(nb1 >= 0, loc2[nb1.clamp_min(0)], torch.full_like(nb1, -1))
    h2 = F.relu(model.l2(CXT.agg(h1[loc2[idx]], h1, nbl1)))
    return model.out(h2)

def to_t(a, dt=None):
    t = torch.as_tensor(a, device=dev); return t if dt is None else t.to(dt)

# ------------------------------ metrics / stats helpers ----------------------
def macro_f1(y, p, min_support=None):
    min_support = PAPER["min_support"] if min_support is None else min_support
    f = []
    for k in (0, 1):
        if (y == k).sum() < min_support: continue
        tp = ((y == k) & (p == k)).sum(); fp = ((y != k) & (p == k)).sum(); fn = ((y == k) & (p != k)).sum()
        f.append(2 * tp / max(2 * tp + fp + fn, 1))
    return float(np.mean(f)) if f else float("nan")

def conf_to_f1(conf):
    tp, fp, fn, tn = (conf[..., i] for i in range(4))
    return (2 * tp / np.maximum(2 * tp + fp + fn, 1) + 2 * tn / np.maximum(2 * tn + fn + fp, 1)) / 2

def block_conf(y, p, blk, nb):
    c = np.zeros((nb, 4))
    c[:, 0] = np.bincount(blk, weights=((y == 1) & (p == 1)), minlength=nb); c[:, 1] = np.bincount(blk, weights=((y == 0) & (p == 1)), minlength=nb)
    c[:, 2] = np.bincount(blk, weights=((y == 1) & (p == 0)), minlength=nb); c[:, 3] = np.bincount(blk, weights=((y == 0) & (p == 0)), minlength=nb)
    return c

def make_blocks(S):
    te = S.te_idx
    for L in PAPER["block_lens"]:
        key = S.host[te].astype(np.int64) * 10_000_000 + np.floor(S.sig[te] / L).astype(np.int64) % 10_000_000
        u, blk = np.unique(key, return_inverse=True)
        if len(u) >= PAPER["blocks_min"]: return blk, len(u), L, ""
    flag = "low block count" if len(u) >= PAPER["blocks_low"] else "seed-level only"
    return blk, len(u), L, flag

def holm(pvals, alpha=0.05):
    pv = np.asarray(pvals, float); order = np.argsort(pv); rej = np.zeros(len(pv), bool)
    for r, i in enumerate(order):
        if pv[i] <= alpha / (len(pv) - r): rej[i] = True
        else: break
    return rej

def paired_p(d, alternative="greater"):
    d = np.asarray(d, float)
    if np.all(d == 0): return 1.0
    try: return float(wilcoxon(d, alternative=alternative, method="exact", zero_method="wilcox").pvalue)
    except Exception: return float(wilcoxon(d, alternative=alternative, zero_method="wilcox").pvalue)

def signed_rank_cdf(n):
    """exact null distribution of W+ for n (no ties): counts[k]=#subsets of {1..n} with sum k."""
    mx = n * (n + 1) // 2; c = np.zeros(mx + 1); c[0] = 1
    for r in range(1, n + 1): c[r:] = c[r:] + c[:mx + 1 - r].copy()
    return np.cumsum(c) / 2.0 ** n

def mde_power(sd, n, level, power, inflate, reps, rng):
    """smallest true mean difference with MC power>=power (normal diffs, sd*inflate, exact one-sided signed-rank)."""
    cdf = signed_rank_cdf(n); crit = max([k for k in range(len(cdf)) if cdf[k] <= level] or [-1])
    def pw(delta):
        d = rng.normal(delta, inflate * sd, size=(reps, n)); rk = np.argsort(np.argsort(np.abs(d), 1), 1) + 1
        wneg = (rk * (d < 0)).sum(1); return (wneg <= crit).mean()
    lo, hi = 0.0, max(10 * sd, 0.05)
    if crit < 0: return float("inf")
    for _ in range(25):
        mid = (lo + hi) / 2
        if pw(mid) >= power: hi = mid
        else: lo = mid
    return hi

# ----------------------------- dataset preparation ---------------------------
def prep_frames(S):
    frames = frames_of(S)
    te_loc = compact(S, S.te_idx, S.NB); va_loc = compact(S, S.va_idx, S.NB)
    info = {}
    for name, (dl, mm) in frames.items():
        NBf, _ = make_NB(S.sig, S.grp, S.valid, dl, mm)
        te_nb, va_nb = compact(S, S.te_idx, NBf), compact(S, S.va_idx, NBf)
        ch_te, ch_va = changed_rows(te_loc, te_nb), changed_rows(va_loc, va_nb)
        yv, yt = S.y[S.va_idx] == 1, S.y[S.te_idx] == 1
        a_va = float(ch_va[yv].mean()) if yv.any() else 0.0
        info[name] = dict(te_nb=te_nb, va_nb=va_nb, ch_te=ch_te, ch_va=ch_va, keep=a_va >= PAPER["pc3_retain"],
                          frac_te=float(ch_te.mean()), frac_te_atk=float(ch_te[yt].mean()) if yt.any() else 0.0, frac_va_atk=a_va)
    return info

def prepare(name, heldout=False, force=False):
    """Load -> PC1 -> dataset tensors. Returns None when PC1 fails and ALLOW_PC1_FAIL is False."""
    cache_key = f"pc1|{name}"
    D, Xraw, fnames = load_dataset(name)
    P1 = pc1(D)
    if not P1["passed"]:
        if not ALLOW_PC1_FAIL and not force:
            log(f"!! {name}: PC1 FAILED -> no claim for this dataset (paper rule). Set ALLOW_PC1_FAIL=True for an EXPLORATORY run.")
            STORE.put(cache_key, dict(passed=False, gates={k: bool(v) for k, v in P1["gates"].items()}, delta=P1["delta"])); return None
        log(f"!! {name}: PC1 failed; continuing as EXPLORATORY (no confirmatory claim).")
        if P1["delta"] is None: P1.update(delta=300, Q=5)
    delta, Q = P1["delta"], P1["Q"]
    S = build_dataset(D, Xraw, delta, Q, heldout=heldout)
    S.feature_names = fnames
    DS = SimpleNamespace(name=name, tag="heldout" if heldout else "main", S=S, P1=P1, exploratory=not P1["passed"],
                         spec=DATASET_SPECS[name])
    G = SimpleNamespace()
    G.X = to_t(S.X); G.y = to_t(S.y, torch.long); G.NB = to_t(S.NB, torch.long)
    G.loc = torch.full((S.N,), -1, dtype=torch.long, device=dev)
    # B8 inputs: [x, mean_R x, max_R x]
    parts = []
    for i in range(0, S.N, 20000):
        nb = G.NB[i:i + 20000]; mk = nb >= 0; g = G.X[nb.clamp_min(0)]
        mean = (g * mk.unsqueeze(-1)).sum(1) / mk.sum(1, keepdim=True).clamp_min(1)
        mx = g.masked_fill(~mk.unsqueeze(-1), -1e9).max(1).values
        parts.append(torch.cat([G.X[i:i + 20000], mean, mx], 1))
    G.Xagg = torch.cat(parts)
    DS.G = G
    DS.cs_tr, DS.cs_va = ChunkSet(S, S.tr_idx), ChunkSet(S, S.va_idx)
    DS.fr = prep_frames(S); DS.retained = [n for n, v in DS.fr.items() if v["keep"]]
    DS.blk, DS.nblk, DS.blkL, DS.blkflag = make_blocks(S)
    DS.yte = S.y[S.te_idx]; DS.yva = S.y[S.va_idx]
    STORE.put(cache_key + "|" + DS.tag, dict(passed=bool(P1["passed"]), delta=delta, Q=Q, gates={k: bool(v) for k, v in P1["gates"].items()},
              mixed_frac=float(P1.get("mixed_frac", float("nan"))), purity=float(P1.get("purity", float("nan"))),
              n=int(S.N), d=int(S.d), retained=DS.retained, frames={k: {a: (float(b) if not isinstance(b, (np.ndarray, bool)) else bool(b)) for a, b in v.items() if a in ("frac_te", "frac_te_atk", "frac_va_atk", "keep")} for k, v in DS.fr.items()}))
    return DS

# ------------------------------ teachers -------------------------------------
class ChunkSet:
    def __init__(self, S, ids):
        self.ids = ids; _, hid = np.unique(S.host[ids], return_inverse=True)
        starts = np.r_[0, np.flatnonzero(np.diff(hid)) + 1]
        self.hid = hid; self.pos = np.arange(len(ids)) - starts[hid]
        self.nh = int(hid.max()) + 1 if len(ids) else 0
        self.mul = int(self.pos.max()) // CLEN + 3 if len(ids) else 3

def train_teacher(DS, seed, ids, ttype, epochs=None, bs=None):
    S, G = DS.S, DS.G; epochs = epochs or ASSUMED["teacher_epochs"]; bs = bs or ASSUMED["teacher_batch"]
    torch.manual_seed(seed)
    model = (CXT if ttype == "CXT" else PFT)(S.d).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=ASSUMED["teacher_lr"])
    ids = to_t(ids, torch.long)
    for ep in range(epochs):
        perm = ids[torch.randperm(len(ids), device=dev)]
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            loss = F.cross_entropy(teacher_logits(model, G.X, G.NB, b), G.y[b])
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval(); return model

@torch.no_grad()
def teacher_predict(model, DS, ids, bs=4096):
    ids = to_t(ids, torch.long); out = []
    for i in range(0, len(ids), bs): out.append(teacher_logits(model, DS.G.X, DS.G.NB, ids[i:i + bs]))
    return torch.cat(out)

def teacher_ctx(DS, z, z_dep, flagged=0):
    """wrap logits; precompute B9's hard Dia(atk) of the teacher over every flow's window (kappa=1)."""
    G = DS.G; pt1 = F.softmax(z, 1)[:, 1]
    dia = torch.where(G.NB >= 0, pt1[G.NB.clamp_min(0)], torch.full_like(G.NB, -1, dtype=torch.float32)).max(1).values
    return SimpleNamespace(z=z, z_dep=z_dep, flagged=flagged, dia_t=dia)

def get_teacher(DS, tseed, ttype="CXT", insample=False):
    """cross-fitted teacher: OOF logits on train flows, deployed logits on val/test. Cached on disk (float32)."""
    S = DS.S; path = os.path.join(RUN_DIR, f"teacher_{DS.name}_{DS.tag}_{ttype}_{tseed}.npz")
    if os.path.exists(path):
        z = np.load(path); zt, zd, fl = to_t(z["z"], torch.float32), to_t(z["z_dep"], torch.float32), int(z["flagged"])
    else:
        check_time(); t0 = time.time(); TE = ASSUMED["teacher_epochs"]
        dep = train_teacher(DS, tseed * 100 + 99, S.tr_idx, ttype); zd = teacher_predict(dep, DS, np.arange(S.N)); zt = zd.clone(); fl = 0
        for q in range(S.Q):
            te_q = np.where((S.split == 0) & (S.fold == q))[0]
            if len(te_q) == 0: continue
            excl = np.zeros(S.N, bool)
            for c in np.unique(S.cap):
                if (c, q) in S.fold_lo:
                    lo, hi = S.fold_lo[(c, q)] - S.delta, S.fold_hi[(c, q)] + S.delta
                    excl |= (S.cap == c) & (S.sig >= lo) & (S.sig <= hi)
            tr_q = np.where((S.split == 0) & (S.fold != q) & ~excl)[0]
            mq = train_teacher(DS, tseed * 100 + q, tr_q, ttype)
            zt[to_t(te_q, torch.long)] = teacher_predict(mq, DS, te_q)
        for c in np.unique(S.cap):
            for k in (0, 1):
                tot = ((S.cap == c) & (S.split == 0) & (S.y == k)).sum()
                for q in range(S.Q):
                    sel = (S.cap == c) & (S.split == 0) & (S.y == k) & (S.fold == q)
                    if tot > 0 and sel.sum() / tot > 0.5:
                        ii = to_t(np.where(sel)[0], torch.long); zt[ii] = zd[ii]; fl += int(sel.sum())
        np.savez(path, z=zt.cpu().numpy(), z_dep=zd.cpu().numpy(), flagged=fl)
        STORE.put(f"teacher|{DS.name}|{DS.tag}|{ttype}|{tseed}", dict(secs=time.time() - t0, flagged=fl,
                  f1_te=macro_f1(DS.yte, zd[to_t(S.te_idx, torch.long)].argmax(1).cpu().numpy()),
                  f1_va=macro_f1(DS.yva, zd[to_t(S.va_idx, torch.long)].argmax(1).cpu().numpy())))
    return teacher_ctx(DS, zd if insample else zt, zd, fl)

# ----------------------- chunk / closure plans (exact closures) -------------
def shuffled_windows(S, anchors, rng):
    nb = S.NB[anchors]; Kc = nb.shape[1] - 1
    k = (nb[:, 1:] >= 0).sum(1); lo = np.where(nb >= 0, nb, np.iinfo(np.int64).max).min(1)
    wl = anchors - lo + 1; gs, ge = S.gstart[anchors], S.gend[anchors]
    avail = (ge - gs + 1) - wl; k = np.where(avail > 0, k, 0)
    u = (rng.random((len(anchors), Kc)) * np.maximum(avail, 1)[:, None]).astype(np.int64)
    pos = gs[:, None] + u; pos = np.where(pos >= lo[:, None], pos + wl[:, None], pos)
    out = np.full((len(anchors), Kc + 1), -1, np.int64); out[:, 0] = anchors
    out[:, 1:] = np.where(np.arange(Kc)[None, :] < k[:, None], pos, -1); return out

def make_plan(S, cs, rng, method, depth2=False, max_batches=None):
    off = rng.integers(0, CLEN, size=cs.nh)
    key = cs.hid.astype(np.int64) * cs.mul + (cs.pos + off[cs.hid]) // CLEN
    uniq, inv = np.unique(key, return_inverse=True)
    perm = np.argsort(inv, kind="stable"); cnt = np.bincount(inv); st = np.r_[0, np.cumsum(cnt)[:-1]]
    corder = rng.permutation(len(uniq)); raw, closures = [], []
    for bi in range(0, len(corder), NCHUNK):
        mem = np.concatenate([perm[st[c]:st[c] + cnt[c]] for c in corder[bi:bi + NCHUNK]]); a = cs.ids[mem]
        nb = S.NB[a]; cl = np.unique(nb[nb >= 0]); Wx = U = None
        if method == "X2":
            Wx = shuffled_windows(S, a, rng); U = np.unique(np.concatenate([cl, Wx[Wx >= 0]]))
        elif depth2:
            n2 = S.NB[cl]; U = np.unique(n2[n2 >= 0])
        raw.append((a, cl, U, Wx)); closures.append(cl)
        if max_batches and len(raw) >= max_batches: break
    nf = np.bincount(np.concatenate(closures), minlength=S.N)
    inv_nf = to_t(1.0 / np.maximum(nf, 1), torch.float32); batches = []
    for a, cl, U, Wx in raw:
        a_t, cl_t = to_t(a, torch.long), to_t(cl, torch.long)
        batches.append((a_t, cl_t, cl_t if U is None else to_t(U, torch.long), None if Wx is None else to_t(Wx, torch.long)))
    return batches, inv_nf

# ----------------------------- soft operators / losses ------------------------
def smax(u, mask, tau):
    n = mask.sum(1).clamp_min(1).to(u.dtype); v = (u / tau).masked_fill(~mask, float("-inf"))
    return tau * (torch.logsumexp(v, 1) - torch.log(n))
def smin(u, mask, tau):
    n = mask.sum(1).clamp_min(1).to(u.dtype); v = (-u / tau).masked_fill(~mask, float("-inf"))
    return -tau * (torch.logsumexp(v, 1) - torch.log(n))

def modal_mean_sq(us, ut, mask, tau):               # formulas {Box atk, Dia atk}; omega sums to 1 incl. duplicate p1 atom
    return (((smin(us, mask, tau) - smin(ut, mask, tau)) ** 2 + (smax(us, mask, tau) - smax(ut, mask, tau)) ** 2) / 2).mean()

def rkd_loss(zs, zt):
    def pd_(z):
        d = torch.cdist(z, z); pos = d[d > 0]; mu = pos.mean() if pos.numel() else d.new_ones(())
        return d / mu.clamp_min(1e-8)
    ld = F.smooth_l1_loss(pd_(zs), pd_(zt).detach())
    n = zs.shape[0]; T = PAPER["rkd_triplets"]
    i, j, k = (torch.randint(0, n, (T,), device=zs.device) for _ in range(3))
    def ang(z):
        v1, v2 = z[i] - z[j], z[k] - z[j]
        return (v1 * v2).sum(1) / (v1.norm(dim=1) * v2.norm(dim=1) + 1e-8)
    la = F.smooth_l1_loss(ang(zs), ang(zt).detach())
    w = PAPER["rkd_weights"]; return w[0] * ld + w[1] * la

TERM_METHODS = ["X0", "X1", "X2", "MAND", "B3", "B4", "B6", "B7"]

def run_losses(net, Xin, G, ZT, method, hp, batch, inv_nf, tau, ab, eps6=None):
    a, cl, U, Wx = batch
    z = net(Xin[U]); G.loc[U] = torch.arange(U.numel(), device=dev)
    zc = z if U is cl else z[G.loc[cl]]
    w = inv_nf[cl]; ws = w.sum(); th = hp["theta"]; zt = ZT.z[cl]
    ce = (w * F.cross_entropy(zc, G.y[cl], reduction="none")).sum() / ws
    pt = F.softmax(zt / th, 1); logps = F.log_softmax(zc / th, 1)
    if method == "B9":
        q = pt.clone(); q[:, 0] = pt[:, 0] * torch.exp(-hp["lam"] * F.relu(ZT.dia_t[cl] - 0.5)); q = q / q.sum(1, keepdim=True)
        kd_f = th * th * (q * (torch.log(q.clamp_min(1e-12)) - logps)).sum(1)
    else:
        kd_f = th * th * (pt * (torch.log(pt.clamp_min(1e-12)) - logps)).sum(1)
    kd = (w * kd_f).sum() / ws; terms, aux = [], {}
    if method in ("MAND", "X0", "X1", "X2", "B6"):
        ka = 1.0 if ab.get("kappa1") else hp["kappa"]
        W = a[:, None] if method == "X0" else (Wx if method == "X2" else G.NB[a])
        mask = W >= 0; loc = G.loc[W.clamp_min(0)]
        ps = F.softmax(z / ka, 1)[:, 1]
        with torch.no_grad(): pta = F.softmax(ZT.z[U] / ka, 1)[:, 1]
        us = ps[loc]
        if method == "X1":
            n = mask.sum(1).clamp_min(1).to(us.dtype)
            terms.append((((us * mask).sum(1) / n - (pta[loc] * mask).sum(1) / n) ** 2).mean())
        elif method == "B6":
            yt = (1 - eps6) * G.y[W.clamp_min(0)].float() + eps6 / 2
            terms.append(modal_mean_sq(us, yt, mask, tau))
        elif ab.get("depth2") and method == "MAND":
            nbv = G.NB[cl]; mv = nbv >= 0; lv = G.loc[nbv.clamp_min(0)]
            ib_s, id_s = smin(ps[lv], mv, tau), smax(ps[lv], mv, tau)
            with torch.no_grad(): ib_t, id_t = smin(pta[lv], mv, tau), smax(pta[lv], mv, tau)
            pos = torch.searchsorted(cl, W.clamp_min(0))
            f_s = [smin(us, mask, tau), smax(us, mask, tau), smax(ib_s[pos], mask, tau), smin(id_s[pos], mask, tau)]
            with torch.no_grad():
                ut = pta[loc]; f_t = [smin(ut, mask, tau), smax(ut, mask, tau), smax(ib_t[pos], mask, tau), smin(id_t[pos], mask, tau)]
            terms.append(sum(((x - y) ** 2) for x, y in zip(f_s, f_t)).mean() / 4)
        else:
            terms.append(modal_mean_sq(us, pta[loc], mask, tau))
    elif method == "B3":
        terms.append(rkd_loss(zc, zt))
    elif method == "B4":
        W = G.NB[a]; mask = W >= 0; P2 = F.softmax(z, 1); loc = G.loc[W.clamp_min(0)]
        mean = (P2[loc] * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        terms.append(((P2[G.loc[a]] - mean) ** 2).sum(1).mean())
    elif method == "B7":
        p1 = F.softmax(zt, 1); h = -(p1 * torch.log(p1.clamp_min(1e-12))).sum(1) / math.log(2)
        le = (w * h * kd_f).sum() / ws
        W = G.NB[a]; mask = W >= 0; pos = torch.searchsorted(cl, W.clamp_min(0))
        lx = kd_f[pos].masked_fill(~mask, float("-inf")).max(1).values.mean()
        terms = [le, lx]; aux["hbar"] = (w * h).sum() / ws
    return ce, kd, terms, aux

def gammas(method, hp, wstar):
    if method in ("MAND", "X0", "X1", "X2", "B3", "B4", "B6"): return [wstar[method] * hp["mult"]]
    if method == "B7": return [wstar["B7"][0] * hp["mult_e"], wstar["B7"][1] * hp["mult_x"]]
    return []

def assemble(method, hp, ce, kd, terms, aux, gam, beta):
    if method == "B7":
        le_w, lx_w = gam
        return ce + beta * (kd + le_w * terms[0]) / (1 + le_w * aux["hbar"]) + lx_w * terms[1]
    return ce + beta * kd + sum(g * t for g, t in zip(gam, terms))

def train_student(DS, ZT, method, hp, seed, wstar, ab=None, E_run=None):
    S, G = DS.S, DS.G; ab = ab or {}; E = PAPER["E"]
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed); rng = np.random.default_rng(seed)
    Xin = G.Xagg if method == "B8" else G.X
    net = Student(Xin.shape[1]).to(dev); opt = torch.optim.Adam(net.parameters(), lr=hp["lr"])
    beta = 0.0 if method == "B1" else hp["beta"]
    gmax = gammas(method, hp, wstar) if wstar is not None else []
    eps6 = None
    if method == "B6":
        ka = hp["kappa"]; mmp = F.softmax(ZT.z[to_t(S.tr_idx, torch.long)] / ka, 1).max(1).values.mean().item()
        eps6 = (1 - mmp) / (1 - 1 / 2)
    tr_t = S.tr_idx
    for ep in range(1, (E_run or E) + 1):
        rho = 1.0 if ab.get("gamma_const") else min(1.0, 2.0 * ep / E)
        tau = hp.get("tau1", 0.05) if ab.get("tau_fixed") else hp.get("tau1", 0.05) ** min(1.0, 2.0 * ep / E)
        gam = [g * rho for g in gmax]
        if method == "B2i":
            perm = to_t(rng.permutation(tr_t), torch.long)
            for i in range(0, len(tr_t), CLEN * NCHUNK):
                b = perm[i:i + CLEN * NCHUNK]; z = net(G.X[b]); th = hp["theta"]; pt = F.softmax(ZT.z[b] / th, 1)
                kd = th * th * (pt * (torch.log(pt.clamp_min(1e-12)) - F.log_softmax(z / th, 1))).sum(1).mean()
                loss = F.cross_entropy(z, G.y[b]) + beta * kd
                opt.zero_grad(); loss.backward(); opt.step()
            continue
        batches, inv_nf = make_plan(S, DS.cs_tr, rng, method, depth2=bool(ab.get("depth2")) and method == "MAND")
        for batch in batches:
            ce, kd, terms, aux = run_losses(net, Xin, G, ZT, method, hp, batch, inv_nf, tau, ab, eps6)
            loss = assemble(method, hp, ce, kd, terms, aux, gam, beta)
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); return net

def calibrate_wstar(DS, ZT, method_list=TERM_METHODS):
    """PC4: w* = ||grad L_KD|| / ||grad term|| at epoch E/2 on a B2 student (calibration settings), validation batches, rounded to 2^k."""
    S, G = DS.S, DS.G; hp = dict(PAPER["calib"]); hp.update(mult=1.0, lam=1.0)
    net = train_student(DS, ZT, "B2", hp, ASSUMED["calibration_seed"], None, E_run=max(1, PAPER["E"] // 2))
    params = list(net.parameters()); rng = np.random.default_rng(123); res = {}
    for method in method_list:
        ratios = []
        eps6 = None
        if method == "B6":
            mmp = F.softmax(ZT.z[to_t(S.tr_idx, torch.long)] / hp["kappa"], 1).max(1).values.mean().item(); eps6 = 2 * (1 - mmp)
        batches, inv_nf = make_plan(S, DS.cs_va, rng, method, max_batches=ASSUMED["calibration_batches"])
        for batch in batches:
            ce, kd, terms, aux = run_losses(net, G.X, G, ZT, method, dict(hp, theta=hp["theta"]), batch, inv_nf, hp["tau1"], {}, eps6)
            gk = torch.autograd.grad(kd, params, retain_graph=True, allow_unused=True)
            nk = torch.sqrt(sum((g ** 2).sum() for g in gk if g is not None))
            rr = []
            for t in terms:
                gm = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
                nm = torch.sqrt(sum((g ** 2).sum() for g in gm if g is not None)).clamp_min(1e-12); rr.append((nk / nm).item())
            ratios.append(rr)
        r = np.mean(np.array(ratios), 0); pw = [float(2.0 ** round(math.log2(max(x, 1e-6)))) for x in r]
        res[method] = pw if method == "B7" else pw[0]
    return res
# ============================================================================
# ================================ STAGES =====================================
# ============================================================================
ASSUMED["grid_shift_ratio_definition"] = "(gamma*||grad term||)/(beta*||grad KD||) on 8 validation batches after training the selected config (seed 401, reserved teacher 1)"
ASSUMED["pc2_aggregation"] = "mean over the two reserved teacher seeds"
DEFAULT_HP = dict(lr=1e-3, beta=1.0, theta=2.0, kappa=2.0, tau1=0.05, mult=1.0, lam=1.0, mult_e=1.0, mult_x=1.0)
HP_EXTRA = {"B1": [], "B2": [], "B2i": [], "B8": [], "B3": ["mult"], "B4": ["mult"], "B9": ["lam"], "X0": ["kappa", "mult"],
            "X1": ["kappa", "mult"], "X2": ["kappa", "tau1", "mult"], "MAND": ["kappa", "tau1", "mult"],
            "B6": ["kappa", "tau1", "mult"], "B7": ["mult_e", "mult_x"]}
MAIN_METHODS = ["B1", "B2", "B2i", "X0", "X1", "X2", "MAND", "B3", "B4", "B6", "B7", "B9", "B8", "B10"]
ABL = {"depth2": dict(depth2=True), "tau_fixed": dict(tau_fixed=True), "gamma_const": dict(gamma_const=True),
       "kappa1": dict(kappa1=True), "insample": {}, "pft": {}}
def hp_key(hp): return json.dumps({k: hp[k] for k in sorted(hp)}, separators=(",", ":"), default=float)
def pair_students(i):                       # teacher 11+i <-> students 301+2i, 302+2i  (generalised)
    n = len(PAPER["student_seeds"]) // len(PAPER["teacher_seeds"]); return PAPER["student_seeds"][n * i:n * i + n]

# ---------------------------- B10 (tree references) --------------------------
def b10_candidates():
    c = [dict(kind="rf", trees=t, depth=d) for t in PAPER["b10_rf"]["trees"] for d in PAPER["b10_rf"]["depth"]]
    c += [dict(kind="xgb", n=n, depth=d, lr=l) for n in PAPER["b10_xgb"]["n"] for d in PAPER["b10_xgb"]["depth"] for l in PAPER["b10_xgb"]["lr"]]
    return c[:2] if PROFILE == "smoke" else c

def run_b10(DS, hp, sseed, keep=True):
    S = DS.S; t0 = time.time()
    if hp["kind"] == "rf":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(n_estimators=hp["trees"], max_depth=hp["depth"], random_state=sseed, n_jobs=-1)
    else:
        try: from xgboost import XGBClassifier
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "xgboost"]); from xgboost import XGBClassifier
        clf = XGBClassifier(n_estimators=hp["n"], max_depth=hp["depth"], learning_rate=hp["lr"], random_state=sseed, tree_method="hist", n_jobs=-1)
    clf.fit(S.X[S.tr_idx], S.y[S.tr_idx])
    p_te = clf.predict_proba(S.X[S.te_idx])[:, 1]; p_va = clf.predict_proba(S.X[S.va_idx])[:, 1]
    return finish_record(DS, p_va > 0.5, p_te > 0.5, p_te, None, time.time() - t0, None, keep)

def finish_record(DS, pred_va, pred_te, p_te, z_dep_te_pred, secs, net, keep=True):
    rec = dict(f1_va=macro_f1(DS.yva, pred_va.astype(int)), f1_te=macro_f1(DS.yte, pred_te.astype(int)),
               fnr_te=float(((DS.yte == 1) & (~pred_te)).sum() / max((DS.yte == 1).sum(), 1)), secs=secs, n_te=int(len(pred_te)),
               fid=None if z_dep_te_pred is None else float((pred_te.astype(int) == z_dep_te_pred).mean()))
    if keep: rec["pred"] = b64(np.packbits(pred_te.astype(np.uint8))); rec["p"] = b64(p_te.astype(np.float32))
    if net is not None: rec["w"] = b64(torch.cat([p.detach().flatten() for p in net.parameters()]).cpu().numpy().astype(np.float32))
    return rec

def rec_pred(rec): return np.unpackbits(unb64(rec["pred"], np.uint8))[:rec["n_te"]].astype(int)
def rec_p(rec):    return unb64(rec["p"], np.float32)

def do_run(DS, method, hp, tseed, sseed, wstar, ttype="CXT", variant="base", save_w=False):
    key = f"run|{DS.name}|{DS.tag}|{variant}|{ttype}|{method}|{tseed}|{sseed}|{hp_key(hp)}"
    if STORE.has(key): return STORE.get(key)
    check_time()
    if method == "B10": rec = run_b10(DS, hp, sseed, keep=(variant != "tune"))
    else:
        S = DS.S; ZT = get_teacher(DS, tseed, ttype, insample=(variant == "insample")); t0 = time.time()
        net = train_student(DS, ZT, method, hp, sseed, wstar, ab=ABL.get(variant, {}))
        Xin = DS.G.Xagg if method == "B8" else DS.G.X
        with torch.no_grad(): z = net(Xin)
        pv, pt = z[to_t(S.va_idx, torch.long)].argmax(1).cpu().numpy(), z[to_t(S.te_idx, torch.long)].argmax(1).cpu().numpy()
        p = F.softmax(z[to_t(S.te_idx, torch.long)], 1)[:, 1].cpu().numpy()
        zd = ZT.z_dep[to_t(S.te_idx, torch.long)].argmax(1).cpu().numpy()
        rec = finish_record(DS, pv.astype(bool), pt.astype(bool), p, zd, time.time() - t0, net if save_w else None, keep=(variant != "tune"))
    STORE.put(key, rec); return rec

# ------------------------------ PC2 / PC4 ------------------------------------
def stage_pc2_pc4(DS):
    key = f"pc2|{DS.name}|{DS.tag}"
    if STORE.has(key): return STORE.get(key)
    S = DS.S; res = {}
    va_nb = compact(S, S.va_idx, S.NB); yv = S.y[S.va_idx]
    ymn, ymx = yv.copy(), yv.copy()
    for k in range(1, va_nb.shape[1]):
        col = va_nb[:, k]; m = col >= 0; v = yv[np.clip(col, 0, None)]
        ymn = np.where(m, np.minimum(ymn, v), ymn); ymx = np.where(m, np.maximum(ymx, v), ymx)
    mixv = ymn != ymx
    for ttype in ("CXT", "PFT"):
        sp_all, sp_mix = [], []
        for rs in PAPER["reserved_teachers"]:
            ZT = get_teacher(DS, rs, ttype)
            p = F.softmax(ZT.z_dep / 2.0, 1)[:, 1].cpu().numpy()[S.va_idx]       # kappa=2, deployed teacher on validation
            f = hard_formulas(p, va_nb); sp = f["Dia"] - f["Box"]
            sp_all.append(sp.mean()); sp_mix.append(sp[mixv].mean() if mixv.any() else float("nan"))
        res[ttype] = dict(spread_all=float(np.mean(sp_all)), spread_mixed=float(np.nanmean(sp_mix)))
    c, p_ = res["CXT"]["spread_mixed"], res["PFT"]["spread_mixed"]; th = PAPER["pc2_spread"]
    primary = "CXT" if not (c < th and p_ >= th) else "PFT"
    applicable = not (c < th and p_ < th)
    ZT1 = get_teacher(DS, PAPER["reserved_teachers"][0], primary)
    wstar = calibrate_wstar(DS, ZT1)
    out = dict(res=res, primary=primary, applicable=bool(applicable), wstar=wstar)
    STORE.put(key, out); log(f"[PC2/PC4] {DS.name}: {out}"); return out

# ------------------------------- tuning --------------------------------------
def tuning_candidates(method):
    if method == "B10": return b10_candidates()
    names = (["lr"] if method == "B1" else ["lr", "beta", "theta"]) + HP_EXTRA[method]
    grids = PAPER["grids"]; gl = [grids["mult"] if n in ("mult_e", "mult_x") else grids[n] for n in names]
    allc = [dict(DEFAULT_HP, **dict(zip(names, v))) for v in itertools.product(*gl)]
    budget = PAPER["tune_base"] + PAPER["tune_per_h"] * PAPER["n_h"][method]
    if len(allc) <= budget: return allc
    idx = np.random.default_rng(1000 + MAIN_METHODS.index(method)).choice(len(allc), size=budget, replace=False)
    return [allc[i] for i in sorted(idx)]

def tuning_teacher_assign():
    ts, rt = PAPER["tuning_seeds"], PAPER["reserved_teachers"]
    if len(ts) == 5: return [rt[0]] * 3 + [rt[1]] * 2
    return [rt[j % len(rt)] for j in range(len(ts))]

def score_cfg(DS, method, hp, wstar, ttype):
    per = {}
    for ss, rs in zip(PAPER["tuning_seeds"], tuning_teacher_assign()):
        r = do_run(DS, method, hp, rs, ss, wstar, ttype, variant="tune"); per.setdefault(rs, []).append(r["f1_va"])
    return float(np.mean([x for v in per.values() for x in v])), {k: float(np.mean(v)) for k, v in per.items()}

def end_ratio(DS, method, hp, wstar, ttype):
    ZT = get_teacher(DS, PAPER["reserved_teachers"][0], ttype)
    net = train_student(DS, ZT, method, hp, PAPER["tuning_seeds"][0], wstar)
    params = list(net.parameters()); rng = np.random.default_rng(321); G = DS.G; S = DS.S; out = []
    eps6 = None
    if method == "B6":
        mmp = F.softmax(ZT.z[to_t(S.tr_idx, torch.long)] / hp["kappa"], 1).max(1).values.mean().item(); eps6 = 2 * (1 - mmp)
    gam = gammas(method, hp, wstar); batches, inv_nf = make_plan(S, DS.cs_va, rng, method, max_batches=8)
    Xin = G.X
    for b in batches:
        ce, kd, terms, aux = run_losses(net, Xin, G, ZT, method, hp, b, inv_nf, hp.get("tau1", 0.05), {}, eps6)
        gk = torch.autograd.grad(kd, params, retain_graph=True, allow_unused=True)
        nk = torch.sqrt(sum((g ** 2).sum() for g in gk if g is not None)).clamp_min(1e-12) * hp["beta"]
        nt = 0.0
        for t, g_ in zip(terms, gam):
            gm = torch.autograd.grad(t, params, retain_graph=True, allow_unused=True)
            nt += g_ * torch.sqrt(sum((g ** 2).sum() for g in gm if g is not None))
        out.append(float(nt / nk))
    return float(np.mean(out))

def stage_tune(DS, pc):
    wstar, ttype = pc["wstar"], pc["primary"]; tuned = {}
    for method in MAIN_METHODS:
        key = f"hp|{DS.name}|{method}"
        if STORE.has(key): tuned[method] = STORE.get(key)["hp"]; continue
        cands = tuning_candidates(method); scores, per_teacher = [], []
        if method == "B10":
            for hp in cands:
                r = [do_run(DS, "B10", hp, 0, 401, None, ttype, variant="tune")["f1_va"]]; scores.append(float(np.mean(r))); per_teacher.append({})
        else:
            for hp in cands:
                s, pt = score_cfg(DS, method, hp, wstar, ttype); scores.append(s); per_teacher.append(pt)
        best = int(np.argmax(scores)); hp = cands[best]; shifted = None
        if method in TERM_METHODS:
            ratio = end_ratio(DS, method, hp, wstar, ttype); lo, hi = PAPER["grid_shift_lo"], PAPER["grid_shift_hi"]
            if ratio < lo or ratio > hi:
                g = PAPER["grids"]["mult"]; step = 10 ** 0.5
                new = [x * step for x in g] if ratio < lo else [x / step for x in g]
                extra = []
                for mv in new:
                    h2 = dict(hp); 
                    if method == "B7": h2.update(mult_e=mv, mult_x=mv)
                    else: h2["mult"] = mv
                    s, _ = score_cfg(DS, method, h2, wstar, ttype); extra.append((s, h2))
                sb, hb = max(extra, key=lambda t: t[0])
                if sb > scores[best]: hp, best = hb, -1; scores.append(sb)
                shifted = dict(ratio=ratio, direction="up" if ratio < lo else "down", new_grid=new)
            else: shifted = dict(ratio=ratio)
        rho = None
        if len(PAPER["reserved_teachers"]) == 2 and all(len(p) == 2 for p in per_teacher if p) and len(per_teacher) > 2:
            a = [p[PAPER["reserved_teachers"][0]] for p in per_teacher]; b = [p[PAPER["reserved_teachers"][1]] for p in per_teacher]
            rho = float(spearmanr(a, b).correlation) if np.std(a) > 0 and np.std(b) > 0 else None
        STORE.put(key, dict(hp=hp, n_trials=len(cands), best_score=float(max(scores)), grid_shift=shifted, rank_corr_reserved=rho))
        tuned[method] = hp; log(f"[tune] {DS.name} {method}: {len(cands)} trials best val F1 {max(scores):.4f} hp={hp} shift={shifted} rho={rho}")
    return tuned

# --------------------------- main / HC / held-out / ablations -----------------
def stage_runs(DS, pc, tuned, methods, tseeds, sseeds_fn, ttype=None, variant="base", save_w_methods=("MAND", "B2i")):
    ttype = ttype or pc["primary"]
    for i, ts in enumerate(tseeds):
        for ss in sseeds_fn(i):
            for m in methods:
                do_run(DS, m, tuned[m], ts, ss, pc["wstar"], ttype, variant, save_w=(m in save_w_methods and variant == "base"))
        log(f"  [{DS.name}/{DS.tag}/{variant}] teacher seed {ts} done ({hours():.2f} h elapsed)")

def collect(DS, method, tuned, tseeds, sseeds_fn, ttype, variant="base", field="f1_te", tag=None):
    """array [n_teacher, n_students_per_teacher] of a record field (None if any missing)."""
    out = []
    for i, ts in enumerate(tseeds):
        row = []
        for ss in sseeds_fn(i):
            key = f"run|{DS.name}|{tag or DS.tag}|{variant}|{ttype}|{method}|{ts}|{ss}|{hp_key(tuned[method])}"
            if not STORE.has(key): return None
            row.append(STORE.get(key)[field])
        out.append(row)
    return np.array(out, float)

def collect_rec(DS, method, tuned, ts, ss, ttype, variant="base"):
    return STORE.get(f"run|{DS.name}|{DS.tag}|{variant}|{ttype}|{method}|{ts}|{ss}|{hp_key(tuned[method])}")
# ============================================================================
# ================================ ANALYSIS ===================================
# ============================================================================
PAPER_REPORTED = {   # Table "Main results" + paired medians (IoT-23), exactly as printed in the paper
    "IoT-23": dict(f1=dict(B5=(0.938, 0.005), B1=(0.893, 0.009), B2=(0.915, 0.007), B2i=(0.913, 0.008), X0=(0.916, 0.007),
                           X1=(0.917, 0.007), X2=(0.914, 0.008), MAND=(0.924, 0.006)),
                   fnr=dict(B5=0.067, B1=0.121, B2=0.089, B2i=0.092, X0=0.087, X1=0.086, X2=0.090, MAND=0.078),
                   med=dict(B2=0.0087, B2i=0.0102, X0=0.0071, X1=0.0064, X2=0.0094), delta=300, Q=3, mixed_frac=0.047, purity_gt=0.95),
    "TON_IoT": dict(f1=dict(B5=(0.951, 0.004), B1=(0.914, 0.007), B2=(0.934, 0.006), B2i=(0.932, 0.006), X0=(0.935, 0.006),
                            X1=(0.936, 0.005), X2=(0.933, 0.006), MAND=(0.942, 0.005)), delta=120, Q=5),
    "MQTT-IoT-IDS2020": dict(f1=dict(B5=(0.965, 0.003), B1=(0.925, 0.006), B2=(0.941, 0.005), B2i=(0.939, 0.005), X0=(0.942, 0.005),
                                     X1=(0.943, 0.005), X2=(0.940, 0.005), MAND=(0.949, 0.004)), delta=60, Q=5)}

def seed_means(DS, method, tuned, ttype, variant="base", field="f1_te", tag=None):
    a = collect(DS, method, tuned, PAPER["teacher_seeds"] if (tag or DS.tag) == "main" else PAPER["heldout_teacher_seeds"],
                (lambda i: pair_students(i)) if (tag or DS.tag) == "main" else (lambda i: heldout_pairs(i)), ttype, variant, field, tag)
    return None if a is None else a.mean(1)

def heldout_pairs(i):
    n = len(PAPER["heldout_student_seeds"]) // len(PAPER["heldout_teacher_seeds"]); return PAPER["heldout_student_seeds"][n * i:n * i + n]

def tseeds_of(DS): return PAPER["teacher_seeds"] if DS.tag == "main" else PAPER["heldout_teacher_seeds"]
def pairs_of(DS):  return pair_students if DS.tag == "main" else heldout_pairs

def conf_array(DS, method, tuned, ttype, variant="base"):
    rows = []
    for i, ts in enumerate(tseeds_of(DS)):
        row = []
        for ss in pairs_of(DS)(i):
            key = f"run|{DS.name}|{DS.tag}|{variant}|{ttype}|{method}|{ts}|{ss}|{hp_key(tuned[method])}"
            if not STORE.has(key): return None
            row.append(block_conf(DS.yte, rec_pred(STORE.get(key)), DS.blk, DS.nblk))
        rows.append(row)
    return np.array(rows)

def boot_f1(cA, cB, reps, rng):
    nb = cA.shape[2]; d = np.empty(reps)
    for b in range(reps):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
        d[b] = (conf_to_f1(np.einsum("stbk,b->stk", cA, w)).mean(1) - conf_to_f1(np.einsum("stbk,b->stk", cB, w)).mean(1)).mean()
    return np.percentile(d, [2.5, 97.5])

def compare(DS, A, B, tuned, ttype, level, sesoi, caches, rng, metric="f1"):
    """paired seed-level comparison A over B. Returns dict with p, median, CI, win, worse, tost."""
    a, b = caches["seed"][A], caches["seed"][B]
    if a is None or b is None: return None
    d = a - b; p = paired_p(d); pw = paired_p(-d); med = float(np.median(d))
    if DS.blkflag == "seed-level only": lo = hi = float("nan"); ci_ok = True
    else:
        if metric == "f1":
            lo, hi = boot_f1(caches["conf"][A], caches["conf"][B], ASSUMED["bootstrap_replicates"], rng)
        else:
            lo, hi = boot_mj(caches["mj"][A], caches["mj"][B], ASSUMED["bootstrap_replicates"], rng)
        ci_ok = lo > 0
    win = (p < level) and (med >= sesoi) and ci_ok
    mg = PAPER["tost_margin"] if metric == "f1" else sesoi
    equiv = (not win) and max(paired_p(d + mg), paired_p(mg - d)) < level
    return dict(median=med, mean=float(d.mean()), p=p, ci=[float(lo), float(hi)], win=bool(win), worse=bool(pw < level),
                equivalent=bool(equiv), parts=dict(sig=bool(p < level), sesoi=bool(med >= sesoi), ci=bool(ci_ok)), n=len(d))

# ------------------------------ MJ_frame --------------------------------------
def mj_pairs(DS, z_dep):
    S = DS.S; pt = F.softmax(z_dep, 1)[:, 1].cpu().numpy(); out = {}
    for name in DS.retained:
        fi = DS.fr[name]; ft = hard_formulas(pt[S.te_idx], fi["te_nb"]); fv = hard_formulas(pt[S.va_idx], fi["va_nb"])
        for k in ft:
            prev = float((fv[k] > 0.5)[fi["ch_va"]].mean()) if fi["ch_va"].any() else 0.0
            if prev >= PAPER["mj_prev_min"]: out[(name, k)] = ft[k] > 0.5
    return out

def mj_counts(DS, pairs, p_s):
    """per (frame,formula) block-level intersection / union counts over anchors whose window changed."""
    inter = np.zeros((len(pairs), DS.nblk)); uni = np.zeros((len(pairs), DS.nblk)); cache = {}
    for j, ((name, k), T) in enumerate(pairs.items()):
        fi = DS.fr[name]
        if name not in cache: cache[name] = hard_formulas(p_s, fi["te_nb"])
        Sx = cache[name][k] > 0.5; ch = fi["ch_te"]
        inter[j] = np.bincount(DS.blk[ch & T & Sx], minlength=DS.nblk); uni[j] = np.bincount(DS.blk[ch & (T | Sx)], minlength=DS.nblk)
    return inter, uni

def mj_value(inter, uni):
    I, U = inter.sum(-1), uni.sum(-1); v = np.where(U > 0, I / np.maximum(U, 1), np.nan)
    return np.nanmean(v, -1)

def _mj_stat(I, U, w):
    Iw, Uw = np.einsum("stpb,b->stp", I, w), np.einsum("stpb,b->stp", U, w)
    return np.nanmean(np.where(Uw > 0, Iw / np.maximum(Uw, 1), np.nan), -1)

def boot_mj(cA, cB, reps, rng):
    (iA, uA), (iB, uB) = cA, cB; nb = iA.shape[-1]; d = np.empty(reps)
    for b in range(reps):
        w = np.bincount(rng.integers(0, nb, nb), minlength=nb).astype(float)
        d[b] = np.nanmean(_mj_stat(iA, uA, w).mean(1) - _mj_stat(iB, uB, w).mean(1))
    return np.nanpercentile(d, [2.5, 97.5])

def mj_cache(DS, method, tuned, ttype, variant="base"):
    if not DS.retained: return None
    I, U, M_ = [], [], []
    for i, ts in enumerate(tseeds_of(DS)):
        pairs = mj_pairs(DS, get_teacher(DS, ts, ttype).z_dep); ri, ru = [], []
        for ss in pairs_of(DS)(i):
            key = f"run|{DS.name}|{DS.tag}|{variant}|{ttype}|{method}|{ts}|{ss}|{hp_key(tuned[method])}"
            if not STORE.has(key): return None
            it, un = mj_counts(DS, pairs, rec_p(STORE.get(key))); ri.append(it); ru.append(un)
        I.append(ri); U.append(ru)
    # pair sets can differ across teacher seeds: align to the intersection by padding with zeros (union 0 => ignored)
    P = max(x.shape[0] for r in I for x in r)
    pad = lambda x: np.concatenate([x, np.zeros((P - x.shape[0], x.shape[1]))]) if x.shape[0] < P else x
    return (np.array([[pad(x) for x in r] for r in I]), np.array([[pad(x) for x in r] for r in U]))

# ------------------------------ main analysis ---------------------------------
def analyze(DS, pc, tuned, level):
    ttype = pc["primary"]; rng = np.random.default_rng(ASSUMED["bootstrap_seed"]); R = dict(level=level)
    methods = [m for m in MAIN_METHODS if m != "B10"] + ["B10"]
    caches = dict(seed={}, conf={}, mj={})
    for m in methods:
        sm = seed_means(DS, m, tuned, ttype); caches["seed"][m] = sm
        if sm is not None: caches["conf"][m] = conf_array(DS, m, tuned, ttype)
    caches["seed"]["B5"] = None
    tf = [STORE.get(f"teacher|{DS.name}|{DS.tag}|{ttype}|{t}") if STORE.has(f"teacher|{DS.name}|{DS.tag}|{ttype}|{t}") else None for t in tseeds_of(DS)]
    teach_te = np.array([x["f1_te"] for x in tf]) if all(tf) else None; teach_va = np.array([x["f1_va"] for x in tf]) if all(tf) else None
    R["table"] = {}
    if teach_te is not None: R["table"]["B5"] = dict(f1=float(teach_te.mean()), sd=float(teach_te.std(ddof=1)) if len(teach_te) > 1 else 0.0)
    for m in methods:
        sm = caches["seed"][m]
        if sm is not None:
            fn = seed_means(DS, m, tuned, ttype, field="fnr_te"); fid = seed_means(DS, m, tuned, ttype, field="fid") if m != "B10" else None
            R["table"][m] = dict(f1=float(sm.mean()), sd=float(sm.std(ddof=1)) if len(sm) > 1 else 0.0, fnr=float(fn.mean()),
                                 fid=None if fid is None else float(fid.mean()))
    # MJ_frame
    if DS.retained:
        for m in ["B1", "B2", "B2i", "X0", "X1", "X2", "MAND", "B3", "B4", "B6", "B7", "B9", "B8"]:
            if caches["seed"].get(m) is not None: caches["mj"][m] = mj_cache(DS, m, tuned, ttype)
        for m, c in caches["mj"].items():
            if c is not None: R["table"][m]["mj"] = float(np.nanmean(mj_value(c[0], c[1])))
    # headroom check (validation only)
    hc = {}
    va = {m: seed_means(DS, m, tuned, ttype, field="f1_va") for m in ["B2", "B2i", "X0", "X1", "X2", "MAND", "B8"]}
    if teach_va is not None and va["B2"] is not None and va["B2i"] is not None:
        ceil = teach_va if ttype == "PFT" or va["B8"] is None else np.minimum(teach_va, va["B8"])
        hc["eta"] = float((ceil - np.maximum(va["B2"], va["B2i"])).mean())
        mrng = np.random.default_rng(ASSUMED["mde_seed"])
        for name, comps in (("C1", ["B2", "B2i"]), ("X", ["X0", "X1", "X2"])):
            if va["MAND"] is None or any(va[c] is None for c in comps): continue
            sd = max(float(np.std(va["MAND"] - va[c], ddof=1)) for c in comps)
            hc["mde_" + name] = float(mde_power(sd, len(va["MAND"]), level, PAPER["hc"]["power"], PAPER["hc"]["sd_inflate"], ASSUMED["mde_mc_reps"], mrng))
        hc["C1_informative"] = bool(hc.get("eta", -1) >= PAPER["hc"]["eta_min"] and hc.get("mde_C1", 9) <= PAPER["hc"]["mde_max"])
        hc["X_testable"] = bool(hc.get("mde_X", 9) <= PAPER["hc"]["mde_max"])
    R["hc"] = hc
    # claims
    cmp = {}
    for B in ["B2", "B2i", "X0", "X1", "X2", "B3", "B4", "B6", "B7", "B9", "B8", "B10"]:
        cmp[B] = compare(DS, "MAND", B, tuned, ttype, level, PAPER["sesoi_f1"], caches, rng)
    R["cmp"] = cmp
    w = lambda k: bool(cmp[k] and cmp[k]["win"])
    R["C1"] = w("B2") and w("B2i"); R["C3"] = R["C1"] and w("X0") and w("X1"); R["C4"] = R["C1"] and w("X2")
    # C2
    c2 = {}
    if DS.retained and all(caches["mj"].get(m) is not None for m in ["MAND", "B2", "B2i", "X1"]):
        for B in ["B2", "B2i", "X1"]:
            caches2 = dict(seed={m: np.nanmean(mj_value(c[0], c[1]), 1) for m, c in caches["mj"].items() if c is not None}, mj=caches["mj"], conf={})
            c2[B] = compare(DS, "MAND", B, tuned, ttype, level, PAPER["sesoi_mj"], caches2, rng, metric="mj")
        rej = holm([c2[B]["p"] for B in c2], 0.05)
        for B, r in zip(c2, rej): c2[B]["holm_reject"] = bool(r); c2[B]["win_holm"] = bool(r and c2[B]["median"] >= PAPER["sesoi_mj"] and c2[B]["parts"]["ci"])
    R["C2"] = c2
    # AF
    af = [B for B in ["B3", "B4", "B6", "B7", "B9"] if cmp.get(B)]
    if af:
        rej = holm([cmp[B]["p"] for B in af], 0.05); R["AF"] = {B: dict(median=cmp[B]["median"], p=cmp[B]["p"], ci=cmp[B]["ci"], holm_reject=bool(r)) for B, r in zip(af, rej)}
    R["blocks"] = dict(n=int(DS.nblk), length=DS.blkL, flag=DS.blkflag)
    R["_caches"] = caches
    return R

# ------------------------------ ablations ------------------------------------
def analyze_ablations(DS, pc, tuned, base_seed):
    out = {}; ttype = pc["primary"]
    if base_seed is None: return out
    for v in ABL:
        tt = ("PFT" if ttype == "CXT" else "CXT") if v == "pft" else ttype
        sm = seed_means(DS, "MAND", tuned, tt, variant=v)
        if sm is None: continue
        d = sm - base_seed; out[v] = dict(f1=float(sm.mean()), delta_vs_base_median=float(np.median(d)), p_worse=paired_p(-d), p_better=paired_p(d))
    return out

# ------------------------------ robustness R1-R4 ------------------------------
def load_student(DS, rec):
    net = Student(DS.S.d).to(dev); w = torch.as_tensor(unb64(rec["w"], np.float32).copy(), device=dev); k = 0
    for p in net.parameters(): n = p.numel(); p.data.copy_(w[k:k + n].view_as(p)); k += n
    return net.eval()

def perturb_windows(DS, nb, kind, rng):
    S = DS.S; te = S.te_idx; sig = S.sig[te]; y = S.y[te]; host = S.host[te]; n, K = nb.shape; rb = PAPER["robust"]
    out = nb.copy(); mem = nb >= 0; mem[:, 0] = False
    if kind == "R2":
        ev = rng.random((n, K)) < rb["p_event"]; typ = rng.integers(0, 3, (n, K)); dl = rng.random((n, K)) * (S.delta / 2)
        sm = sig[np.clip(nb, 0, None)]; rem = mem & ev & ((typ == 0) | ((typ == 1) & (sm + dl >= sig[:, None])))
        out[rem] = -1; return out
    ymn = np.where(nb >= 0, y[np.clip(nb, 0, None)], 2).min(1); ymx = np.where(nb >= 0, y[np.clip(nb, 0, None)], -1).max(1)
    if kind == "R3":
        pool = np.where(y == 1)[0]; tgt = np.where((ymn == 0) & (ymx == 0))[0]
        if len(pool) == 0 or len(tgt) == 0: return out
        pick = rng.choice(pool, size=len(tgt)); bad = host[pick] == host[tgt]
        for _ in range(5):
            if not bad.any(): break
            pick[bad] = rng.choice(pool, size=int(bad.sum())); bad = host[pick] == host[tgt]
        add = np.full((n, 1), -1, nb.dtype); add[tgt, 0] = pick; return np.concatenate([out, add], 1)
    if kind == "R4":
        pool = np.where(y == 0)[0]; tgt = np.where(y == 1)[0]
        if len(pool) == 0 or len(tgt) == 0: return out
        add = np.full((n, K), -1, nb.dtype); ins = rng.random((len(tgt), K)) < rb["benign_insert"]
        draw = rng.choice(pool, size=(len(tgt), K)); sub = add[tgt]; sub[ins] = draw[ins]; add[tgt] = sub
        return np.concatenate([out, add], 1)

def mj_generic(DS, p_s, p_t, nbs, chs, valid):
    vals = []
    for name in DS.retained:
        ch = chs[name]
        if not ch.any(): continue
        fs, ft = hard_formulas(p_s, nbs[name]), hard_formulas(p_t, nbs[name])
        for k in fs:
            if (name, k) not in valid: continue
            T, Sx = (ft[k] > 0.5)[ch], (fs[k] > 0.5)[ch]; u = (T | Sx).sum()
            if u > 0: vals.append((T & Sx).sum() / u)
    return float(np.mean(vals)) if vals else float("nan")

def robustness(DS, pc, tuned, base_cmp):
    ttype = pc["primary"]; rb = PAPER["robust"]; S = DS.S; res = {}; tests = []
    ts_list = tseeds_of(DS); per = {k: {"MAND": [], "B2i": []} for k in ("R1_0", "R1_1", "R1_2", "R2", "R3", "R4")}
    lo, hi = S.X[S.tr_idx].min(0), S.X[S.tr_idx].max(0)
    for i, ts in enumerate(ts_list):
        zt = get_teacher(DS, ts, ttype); p_t = F.softmax(zt.z_dep, 1)[:, 1].cpu().numpy()[S.te_idx]; valid = set(mj_pairs(DS, zt.z_dep).keys())
        rng = np.random.default_rng(555 + ts); pert = {}
        for kind in ("R2", "R3", "R4"):
            pert[kind] = {}
            for name in DS.retained:
                nb = DS.fr[name]["te_nb"]; npb = perturb_windows(DS, nb, kind, rng)
                a, b = np.sort(np.pad(nb, ((0, 0), (0, max(0, npb.shape[1] - nb.shape[1]))), constant_values=-1), 1), np.sort(npb, 1)
                pert[kind][name] = (npb, (a != b).any(1))
        for m in ("MAND", "B2i"):
            acc = {k: [] for k in per}
            for ss in pairs_of(DS)(i):
                key = f"run|{DS.name}|{DS.tag}|base|{ttype}|{m}|{ts}|{ss}|{hp_key(tuned[m])}"
                if not STORE.has(key): return None
                rec = STORE.get(key); p_s = rec_p(rec)
                if "w" not in rec: return None
                net = load_student(DS, rec); clean = rec["f1_te"]
                for j, sd_ in enumerate(rb["noise"]):
                    g = torch.Generator(device="cpu").manual_seed(900 + ts * 10 + j)
                    x = S.X[S.te_idx] + sd_ * torch.randn(len(S.te_idx), S.d, generator=g).numpy()
                    x = np.clip(x, lo, hi).astype(np.float32)
                    with torch.no_grad(): pr = net(to_t(x)).argmax(1).cpu().numpy()
                    acc[f"R1_{j}"].append(clean - macro_f1(DS.yte, pr))
                base_nb = {n: DS.fr[n]["te_nb"] for n in DS.retained}; base_ch = {n: DS.fr[n]["ch_te"] for n in DS.retained}
                clean_mj = mj_generic(DS, p_s, p_t, base_nb, base_ch, valid)
                for kind in ("R2", "R3", "R4"):
                    nbs = {n: pert[kind][n][0] for n in DS.retained}; chs = {n: pert[kind][n][1] for n in DS.retained}
                    acc[kind].append(clean_mj - mj_generic(DS, p_s, p_t, nbs, chs, valid))
            for k in per: per[k][m].append(float(np.nanmean(acc[k])) if acc[k] else float("nan"))
    names = {"R1_0": "R1 noise 0.05", "R1_1": "R1 noise 0.1", "R1_2": "R1 noise 0.2", "R2": "R2 drop/delay/dup", "R3": "R3 outlier", "R4": "R4 benign insertion"}
    for k in per:
        a, b = np.array(per[k]["MAND"]), np.array(per[k]["B2i"]); diff = a - b; mg = rb["f1_margin"] if k.startswith("R1") else rb["mj_margin"]
        ok = np.isfinite(diff).all() and len(diff) > 0
        res[k] = dict(name=names[k], mean_diff=float(np.nanmean(diff)) if ok else None, p=paired_p(mg - diff) if ok else None, evaluable=bool(ok), margin=mg)
    ev = [k for k in res if res[k]["evaluable"]]
    if ev:
        rej = holm([res[k]["p"] for k in ev], 0.05)
        for k, r in zip(ev, rej): res[k]["holm_noninferior"] = bool(r)
    return res

# ------------------------------ cost ------------------------------------------
def cost_table(DS, tuned):
    out = {}
    for m in ("B2", "MAND", "B8"):
        d_in = DS.S.d * (3 if m == "B8" else 1); net = Student(d_in); w = PAPER["student_width"]
        params = sum(p.numel() for p in net.parameters()); flops = 2 * (d_in * w + w * w + w * 2)
        torch.set_num_threads(1); x = torch.randn(1, d_in); net.eval()
        with torch.no_grad():
            for _ in range(50): net(x)
            t0 = time.perf_counter(); [net(x) for _ in range(1000)]; lat = (time.perf_counter() - t0) / 1000 * 1e3
        out[m] = dict(params=params, flops=flops, cpu_latency_ms=lat)
    return out

# ------------------------------ validation vs paper ---------------------------
def validate_against_paper(DS, R, pc):
    ref = PAPER_REPORTED.get(DS.name); rows = []
    if not ref: return rows
    for m, (mu, sd) in ref["f1"].items():
        got = R["table"].get(m)
        if got is None: rows.append((f"macro-F1 {m}", mu, None, "not run")); continue
        tol = max(0.01, 2 * sd); ok = abs(got["f1"] - mu) <= tol; rows.append((f"macro-F1 {m}", mu, got["f1"], "PASS" if ok else f"FAIL (|d|>{tol:.3f})"))
    if "med" in ref:
        for B, mv in ref["med"].items():
            c = R["cmp"].get(B)
            if c: rows.append((f"median diff MAND-{B}", mv, c["median"], "PASS" if abs(c["median"] - mv) <= 0.005 else "FAIL (|d|>0.005)"))
            if c: rows.append((f"win MAND>{B}", True, c["win"], "PASS" if c["win"] else "FAIL"))
    rows.append(("selected Delta", ref["delta"], DS.P1.get("delta"), "PASS" if DS.P1.get("delta") == ref["delta"] else "FAIL"))
    rows.append(("Q", ref["Q"], DS.S.Q, "PASS" if DS.S.Q == ref["Q"] else "FAIL"))
    if "mixed_frac" in ref: rows.append(("mixed-window fraction", ref["mixed_frac"], DS.P1.get("mixed_frac"), "PASS" if abs(DS.P1.get("mixed_frac", 9) - ref["mixed_frac"]) < 0.01 else "FAIL"))
    if "purity_gt" in ref: rows.append(("label purity > 0.95", True, DS.P1.get("purity", 0) > 0.95, "PASS" if DS.P1.get("purity", 0) > 0.95 else "FAIL"))
    return rows
# ============================================================================
# ============================ ORCHESTRATION / REPORT =========================
# ============================================================================
def heldout_conjunct(DSh, pc, tuned, level):
    ttype = pc["primary"]; rng = np.random.default_rng(ASSUMED["bootstrap_seed"] + 1); caches = dict(seed={}, conf={}, mj={})
    for m in ["MAND", "B2", "B2i", "X0", "X1"]:
        sm = seed_means(DSh, m, tuned, ttype); caches["seed"][m] = sm
        if sm is not None: caches["conf"][m] = conf_array(DSh, m, tuned, ttype)
    out = {B: compare(DSh, "MAND", B, tuned, ttype, level, PAPER["sesoi_f1"], caches, rng) for B in ["B2", "B2i", "X0", "X1"]}
    va = {m: seed_means(DSh, m, tuned, ttype, field="f1_va") for m in ["MAND", "B2", "B2i", "X0", "X1"]}
    tf = [STORE.get(f"teacher|{DSh.name}|{DSh.tag}|{ttype}|{t}") if STORE.has(f"teacher|{DSh.name}|{DSh.tag}|{ttype}|{t}") else None for t in PAPER["heldout_teacher_seeds"]]
    hc = {}
    if all(tf) and all(v is not None for v in va.values()):
        teach = np.array([x["f1_va"] for x in tf]); hc["eta"] = float((teach - np.maximum(va["B2"], va["B2i"])).mean())
        mrng = np.random.default_rng(ASSUMED["mde_seed"])
        hc["mde"] = float(mde_power(max(float(np.std(va["MAND"] - va[c], ddof=1)) for c in ["B2", "B2i", "X0", "X1"]), len(teach), level,
                                    PAPER["hc"]["power"], PAPER["hc"]["sd_inflate"], ASSUMED["mde_mc_reps"], mrng))
    c1 = all(out[b] and out[b]["win"] for b in ("B2", "B2i")); c3 = all(out[b] and out[b]["win"] for b in ("X0", "X1"))
    hc_ok = bool(hc and hc.get("eta", -1) >= PAPER["hc"]["eta_min"] and hc.get("mde", 9) <= PAPER["hc"]["mde_max"])
    return dict(cmp=out, hc=hc, hc_ok=hc_ok, c1_conjunct=bool(c1 and hc_ok), c3_conjunct=bool(c3 and hc_ok))

def determinism_check(DS, pc, tuned):
    """train the same student twice with the same seed; report max |logit difference| (0.0 = bitwise repeatable on this device)."""
    key = f"determinism|{DS.name}"
    if STORE.has(key): return STORE.get(key)
    ZT = get_teacher(DS, PAPER["teacher_seeds"][0], pc["primary"]); out = {}
    for m in ("B2", "MAND"):
        zs = []
        for _ in range(2):
            net = train_student(DS, ZT, m, tuned[m], 999, pc["wstar"], E_run=1)
            with torch.no_grad(): zs.append(net(DS.G.X).cpu().numpy())
        out[m] = float(np.abs(zs[0] - zs[1]).max())
    out["device"] = str(dev); out["torch"] = torch.__version__; STORE.put(key, out); return out


# ============================ BUILT-IN SELF TESTS ============================
def selftest_stats():
    res = {}
    cdf = signed_rank_cdf(10); res["signed-rank P(W<=0)=2^-10"] = abs(cdf[0] - 2 ** -10) < 1e-15
    r = np.random.default_rng(0); worst = 0
    for _ in range(100):
        d = r.normal(0.3, 1, 10); rk = np.argsort(np.argsort(np.abs(d))) + 1; wneg = int(sum(rr for rr, v in zip(rk, d) if v < 0))
        worst = max(worst, abs(cdf[wneg] - wilcoxon(d, alternative="greater", method="exact").pvalue))
    res["exact CDF == scipy exact p"] = worst < 1e-12
    u = torch.tensor([[0.9, 0.1, 0.5]]); mk = torch.tensor([[True, True, True]])
    res["soft max/min limits"] = abs(smax(u, mk, 1e-3).item() - 0.9) < 1e-2 and abs(smin(u, mk, 1e-3).item() - 0.1) < 1e-2
    res["n=1 window: smax=smin=u"] = abs(smax(torch.tensor([[0.7, 0.]]), torch.tensor([[True, False]]), 0.3).item() - 0.7) < 1e-6
    res["modal loss, window {w} = squared error"] = abs(modal_mean_sq(torch.tensor([[0.8]]), torch.tensor([[0.5]]), torch.tensor([[True]]), 0.1).item() - 0.09) < 1e-6
    return res

def selftest_dataset(DS):
    S = DS.S; rng = np.random.default_rng(3); res = {}
    for method in ("MAND", "X2"):
        batches, inv = make_plan(S, DS.cs_tr, rng, method); anc = torch.cat([b[0] for b in batches]).cpu().numpy()
        res[f"[{method}] every train flow is an anchor exactly once per epoch"] = len(anc) == len(S.tr_idx) == len(np.unique(anc))
        cl = torch.cat([b[1] for b in batches]).cpu().numpy(); nf = np.bincount(cl, minlength=S.N)
        res[f"[{method}] weights 1/n_f sum to 1 per flow"] = bool(np.allclose(inv.cpu().numpy()[nf > 0] * nf[nf > 0], 1.0))
        res[f"[{method}] closure size <= {NCHUNK}*(c+2m)"] = max(len(b[1]) for b in batches) <= NCHUNK * (CLEN + 2 * M)
        res[f"[{method}] closure contains every window member"] = all(set(S.NB[b[0].cpu().numpy()][S.NB[b[0].cpu().numpy()] >= 0].tolist()) <= set(b[1].cpu().numpy().tolist()) for b in batches[:50])
    res["no window crosses a split"] = all((S.split[S.NB[i][S.NB[i] >= 0]] == S.split[i]).all() for i in range(0, S.N, 7))
    ok = True
    for c in np.unique(S.cap):
        m = S.cap == c
        if (m & (S.split == 1)).any() and (m & (S.split == 0)).any(): ok &= S.sig[m & (S.split == 1)].min() - S.sig[m & (S.split == 0)].max() >= S.delta - 1e-6
        if (m & (S.split == 2)).any() and (m & (S.split == 1)).any(): ok &= S.sig[m & (S.split == 2)].min() - S.sig[m & (S.split == 1)].max() >= S.delta - 1e-6
    res["guard gaps >= Delta"] = bool(ok)
    leak = False
    for q in range(S.Q):
        for c in np.unique(S.cap):
            if (c, q) in S.fold_lo:
                lo, hi = S.fold_lo[(c, q)] - S.delta, S.fold_hi[(c, q)] + S.delta
                tr_q = (S.split == 0) & (S.fold != q) & ~((S.cap == c) & (S.sig >= lo) & (S.sig <= hi))
                leak |= bool((tr_q & (S.cap == c) & (S.sig >= lo) & (S.sig <= hi)).any())
    res["cross-fit: teacher q never trains within Delta of fold q"] = not leak
    return res

def run_selftests(res, label):
    bad = [k for k, v in res.items() if not v]
    log(f"[selftest:{label}] " + ("all %d passed" % len(res) if not bad else f"FAILED: {bad}"))
    STORE.put(f"selftest|{label}", {k: bool(v) for k, v in res.items()})
    if bad: raise RuntimeError(f"self-test failed ({label}): {bad}  -> results would not be trustworthy")

def run_dataset(name, report):
    spec = DATASET_SPECS[name]; DS = prepare(name)
    if DS is None: report[name] = dict(status="PC1_FAIL (no claim; paper rule)"); return
    run_selftests(selftest_dataset(DS), f"{name} dataset")
    level = PAPER["p_levels"]["confirmatory" if spec["role"] == "confirmatory" else "development"]
    rep = dict(status="EXPLORATORY (PC1 failed)" if DS.exploratory else "ok", role=spec["role"], level=level,
               pc1=STORE.get(f"pc1|{name}|main"), n_flows=int(DS.S.N), d=int(DS.S.d), blocks=dict(n=DS.nblk, L=DS.blkL, flag=DS.blkflag),
               retained_frames=DS.retained)
    report[name] = rep
    pc = stage_pc2_pc4(DS); rep["pc2_pc4"] = pc
    if not pc["applicable"]: rep["status"] = "MAND not applicable (PC2: both spreads < 0.05)"; return
    tuned = stage_tune(DS, pc); rep["tuned"] = tuned; rep["determinism"] = determinism_check(DS, pc, tuned)
    stage_runs(DS, pc, tuned, MAIN_METHODS, PAPER["teacher_seeds"], pair_students)
    R = analyze(DS, pc, tuned, level); caches = R.pop("_caches"); rep["analysis"] = R
    rep["validation"] = validate_against_paper(DS, R, pc)
    base_seed = caches["seed"].get("MAND")
    if DS.P1.get("purity", 0) > PAPER["label_purity_trigger"] and not DS.exploratory:
        DSh = prepare(name, heldout=True)
        if DSh is not None:
            stage_runs(DSh, pc, tuned, ["B2", "B2i", "X0", "X1", "MAND"], PAPER["heldout_teacher_seeds"], heldout_pairs)
            rep["heldout"] = heldout_conjunct(DSh, pc, tuned, level)
    else: rep["heldout"] = "not triggered (label purity <= 0.95) or exploratory"
    A = rep["analysis"]; H = rep["heldout"]
    if isinstance(H, dict):      # paper: failed held-out conjunct or held-out HC withdraws the parent claim
        A["C1_final"] = bool(A["C1"] and H["c1_conjunct"]); A["C3_final"] = bool(A["C3"] and A["C1_final"] and H["c3_conjunct"]); A["C4_final"] = bool(A["C4"] and A["C1_final"])
    else: A["C1_final"], A["C3_final"], A["C4_final"] = A["C1"], A["C3"], A["C4"]
    for v in ABL:
        tt = ("PFT" if pc["primary"] == "CXT" else "CXT") if v == "pft" else pc["primary"]
        stage_runs(DS, pc, tuned, ["MAND"], PAPER["teacher_seeds"], pair_students, ttype=tt, variant=v)
    rep["ablations"] = analyze_ablations(DS, pc, tuned, base_seed)
    rep["robustness"] = robustness(DS, pc, tuned, R["cmp"]); rep["cost"] = cost_table(DS, tuned)
    rep["train_secs_mean"] = {m: float(np.mean([STORE.get(k)["secs"] for k in STORE.d if k.startswith(f"run|{name}|main|base|") and f"|{m}|" in k])) for m in MAIN_METHODS
                              if any(k.startswith(f"run|{name}|main|base|") and f"|{m}|" in k for k in STORE.d)}

def _clean(o):
    if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)): return o.item()
    if isinstance(o, np.bool_): return bool(o)
    return o

def write_report(report, complete):
    with open(os.path.join(RUN_DIR, "report.json"), "w") as f: json.dump(_clean(report), f, indent=1, default=str)
    L = [f"# MAND reproduction report ({PROFILE}, fingerprint {FPRINT}, complete={complete})", "",
         "Every assumed (unspecified-by-paper) parameter is in manifest.json -> ASSUMED. Results below are only claims where PC1 passed.", ""]
    for name, r in report.items():
        L += [f"## {name}: {r.get('status')}"]
        if "analysis" not in r: continue
        A = r["analysis"]; L += [f"level={A['level']} blocks={A['blocks']}", "", "| method | macro-F1 | paper | FNR | MJ_frame | fidelity |", "|---|---|---|---|---|---|"]
        ref = PAPER_REPORTED.get(name, {}).get("f1", {})
        for m, v in A["table"].items():
            L.append(f"| {m} | {v['f1']:.4f}±{v['sd']:.4f} | {ref.get(m, ('',))[0]} | {v.get('fnr', '')} | {v.get('mj', '')} | {v.get('fid', '')} |")
        L += ["", f"HC: {A['hc']}", "", "| MAND vs | median | p | CI | win | worse | equivalent |", "|---|---|---|---|---|---|---|"]
        for B, c in A["cmp"].items():
            if c: L.append(f"| {B} | {c['median']:+.4f} | {c['p']:.4f} | [{c['ci'][0]:+.4f},{c['ci'][1]:+.4f}] | {c['win']} | {c['worse']} | {c['equivalent']} |")
        L += ["", f"determinism check (max |logit diff|, 0 = bitwise repeatable): {r.get('determinism')}", f"C1={A['C1_final']}  C3={A['C3_final']}  C4={A['C4_final']} (before held-out: {A['C1']},{A['C3']},{A['C4']})  C2={ {k: v.get('win_holm') for k, v in A['C2'].items()} }", f"AF={A.get('AF')}",
              f"held-out: {r.get('heldout')}", f"ablations: {r.get('ablations')}", f"robustness: {r.get('robustness')}", f"cost: {r.get('cost')}", "",
              "### Validation against the paper's reported numbers", "| quantity | paper | reproduced | verdict |", "|---|---|---|---|"]
        for q, pv, gv, vd in r.get("validation", []): L.append(f"| {q} | {pv} | {gv} | {vd} |")
        L.append("")
    with open(os.path.join(RUN_DIR, "report.md"), "w") as f: f.write("\n".join(L))
    log("\n".join(L[:400]))

def main():
    global T_START
    T_START = time.time()
    try:
        from google.colab import drive; drive.mount("/content/drive")
    except Exception: pass
    log(COVERAGE); run_selftests(selftest_stats(), "statistics"); man = write_manifest()
    log(f"profile={PROFILE} fingerprint={FPRINT} run dir={RUN_DIR}\nASSUMED parameters ({len(ASSUMED)}), each must be disclosed in the paper/repo:")
    for k, v in ASSUMED.items(): log(f"  - {k}: {v}")
    gpu_note = "cuda" if torch.cuda.is_available() else "CPU ONLY (paper profile would take years)"
    log(f"device: {gpu_note}")
    report = {}; complete = True
    try:
        for name in RUN_DATASETS: run_dataset(name, report)
    except OutOfTime:
        complete = False
    write_report(report, complete)
    log("ALL STAGES COMPLETE" if complete else f"TIME BUDGET ({MAX_HOURS} h) REACHED -> re-run this cell to resume; finished runs are cached in {RUN_DIR}")

if __name__ == "__main__" or "google.colab" in sys.modules:
    main()
