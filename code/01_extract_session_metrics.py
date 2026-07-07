"""
Three-analysis EEG memory pipeline
====================================
Analysis 1 — Subsequent Memory Effect (SME)
    Encoding epochs split by whether the word was later correctly recalled.
    Both classes: same task state (passive reading). Only memory outcome differs.
    This is the cleanest memory-specific contrast.

Analysis 3 — Encoding vs Retrieval (ERS)
    Classic encoding/retrieval flip. High accuracy expected due to task-state
    differences but correlation with recall rate is the meaningful test.
    Retrieval window is [-2, -1s] pre-vocal to minimise motor prep confound.

All three analyses:
  - Use CSP (4 components, ledoit_wolf) + LDA inside a 5-fold stratified CV
  - Report mean accuracy, std, and permutation p-value (100 permutations)
  - Save CSP models, trajectories, and per-session metrics to CSV
"""

import os
import pickle
import warnings
import gc

import mne
import numpy as np
import pandas as pd

from mne.decoding import CSP
from scipy.signal import welch
from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold, cross_val_score, permutation_test_score
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

warnings.filterwarnings("ignore")
mne.set_log_level('WARNING')

# ==========================================================
# CONFIG
# ==========================================================

BASE_PATH = r"C:\Users\Neermita\Desktop\memory_and_task\ds004395"

SUBJECTS = ["LTP063", "LTP064", "LTP065", "LTP066", "LTP067"]
SESSIONS  = range(20)

# Epoch windows (seconds, relative to event onset)
ENC_TMIN,  ENC_TMAX  =  0.0,  1.0   # post word-onset
RET_TMIN,  RET_TMAX  = -2.0, -1.0   # pre-vocal, avoids motor prep window
# Analysis 2 uses the same RET window for both recalled/not-recalled

# Amplitude rejection (peak-to-peak, Volts). Raise or set None if too many drop.
REJECT_THRESHOLD = 500e-6 # 

# Minimum epochs per class to attempt classification
MIN_EPOCHS = 5

# Permutation test: number of permutations (100 is fast; use 1000 for publication)
N_PERMS = 10

# ==========================================================
# DIRECTORIES
# ==========================================================

RESULTS_DIR = "results"
for d in [
    RESULTS_DIR,
    os.path.join(RESULTS_DIR, "csp_sme"),
    os.path.join(RESULTS_DIR, "csp_ers"),
    os.path.join(RESULTS_DIR, "traj_sme"),
    os.path.join(RESULTS_DIR, "traj_ers"),
]:
    os.makedirs(d, exist_ok=True)

# One CSV per analysis so they're easy to load separately in R / Python
def init_csv(path, extra_cols):
    base_cols = [
        "subject", "session",
        "n_class0", "n_class1", "n_balanced",
        "acc", "acc_std",
        "theta_class0", "alpha_class0",
        "theta_class1", "alpha_class1",
    ]
    if not os.path.exists(path):
        pd.DataFrame(columns=base_cols + extra_cols).to_csv(path, index=False)

SME_CSV    = os.path.join(RESULTS_DIR, "sme_metrics.csv")
ERS_CSV    = os.path.join(RESULTS_DIR, "ers_metrics.csv")
FAILED_CSV = os.path.join(RESULTS_DIR, "failed_sessions.csv")

init_csv(SME_CSV,    ["n_words", "n_recalled", "n_not_recalled", "correct_recall_rate"])
init_csv(ERS_CSV,    ["n_words", "n_recalls", "raw_recall_rate", "correct_recall_rate"])

if not os.path.exists(FAILED_CSV):
    pd.DataFrame(columns=["subject", "session", "analysis", "reason"]
                 ).to_csv(FAILED_CSV, index=False)

# ==========================================================
# HELPERS
# ==========================================================

def make_pipeline():
    return Pipeline([
        ("csp", CSP(n_components=4, log=True, norm_trace=False, reg='ledoit_wolf')),
        ("lda", LinearDiscriminantAnalysis())
    ])


def run_classification(X, y, n_perms=N_PERMS):
    """
    5-fold stratified CV + permutation test.
    Returns (mean_acc, std_acc).
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf = make_pipeline()

    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    acc    = float(np.mean(scores))
    std    = float(np.std(scores))

    # _, _, pvalue = permutation_test_score(
    #     make_pipeline(), X, y,
    #     cv=cv, n_permutations=n_perms,
    #     scoring="accuracy", random_state=42, n_jobs=1
    # )

    return acc, std


def balance_classes(X0, X1, rng):
    """Subsample the larger class to match the smaller."""
    n = min(len(X0), len(X1))
    idx0 = rng.choice(len(X0), n, replace=False)
    idx1 = rng.choice(len(X1), n, replace=False)
    X = np.concatenate([X0[idx0], X1[idx1]], axis=0)
    y = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    return X, y, idx0, idx1, n


def apply_baseline(X, sfreq, duration=0.2):
    """Subtract mean of first `duration` seconds from every epoch × channel."""
    n_bl = int(round(duration * sfreq))
    return X - X[:, :, :n_bl].mean(axis=-1, keepdims=True)


def compute_bandpower(X, sfreq, fmin, fmax):
    """Mean bandpower in Hz band. X: (n_epochs, n_ch, n_times). Broadband data only."""
    freqs, psd = welch(X, fs=sfreq, axis=-1, nperseg=min(512, X.shape[-1]))
    idx = (freqs >= fmin) & (freqs <= fmax)
    return float(np.mean(psd[..., idx]))


def save_csp(X, y, subject, ses, tag, csp_dir, traj_dir, info):
    """Fit CSP on full data and save model + trajectory. Separate from CV estimate."""
    model = CSP(n_components=4, log=True, norm_trace=False, reg='ledoit_wolf')
    model.fit(X, y)
    with open(os.path.join(csp_dir, f"{subject}_ses{ses}_{tag}_csp.pkl"), "wb") as f:
        pickle.dump(model, f)
    info.save(
        os.path.join(csp_dir, f"{subject}_ses{ses}_{tag}_info.fif"), overwrite=True
    )
    np.savez(
        os.path.join(traj_dir, f"{subject}_ses{ses}_{tag}.npz"),
        X_csp=model.transform(X), y=y
    )


def log_failure(subject, ses, analysis, reason):
    print(f"    [{analysis}] SKIPPED: {reason}")
    pd.DataFrame([{
        "subject": subject, "session": ses,
        "analysis": analysis, "reason": reason
    }]).to_csv(FAILED_CSV, mode="a", header=False, index=False)


def make_epochs(raw, events_arr, tmin, tmax, reject_param):
    return mne.Epochs(
        raw, events_arr,
        tmin=tmin, tmax=tmax,
        baseline=None,   # applied manually
        detrend=1,       # linear detrend removes DC + slow drift per epoch
        preload=True,
        reject=reject_param,
        verbose=False
    )


def make_events_arr(onsets_sec, label, sfreq):
    arr = np.zeros((len(onsets_sec), 3), dtype=int)
    arr[:, 0] = (onsets_sec * sfreq).astype(int)
    arr[:, 2] = label
    return arr


def reject_overlapping(enc_onsets, ret_onsets, enc_win, ret_win):
    """
    Remove events whose epoch windows overlap across the two classes.
    Returns boolean keep masks.
    """
    keep_enc = np.ones(len(enc_onsets), dtype=bool)
    keep_ret = np.ones(len(ret_onsets), dtype=bool)
    for ri, r_on in enumerate(ret_onsets):
        rs, re = r_on + ret_win[0], r_on + ret_win[1]
        for ei, e_on in enumerate(enc_onsets):
            es, ee = e_on + enc_win[0], e_on + enc_win[1]
            if rs < ee and re > es:
                keep_enc[ei] = False
                keep_ret[ri] = False
    return keep_enc, keep_ret


# ==========================================================
# MAIN LOOP
# ==========================================================

for subject in SUBJECTS:
    print("\n" + "=" * 60)
    print(subject)
    print("=" * 60)

    for ses in SESSIONS:

        raw = raw_broad = None

        try:
            sub_dir = f"sub-{subject}"
            ses_dir = f"ses-{ses}"
            base    = os.path.join(BASE_PATH, sub_dir, ses_dir, "eeg")
            prefix  = f"{sub_dir}_{ses_dir}_task-ltpFR"

            eeg_file       = os.path.join(base, f"{prefix}_eeg.edf")
            event_file     = os.path.join(base, f"{prefix}_events.tsv")
            electrode_file = os.path.join(base, f"{sub_dir}_{ses_dir}_space-CapTrak_electrodes.tsv")

            if not os.path.exists(eeg_file):
                continue

            print(f"\n  Session {ses}")

            # --------------------------------------------------
            # LOAD + PREPROCESS
            # --------------------------------------------------

            raw_broad = mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)

            electrodes = pd.read_csv(electrode_file, sep="\t")
            bad_coords = electrodes[
                electrodes["x"].isna() | (electrodes["x"] == "n/a")
            ]["name"].tolist()
            to_drop = [ch for ch in bad_coords if ch in raw_broad.ch_names]
            if to_drop:
                raw_broad.drop_channels(to_drop)
            print(to_drop)
            electrodes = electrodes[
                electrodes["x"].notna() & (electrodes["x"] != "n/a")
            ]
            ch_pos = {
                row["name"]: [float(row["x"]), float(row["y"]), float(row["z"])]
                for _, row in electrodes.iterrows()
                if row["name"] in raw_broad.ch_names
            }
            raw_broad.set_montage(
                mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head"),
                on_missing="ignore"
            )
           # ==================================================
            # THE FIX: TARGETED SPATIAL CROPPING (6 CHANNELS)
            # ==================================================
            # The exact 6 anterior channels identified from the EGI cap montage
            target_blinks = ["E21", "E17", "E14", "E22", "E15", "E9"]
            
            # Safely check which of these channels are actually present in the data right now
            blinks_to_drop = [ch for ch in target_blinks if ch in raw_broad.ch_names]
            
            # 4. Drop only the ones that exist
            if blinks_to_drop:
                raw_broad.drop_channels(blinks_to_drop)
                print(f"    -> Spatially cropped {len(blinks_to_drop)} anterior channels to avoid blinks: {blinks_to_drop}")
            # ==================================================
            raw_broad.set_eeg_reference("average", projection=False)
            raw_broad.filter(l_freq=1.0, h_freq=40.0, verbose=False)

            # Narrow-band copy for CSP (theta-alpha)
            raw = raw_broad.copy()
            raw.filter(l_freq=4.0, h_freq=12.0, verbose=False)

            sfreq = raw.info["sfreq"]
            reject_param = dict(eeg=REJECT_THRESHOLD) if REJECT_THRESHOLD else None

            # --------------------------------------------------
            # EVENTS & BEHAVIORAL LABELS
            # --------------------------------------------------

            events_df = pd.read_csv(event_file, sep="\t")
            words   = events_df[events_df.trial_type == "WORD"].copy().reset_index(drop=True)
            recalls = events_df[events_df.trial_type == "REC_WORD"].copy().reset_index(drop=True)

            # Mark each studied word as recalled or not (per trial)
            words["recalled"] = 0
            for trial in words["trial"].unique():
                studied  = words.loc[words["trial"] == trial, "item_name"].astype(str)
                recalled_items = set(
                    recalls.loc[recalls["trial"] == trial, "item_name"].astype(str)
                )
                mask = studied.isin(recalled_items)
                words.loc[words["trial"] == trial, "recalled"] = mask.astype(int).values

            enc_onsets  = words["onset"].astype(float).values
            ret_onsets  = recalls["onset"].astype(float).values

            n_words    = len(words)
            n_recalled = int(words["recalled"].sum())
            n_not_rec  = n_words - n_recalled
            n_recalls_raw = len(recalls)

            raw_recall_rate     = n_recalls_raw / max(n_words, 1)
            correct_recall_rate = n_recalled    / max(n_words, 1)

            rng = np.random.default_rng(42)

            # ==================================================
            # ANALYSIS 1 — SUBSEQUENT MEMORY EFFECT (SME)
            # Recalled vs not-recalled ENCODING epochs.
            # Same task state for both classes → cleanest memory signal.
            # ==================================================

            try:
                rec_onsets     = words.loc[words["recalled"] == 1, "onset"].astype(float).values
                not_rec_onsets = words.loc[words["recalled"] == 0, "onset"].astype(float).values

                if len(rec_onsets) < MIN_EPOCHS or len(not_rec_onsets) < MIN_EPOCHS:
                    raise ValueError(
                        f"Too few SME epochs (recalled={len(rec_onsets)}, "
                        f"not_recalled={len(not_rec_onsets)})"
                    )

                ev_rec     = make_events_arr(rec_onsets,     1, sfreq)
                ev_not_rec = make_events_arr(not_rec_onsets, 0, sfreq)

                ep_rec     = make_epochs(raw, ev_rec,     ENC_TMIN, ENC_TMAX, reject_param)
                ep_not_rec = make_epochs(raw, ev_not_rec, ENC_TMIN, ENC_TMAX, reject_param)

                X_rec     = apply_baseline(ep_rec.get_data(),     sfreq)
                X_not_rec = apply_baseline(ep_not_rec.get_data(), sfreq)

                if min(len(X_rec), len(X_not_rec)) < MIN_EPOCHS:
                    raise ValueError(
                        f"Too few SME epochs after rejection "
                        f"(recalled={len(X_rec)}, not_recalled={len(X_not_rec)})"
                    )

                X_sme, y_sme, idx0, idx1, n_sme = balance_classes(X_not_rec, X_rec, rng)

                acc_sme, std_sme = run_classification(X_sme, y_sme)
                print(f"    SME:    acc={acc_sme:.3f}±{std_sme:.3f}  n={n_sme}/class")

                save_csp(X_sme, y_sme, subject, ses, "sme",
                         os.path.join(RESULTS_DIR, "csp_sme"),
                         os.path.join(RESULTS_DIR, "traj_sme"),
                         ep_rec.info)

                # Broadband bandpower per class
                ep_broad_rec     = make_epochs(raw_broad, ev_rec,     ENC_TMIN, ENC_TMAX, reject_param)
                ep_broad_not_rec = make_epochs(raw_broad, ev_not_rec, ENC_TMIN, ENC_TMAX, reject_param)
                Xb_rec     = apply_baseline(ep_broad_rec.get_data(),     sfreq)
                Xb_not_rec = apply_baseline(ep_broad_not_rec.get_data(), sfreq)

                pd.DataFrame([{
                    "subject": subject, "session": ses,
                    "n_class0": len(X_not_rec), "n_class1": len(X_rec),
                    "n_balanced": n_sme,
                    "acc": acc_sme, "acc_std": std_sme, 
                    "theta_class0": compute_bandpower(Xb_not_rec, sfreq, 4, 8),
                    "alpha_class0": compute_bandpower(Xb_not_rec, sfreq, 8, 12),
                    "theta_class1": compute_bandpower(Xb_rec,     sfreq, 4, 8),
                    "alpha_class1": compute_bandpower(Xb_rec,     sfreq, 8, 12),
                    "n_words": n_words,
                    "n_recalled": n_recalled,
                    "n_not_recalled": n_not_rec,
                    "correct_recall_rate": correct_recall_rate,
                }]).to_csv(SME_CSV, mode="a", header=False, index=False)

                del ep_rec, ep_not_rec, ep_broad_rec, ep_broad_not_rec
                del X_rec, X_not_rec, X_sme, y_sme, Xb_rec, Xb_not_rec

            except Exception as e:
                log_failure(subject, ses, "SME", str(e))

            gc.collect()


            # ==================================================
            # ANALYSIS 3 — ENCODING vs RETRIEVAL (ERS)
            # Classic encoding/retrieval flip.
            # Retrieval window: [-2, -1s] pre-vocal to avoid motor prep.
            # Overlap-check between encoding and retrieval windows.
            # Accuracy is expected to be high due to task-state differences;
            # the meaningful test is correlation with recall rate across sessions.
            # ==================================================

            try:
                keep_enc, keep_ret = reject_overlapping(
                    enc_onsets, ret_onsets,
                    (ENC_TMIN, ENC_TMAX),
                    (RET_TMIN, RET_TMAX)
                )
                enc_onsets_clean = enc_onsets[keep_enc]
                ret_onsets_clean = ret_onsets[keep_ret]

                if len(enc_onsets_clean) < MIN_EPOCHS or len(ret_onsets_clean) < MIN_EPOCHS:
                    raise ValueError(
                        f"Too few ERS epochs after overlap rejection "
                        f"(enc={len(enc_onsets_clean)}, ret={len(ret_onsets_clean)})"
                    )

                ev_enc = make_events_arr(enc_onsets_clean, 0, sfreq)
                ev_ret = make_events_arr(ret_onsets_clean, 1, sfreq)

                ep_enc = make_epochs(raw, ev_enc, ENC_TMIN, ENC_TMAX, reject_param)
                ep_ret = make_epochs(raw, ev_ret, RET_TMIN, RET_TMAX, reject_param)

                X_enc = apply_baseline(ep_enc.get_data(), sfreq)
                X_ret = apply_baseline(ep_ret.get_data(), sfreq)

                if min(len(X_enc), len(X_ret)) < MIN_EPOCHS:
                    raise ValueError(
                        f"Too few ERS epochs after rejection "
                        f"(enc={len(X_enc)}, ret={len(X_ret)})"
                    )

                X_ers, y_ers, idx0, idx1, n_ers = balance_classes(X_enc, X_ret, rng)

                acc_ers, std_ers = run_classification(X_ers, y_ers)
                print(f"    ERS:    acc={acc_ers:.3f}±{std_ers:.3f}   n={n_ers}/class")

                save_csp(X_ers, y_ers, subject, ses, "ers",
                         os.path.join(RESULTS_DIR, "csp_ers"),
                         os.path.join(RESULTS_DIR, "traj_ers"),
                         ep_enc.info)

                ep_broad_enc = make_epochs(raw_broad, ev_enc, ENC_TMIN, ENC_TMAX, reject_param)
                ep_broad_ret = make_epochs(raw_broad, ev_ret, RET_TMIN, RET_TMAX, reject_param)
                Xb_enc = apply_baseline(ep_broad_enc.get_data(), sfreq)
                Xb_ret = apply_baseline(ep_broad_ret.get_data(), sfreq)

                pd.DataFrame([{
                    "subject": subject, "session": ses,
                    "n_class0": len(X_enc), "n_class1": len(X_ret),
                    "n_balanced": n_ers,
                    "acc": acc_ers, "acc_std": std_ers, 
                    "theta_class0": compute_bandpower(Xb_enc, sfreq, 4, 8),
                    "alpha_class0": compute_bandpower(Xb_enc, sfreq, 8, 12),
                    "theta_class1": compute_bandpower(Xb_ret, sfreq, 4, 8),
                    "alpha_class1": compute_bandpower(Xb_ret, sfreq, 8, 12),
                    "n_words": n_words,
                    "n_recalls": n_recalls_raw,
                    "raw_recall_rate": raw_recall_rate,
                    "correct_recall_rate": correct_recall_rate,
                }]).to_csv(ERS_CSV, mode="a", header=False, index=False)

                del ep_enc, ep_ret, ep_broad_enc, ep_broad_ret
                del X_enc, X_ret, X_ers, y_ers, Xb_enc, Xb_ret

            except Exception as e:
                log_failure(subject, ses, "ERS", str(e))

            gc.collect()

        except Exception as e:
            log_failure(subject, ses, "LOAD", str(e))

        finally:
            for obj in [raw, raw_broad]:
                if obj is not None:
                    del obj
            gc.collect()

print("\n" + "=" * 60)
print("Done.")
print(f"  SME results    : {SME_CSV}")
print(f"  ERS results    : {ERS_CSV}")
print(f"  Failures       : {FAILED_CSV}")