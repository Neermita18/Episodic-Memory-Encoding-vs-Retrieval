import os
import pickle
import numpy as np
import pandas as pd
import mne
import matplotlib.pyplot as plt
import seaborn as sns

# ==========================================
# CONFIGURATION
# ==========================================
SUBJECT = "LTP066"     # Change to your patient
ANALYSIS = "ers"       # "ers" or "sme"
BEST_SESSION = 10       # <-- PUT YOUR HIGHEST ACCURACY SESSION NUMBER HERE

RESULTS_DIR = "results"
N_COMPONENTS = 4

csp_dir = os.path.join(RESULTS_DIR, f"csp_{ANALYSIS}")
traj_dir = os.path.join(RESULTS_DIR, f"traj_{ANALYSIS}")

# ==========================================
# 1. SINGLE "EXEMPLAR" SESSION (RED/BLUE)
# ==========================================
print(f"Plotting Exemplar Session {BEST_SESSION}...")

pkl_path = os.path.join(csp_dir, f"{SUBJECT}_ses{BEST_SESSION}_{ANALYSIS}_csp.pkl")
info_path = os.path.join(csp_dir, f"{SUBJECT}_ses{BEST_SESSION}_{ANALYSIS}_info.fif")
npz_path = os.path.join(traj_dir, f"{SUBJECT}_ses{BEST_SESSION}_{ANALYSIS}.npz")

if os.path.exists(pkl_path) and os.path.exists(info_path):
    with open(pkl_path, "rb") as f:
        csp_single = pickle.load(f)
    info_single = mne.io.read_info(info_path, verbose=False)
    
    # 1A. Exemplar Topomaps (Raw patterns, standard RdBu colormap)
    fig, axes = plt.subplots(1, N_COMPONENTS, figsize=(3 * N_COMPONENTS, 3))
    fig.suptitle(f"Exemplar {ANALYSIS.upper()} Spatial Patterns ({SUBJECT}, Session {BEST_SESSION})", fontsize=14, y=1.1)
    
    for idx, ax in enumerate(axes):
        # We do NOT use np.abs() here. MNE will automatically use Red/Blue.
        mne.viz.plot_topomap(csp_single.patterns_[idx], info_single, axes=ax, show=False)
        ax.set_title(f"Component {idx + 1}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"exemplar_topomap_{SUBJECT}_ses{BEST_SESSION}_{ANALYSIS}.png"), dpi=300)
    plt.show()

# 1B. Exemplar Trajectory
if os.path.exists(npz_path):
    data_single = np.load(npz_path)
    X_single, y_single = data_single["X_csp"], data_single["y"]
    
    df_single = pd.DataFrame({
        "Component_1": X_single[:, 0], 
        "Component_Last": X_single[:, X_single.shape[1] - 1], 
        "Class": ["Retrieval" if y == 1 else "Encoding" for y in y_single] 
                 if ANALYSIS == "ers" else 
                 ["Remembered" if y == 1 else "Forgotten" for y in y_single]
    })

    plt.figure(figsize=(7, 5))
    sns.kdeplot(data=df_single, x="Component_1", y="Component_Last", hue="Class", fill=True, alpha=0.5, palette="Set1")
    plt.title(f"Exemplar {ANALYSIS.upper()} Feature Space ({SUBJECT}, Session {BEST_SESSION})", fontsize=14, fontweight="bold")
    plt.xlabel("CSP Component 1 (Log-Variance)")
    plt.ylabel(f"CSP Component {X_single.shape[1]} (Log-Variance)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"exemplar_traj_{SUBJECT}_ses{BEST_SESSION}_{ANALYSIS}.png"), dpi=300)
    plt.show()

# ==========================================
# 2. 20-SESSION AVERAGE (ABSOLUTE / REDS)
# ==========================================
print(f"\nAggregating 20-Session Average for {SUBJECT}...")

loaded_sessions = []
for ses in range(20):
    p = os.path.join(csp_dir, f"{SUBJECT}_ses{ses}_{ANALYSIS}_csp.pkl")
    i = os.path.join(csp_dir, f"{SUBJECT}_ses{ses}_{ANALYSIS}_info.fif")
    if os.path.exists(p) and os.path.exists(i):
        with open(p, "rb") as f:
            c = pickle.load(f)
        inf = mne.io.read_info(i, verbose=False)
        loaded_sessions.append({"ses": ses, "patterns": np.abs(c.patterns_), "ch_names": inf.ch_names, "info": inf})

if loaded_sessions:
    common_chs = set(loaded_sessions[0]["ch_names"])
    for s_data in loaded_sessions[1:]:
        common_chs = common_chs.intersection(set(s_data["ch_names"]))
    common_chs = [ch for ch in loaded_sessions[0]["ch_names"] if ch in common_chs]

    aligned_patterns = []
    for s_data in loaded_sessions:
        pat = s_data["patterns"]
        idx = [s_data["ch_names"].index(ch) for ch in common_chs]
        if pat.shape[1] == len(s_data["ch_names"]):
            pat_aligned = pat[:N_COMPONENTS, idx]
        else:
            pat_aligned = pat[idx, :N_COMPONENTS].T
        aligned_patterns.append(pat_aligned)

    avg_patterns = np.mean(aligned_patterns, axis=0)
    
    info_to_pick = loaded_sessions[0]["info"]
    ch_indices = [info_to_pick.ch_names.index(ch) for ch in common_chs]
    master_info = mne.pick_info(info_to_pick, sel=ch_indices)

    # 2A. Average Topomaps (Absolute patterns, Reds colormap)
    fig, axes = plt.subplots(1, N_COMPONENTS, figsize=(3 * N_COMPONENTS, 3))
    fig.suptitle(f"20-Session Average {ANALYSIS.upper()} Spatial Patterns ({SUBJECT})", fontsize=14, y=1.1)
    
    for idx, ax in enumerate(axes):
        mne.viz.plot_topomap(avg_patterns[idx], master_info, axes=ax, show=False, cmap="Reds")
        ax.set_title(f"Component {idx + 1}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"avg_topomap_{SUBJECT}_{ANALYSIS}.png"), dpi=300)
    plt.show()

# 2B. Combined 20-Session Trajectory
X_all, y_all = [], []
for ses in range(20):
    n_path = os.path.join(traj_dir, f"{SUBJECT}_ses{ses}_{ANALYSIS}.npz")
    if os.path.exists(n_path):
        d = np.load(n_path)
        X_all.append(d["X_csp"])
        y_all.append(d["y"])

if X_all:
    X_all = np.vstack(X_all)
    y_all = np.concatenate(y_all)
    last_idx = X_all.shape[1] - 1
    df_all = pd.DataFrame({
        "Component_1": X_all[:, 0], "Component_Last": X_all[:, last_idx], 
        "Class": ["Retrieval" if y == 1 else "Encoding" for y in y_all] if ANALYSIS == "ers" else ["Remembered" if y == 1 else "Forgotten" for y in y_all]
    })

    plt.figure(figsize=(7, 5))
    sns.kdeplot(data=df_all, x="Component_1", y="Component_Last", hue="Class", fill=True, alpha=0.5, palette="Set1")
    plt.title(f"20-Session Combined {ANALYSIS.upper()} Feature Space ({SUBJECT})", fontsize=14, fontweight="bold")
    plt.xlabel("CSP Component 1 (Log-Variance)")
    plt.ylabel(f"CSP Component {last_idx + 1} (Log-Variance)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"combined_traj_{SUBJECT}_{ANALYSIS}.png"), dpi=300)
    plt.show()