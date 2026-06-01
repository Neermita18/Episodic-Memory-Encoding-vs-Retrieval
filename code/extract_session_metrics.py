import os
import warnings
import mne
import numpy as np
import pandas as pd
from mne.preprocessing import ICA
from mne.decoding import CSP
from scipy.signal import welch
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG
# ==========================================================

BASE_PATH = r"C:\Users\Neermita\Desktop\memory_and_task\ds004395"

SUBJECTS = [
    "LTP063", "LTP064", "LTP065", "LTP066", "LTP067"
]

SESSIONS = range(20)

CHANNELS_TO_DROP = ["E8", "E25", "E121", "E126", "E127", "E129"]

# ==========================================================
# HELPERS
# ==========================================================

def compute_bandpower(data, sfreq, fmin, fmax):
    freqs, psd = welch(data, fs=sfreq, axis=-1, nperseg=min(512, data.shape[-1]))
    idx = (freqs >= fmin) & (freqs <= fmax)
    return np.mean(psd[..., idx])

# ==========================================================
# MAIN
# ==========================================================

all_rows = []

for subject in SUBJECTS:
    print("\n" + "="*50)
    print(subject)
    print("="*50)
    
    ica_model = None
    
    for ses in SESSIONS:
        try:
            sub_dir = f"sub-{subject}"
            ses_dir = f"ses-{ses}"
            
            eeg_file = os.path.join(BASE_PATH, sub_dir, ses_dir, "eeg",
                                    f"{sub_dir}_{ses_dir}_task-ltpFR_eeg.edf")
            event_file = os.path.join(BASE_PATH, sub_dir, ses_dir, "eeg",
                                      f"{sub_dir}_{ses_dir}_task-ltpFR_events.tsv")
            electrode_file = os.path.join(BASE_PATH, sub_dir, ses_dir, "eeg",
                                          f"{sub_dir}_{ses_dir}_space-CapTrak_electrodes.tsv")
            
            if not os.path.exists(eeg_file):
                continue
            
            print(f"Session {ses}")
            
            # ==================================================
            # LOAD RAW
            # ==================================================
            raw = mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)
            
            # ==================================================
            # MONTAGE
            # ==================================================
            electrodes = pd.read_csv(electrode_file, sep="\t")
            electrodes = electrodes[electrodes["x"] != "n/a"]
            ch_pos = {row["name"]: [float(row["x"]), float(row["y"]), float(row["z"])]
                      for _, row in electrodes.iterrows() if row["name"] in raw.ch_names}
            montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
            raw.set_montage(montage, on_missing="ignore")
            
            # ==================================================
            # REMOVE FACE CHANNELS
            # ==================================================
            overlap = [ch for ch in CHANNELS_TO_DROP if ch in raw.ch_names]
            if len(overlap) > 0:
                raw.drop_channels(overlap)
            
            # ==================================================
            # FILTER (for analysis - 4-12 Hz theta/alpha)
            # ==================================================
            raw.filter(l_freq=4, h_freq=12, verbose=False)
            
            # ==================================================
            # ICA FOR EYE ARTIFACT REMOVAL 
            # ==================================================
            if ses == 0:
                print("  Fitting ICA on session 0...")
                
                # Create a copy with 1 Hz high-pass for ICA (better separation)
                raw_ica = raw.copy()
                raw_ica.filter(l_freq=1.0, h_freq=None, verbose=False)
                
                # Fit ICA
                ica_model = ICA(n_components=15, random_state=42, max_iter=500)
                ica_model.fit(raw_ica)
                
                # AUTOMATIC EOG DETECTION
                eog_indices, eog_scores = ica_model.find_bads_eog(
                    raw_ica, 
                    ch_name=None,  # Auto-detect
                    threshold=3.0
                )
                
                if len(eog_indices) > 0:
                    ica_model.exclude = eog_indices
                    print(f"    Removing {len(eog_indices)} EOG components: {eog_indices}")
                else:
                    print(f"     No EOG detected automatically.")
                    print(f"    Running manual check...")
                    # Plot for manual inspection (will pop up window)
                    ica_model.plot_components(show=True)
                    print("    Look for frontal components (likely components 0 or 1)")
                    print("    After inspecting, add to exclude list in code")
                    # For LTP dataset, component 0 is often the eye blink
                    # Uncomment after verification:
                    # ica_model.exclude = [0]
            
            # Apply ICA if we have components to remove
            if ica_model is not None and hasattr(ica_model, 'exclude') and len(ica_model.exclude) > 0:
                print(f"    Applying ICA: removing {ica_model.exclude}")
                ica_model.apply(raw)
            else:
                print(f"    No ICA removal for session {ses}")
            
            # ==================================================
            # EVENTS
            # ==================================================
            events_df = pd.read_csv(event_file, sep="\t")
            words = events_df[events_df.trial_type == "WORD"].copy()
            recalls = events_df[events_df.trial_type == "REC_WORD"].copy()
            
            if len(words) < 10 or len(recalls) < 5:
                continue
            
            # ==================================================
            # ENCODING: WORD onset (0 to 1 sec)
            # ==================================================
            encoding = words.copy()
            encoding["label"] = 0
            
            # ==================================================
            # RETRIEVAL: recall onset - 2 sec (0 to 1 sec window)
            # ==================================================
            retrieval = recalls.copy()
            retrieval["onset"] = retrieval["onset"].astype(float) - 2.0
            retrieval["label"] = 1
            
            combined = pd.concat([encoding, retrieval], ignore_index=True)
            
            # ==================================================
            # EVENT MATRIX
            # ==================================================
            events = np.zeros((len(combined), 3), dtype=int)
            events[:, 0] = (combined["onset"].astype(float) * raw.info["sfreq"]).astype(int)
            events[:, 2] = combined["label"].astype(int)
            
            # ==================================================
            # EPOCHS: 0 to 1 second window
            # ==================================================
            epochs = mne.Epochs(
                raw, events,
                event_id={"Encoding": 0, "Retrieval": 1},
                tmin=0.0, tmax=1.0,
                baseline=None,
                preload=True,
                reject=dict(eeg=200e-6),
                verbose=False
            )
            
            X = epochs.get_data()
            y = epochs.events[:, 2]
            
            if len(np.unique(y)) < 2:
                continue
            
            # ==================================================
            # CSP + LDA DECODING
            # ==================================================
            clf = Pipeline([
                ("csp", CSP(n_components=4, log=True, norm_trace=False)),
                ("lda", LinearDiscriminantAnalysis())
            ])
            
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
            ers_acc = np.mean(scores)
            
            # ==================================================
            # BEHAVIORAL METRICS
            # ==================================================
            recall_rate = len(recalls) / len(words)
            
            # ==================================================
            # SPECTRAL POWER
            # ==================================================
            theta_power = compute_bandpower(X, raw.info["sfreq"], 4, 8)
            alpha_power = compute_bandpower(X, raw.info["sfreq"], 8, 12)
            
            # ==================================================
            # SAVE
            # ==================================================
            all_rows.append([
                subject, ses, ers_acc, recall_rate,
                theta_power, alpha_power,
                len(words), len(recalls)
            ])
            
            print(f"  ERS={ers_acc:.3f}, Recall={recall_rate:.3f}")
            
        except Exception as e:
            print(f"  Session {ses} failed: {e}")

# ==========================================================
# SAVE RESULTS
# ==========================================================
df = pd.DataFrame(all_rows, columns=[
    "subject", "session", "ers_acc", "recall_rate",
    "theta", "alpha", "n_words", "n_recalls"
])

df.to_csv("session_metrics.csv", index=False)
print("\n" + "="*50)
print("SAVED: session_metrics.csv")
print("="*50)
print(df.head())