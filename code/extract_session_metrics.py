
"""
Extracts ERS decoding accuracy, recall rate, and spectral power for each session.
"""

import os
import mne
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP

# ==========================================
# CONFIGURATION
# ==========================================
BASE_PATH = r"C:\Users\Neermita\Desktop\memory_and_task\ds004395"

SUBJECTS = ['LTP063', 'LTP064', 'LTP065', 'LTP066', 'LTP067']

SESSIONS = range(20)  # 0 through 19

TMIN = -0.2
TMAX = 1.0

# Create output directories
os.makedirs("csp_models", exist_ok=True)
os.makedirs("../results", exist_ok=True)

# ==========================================
# MAIN PROCESSING LOOP
# ==========================================
all_rows = []

for subject in SUBJECTS:
    print(f"\nProcessing {subject}...")
    
    for ses in SESSIONS:
        try:
            # File paths
            eeg_file = os.path.join(
                BASE_PATH, f"sub-{subject}", f"ses-{ses}", "eeg",
                f"sub-{subject}_ses-{ses}_task-ltpFR_eeg.edf"
            )
            event_file = os.path.join(
                BASE_PATH, f"sub-{subject}", f"ses-{ses}", "eeg",
                f"sub-{subject}_ses-{ses}_task-ltpFR_events.tsv"
            )
            
            if not os.path.exists(eeg_file):
                continue
            
            # Load EEG
            raw = mne.io.read_raw_edf(eeg_file, preload=True, verbose=False)
            raw.filter(1, 40, verbose=False)  # Broadband filter for spectral analysis
            
            # Load events
            events_df = pd.read_csv(event_file, sep="\t")
            
            words = events_df[events_df.trial_type == "WORD"].copy()
            recalls = events_df[events_df.trial_type == "REC_WORD"].copy()
            
            if len(words) == 0:
                continue
            
            # Compute recall rate for this session
            recall_pairs = set(zip(recalls.trial, recalls.item_name))
            words["remembered"] = words.apply(
                lambda r: int((r.trial, r.item_name) in recall_pairs), axis=1
            )
            recall_rate = words.remembered.mean()
            
            # Prepare ERS (Encoding vs Retrieval) epochs
            words["label"] = 0  # Encoding
            
            recalls_shift = recalls.copy()
            recalls_shift["onset"] -= 2.0  # Shift back to capture silent retrieval
            recalls_shift["label"] = 1  # Retrieval
            
            ers_df = pd.concat([words, recalls_shift]).sort_values("onset")
            
            # Create events array for MNE
            events = np.zeros((len(ers_df), 3), dtype=int)
            events[:, 0] = (ers_df.onset.values * raw.info["sfreq"]).astype(int)
            events[:, 2] = ers_df.label.values
            
            # Create epochs
            epochs = mne.Epochs(
                raw, events,
                tmin=TMIN, tmax=TMAX,
                baseline=(None, 0),
                preload=True, verbose=False
            )
            
            X = epochs.get_data()
            y = epochs.events[:, 2]
            
            if len(np.unique(y)) < 2:
                continue
            
            # CSP + LDA decoding pipeline (better than SVM for EEG)
            csp = CSP(n_components=6, log=True)
            lda = LinearDiscriminantAnalysis()
            
            pipe = Pipeline([
                ("csp", csp),
                ("scaler", StandardScaler()),
                ("lda", lda)
            ])
            
            cv = StratifiedKFold(5, shuffle=True, random_state=42)
            scores = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy")
            ers_acc = scores.mean()
            
            # Save CSP filters for topomaps later
            csp.fit(X, y)
            np.save(f"csp_models/{subject}_{ses}.npy", csp.filters_)
            
            # Compute spectral power (theta and alpha bands)
            sfreq = epochs.info["sfreq"]
            theta_power = []
            alpha_power = []
            
            for ep in X:
                freqs, psd = welch(ep, sfreq, nperseg=256)
                
                theta_idx = (freqs >= 4) & (freqs <= 8)
                alpha_idx = (freqs >= 8) & (freqs <= 12)
                
                theta_power.append(np.mean(psd[:, theta_idx]))
                alpha_power.append(np.mean(psd[:, alpha_idx]))
            
            all_rows.append({
                "subject": subject,
                "session": ses,
                "ers_acc": ers_acc,
                "recall_rate": recall_rate,
                "theta_power": np.mean(theta_power),
                "alpha_power": np.mean(alpha_power)
            })
            
            print(f"  Session {ses}: ERS={ers_acc:.3f}, Recall={recall_rate:.3f}")
            
        except Exception as e:
            print(f"  Session {ses}: ERROR - {e}")

# ==========================================
# SAVE RESULTS
# ==========================================
df = pd.DataFrame(all_rows)
df.to_csv("../results/session_metrics.csv", index=False)

print("\n" + "="*60)
print("COMPLETE - Saved to ../results/session_metrics.csv")
print("="*60)
print(f"Total sessions processed: {len(df)}")
print(f"Subjects: {df.subject.nunique()}")
print(f"\nFirst 5 rows:")
print(df.head())