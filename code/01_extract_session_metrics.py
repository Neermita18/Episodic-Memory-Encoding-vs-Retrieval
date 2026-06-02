import os
import pickle
import warnings

import mne
import numpy as np
import pandas as pd

from mne.preprocessing import ICA
from mne.decoding import CSP

from scipy.signal import welch

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_score
)
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis
)

warnings.filterwarnings("ignore")

# ==========================================================
# CONFIG
# ==========================================================

BASE_PATH = r"C:\Users\Neermita\Desktop\memory_and_task\ds004395"

SUBJECTS = [
    "LTP063",
    "LTP064",
    "LTP065",
    "LTP066",
    "LTP067"
]

SESSIONS = range(20)

# CHANNELS_TO_DROP = [
#     "E8",
#     "E25",
#     "E121",
#     "E126",
#     "E127",
#     "E129"
# ]

CHANNELS_TO_DROP =[]

# ==========================================================
# OUTPUT DIRECTORIES
# ==========================================================

RESULTS_DIR = "results"

CSP_DIR = os.path.join(
    RESULTS_DIR,
    "csp_models"
)

TRAJ_DIR = os.path.join(
    RESULTS_DIR,
    "trajectories"
)

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CSP_DIR, exist_ok=True)
os.makedirs(TRAJ_DIR, exist_ok=True)

# ==========================================================
# HELPERS
# ==========================================================

def compute_bandpower(
    data,
    sfreq,
    fmin,
    fmax
):

    freqs, psd = welch(
        data,
        fs=sfreq,
        axis=-1,
        nperseg=min(
            512,
            data.shape[-1]
        )
    )

    idx = (
        (freqs >= fmin)
        &
        (freqs <= fmax)
    )

    return np.mean(
        psd[..., idx]
    )


# ==========================================================
# MAIN
# ==========================================================

all_rows = []

for subject in SUBJECTS:

    print("\n" + "="*60)
    print(subject)
    print("="*60)

   

    for ses in SESSIONS:

        try:

            sub_dir = f"sub-{subject}"
            ses_dir = f"ses-{ses}"

            eeg_file = os.path.join(
                BASE_PATH,
                sub_dir,
                ses_dir,
                "eeg",
                f"{sub_dir}_{ses_dir}_task-ltpFR_eeg.edf"
            )

            event_file = os.path.join(
                BASE_PATH,
                sub_dir,
                ses_dir,
                "eeg",
                f"{sub_dir}_{ses_dir}_task-ltpFR_events.tsv"
            )

            electrode_file = os.path.join(
                BASE_PATH,
                sub_dir,
                ses_dir,
                "eeg",
                f"{sub_dir}_{ses_dir}_space-CapTrak_electrodes.tsv"
            )

            if not os.path.exists(eeg_file):
                continue

            print(f"\nSession {ses}")

            # ==================================================
            # LOAD RAW
            # ==================================================

            raw = mne.io.read_raw_edf(
                eeg_file,
                preload=True,
                verbose=False
            )

            # ==================================================
            # MONTAGE
            # ==================================================

            electrodes = pd.read_csv(electrode_file, sep="\t")

            bad_coords = electrodes[
                electrodes["x"].isna() |
                (electrodes["x"] == "n/a")
            ]["name"].tolist()

            print("Channels with no position:", bad_coords)

            to_drop = [ch for ch in bad_coords if ch in raw.ch_names]

            if to_drop:
                raw.drop_channels(to_drop)

            electrodes = electrodes[
                electrodes["x"].notna() &
                (electrodes["x"] != "n/a")
            ]

            ch_pos = {
                row["name"]: [
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"])
                ]
                for _, row in electrodes.iterrows()
                if row["name"] in raw.ch_names
            }

            montage = mne.channels.make_dig_montage(
                ch_pos=ch_pos,
                coord_frame="head"
            )

            raw.set_montage(
                montage,
                on_missing="ignore"
            ) 

            # ==================================================
            # FILTER
            # ==================================================

            raw.filter(
                l_freq=4,
                h_freq=12,
                verbose=False
            )
            # ==================================================
            # EVENTS
            # ==================================================

            events_df = pd.read_csv(
                event_file,
                sep="\t"
            )
   
            words = events_df[
                events_df.trial_type == "WORD"
            ].copy()

            recalls = events_df[
                events_df.trial_type == "REC_WORD"
            ].copy()


            # ==================================================
            # ENCODING
            # ==================================================

            encoding = words.copy()

            encoding["label"] = 0

            # ==================================================
            # RETRIEVAL
            # ==================================================

            retrieval = recalls.copy()

            retrieval["onset"] = (
                retrieval["onset"]
                .astype(float)
                - 1.0
            )

            retrieval["label"] = 1

            combined = pd.concat(
                [
                    encoding,
                    retrieval
                ],
                ignore_index=True
            )

            # ==================================================
            # EVENTS ARRAY
            # ==================================================

            events = np.zeros(
                (
                    len(combined),
                    3
                ),
                dtype=int
            )

            events[:, 0] = (
                combined["onset"]
                .astype(float)
                *
                raw.info["sfreq"]
            ).astype(int)

            events[:, 2] = (
                combined["label"]
                .astype(int)
            )

            # ==================================================
            # EPOCHS
            # ==================================================
            print("events before epoching:", len(events))
            epochs = mne.Epochs(
                raw,
                events,
                event_id={
                    "Encoding": 0,
                    "Retrieval": 1
                },
                tmin=0.0,
                tmax=1.0,
                baseline=None,
                preload=True,
                reject=dict(
                    eeg=500e-6
                ),
                verbose=False
            )


            X = epochs.get_data()
            print("after get_data")

            y = epochs.events[:, 2]
            print("after y")

            print(np.unique(y))

            if len(np.unique(y)) < 2:
                print("ONLY ONE CLASS")
                continue

            print("before cov")

            cov = np.cov(X.reshape(-1, X.shape[1]).T)

            print("after cov")

            print(np.linalg.matrix_rank(cov))
            print(cov.shape)

            # ==================================================
            # CSP + LDA
            # ==================================================

            clf = Pipeline([
                (
                    "csp",
                    CSP(
                        n_components=4,
                        log=True,
                        norm_trace=False,
                        reg='ledoit_wolf'
                    )
                ),
                (
                    "lda",
                    LinearDiscriminantAnalysis()
                )
            ])

            cv = StratifiedKFold(
                n_splits=5,
                shuffle=True,
                random_state=42
            )

            scores = cross_val_score(
                clf,
                X,
                y,
                cv=cv,
                scoring="accuracy"
            )

            ers_acc = np.mean(scores)

            # ==================================================
            # FIT CSP FOR SAVING
            # ==================================================

            csp_model = CSP(
                n_components=4,
                log=True,
                norm_trace=False,
                reg='ledoit_wolf'
            )

            csp_model.fit(
                X,
                y
            )

            # save CSP

            csp_file = os.path.join(
                CSP_DIR,
                f"{subject}_ses{ses}_csp.pkl"
            )

            with open(
                csp_file,
                "wb"
            ) as f:
                pickle.dump(
                    csp_model,
                    f
                )

            # save info

            info_file = os.path.join(
                CSP_DIR,
                f"{subject}_ses{ses}_info.fif"
            )

            epochs.info.save(
                info_file,
                overwrite=True
            )

            # ==================================================
            # SAVE CSP FEATURES
            # ==================================================

            X_csp = csp_model.transform(
                X
            )

            np.savez(
                os.path.join(
                    TRAJ_DIR,
                    f"{subject}_ses{ses}.npz"
                ),
                X_csp=X_csp,
                y=y
            )

            # ==================================================
            # BEHAVIOR
            # ==================================================

            recall_rate = (
                len(recalls)
                /
                len(words)
            )

            # ==================================================
            # THETA / ALPHA
            # ==================================================

            theta_power = compute_bandpower(
                X,
                raw.info["sfreq"],
                4,
                8
            )

            alpha_power = compute_bandpower(
                X,
                raw.info["sfreq"],
                8,
                12
            )

            # ==================================================
            # SAVE ROW
            # ==================================================

            all_rows.append([
                subject,
                ses,
                ers_acc,
                recall_rate,
                theta_power,
                alpha_power,
                len(words),
                len(recalls)
            ])

            print(
                f"ERS={ers_acc:.3f} "
                f"Recall={recall_rate:.3f}"
            )

        except Exception as e:

            print(
                f"Session {ses} failed:"
            )

            print(e)

# ==========================================================
# SAVE CSV
# ==========================================================

df = pd.DataFrame(
    all_rows,
    columns=[
        "subject",
        "session",
        "ers_acc",
        "recall_rate",
        "theta",
        "alpha",
        "n_words",
        "n_recalls"
    ]
)

csv_file = os.path.join(
    RESULTS_DIR,
    "session_metrics.csv"
)

df.to_csv(
    csv_file,
    index=False
)

print("\nSaved:")
print(csv_file)

print(df.head())