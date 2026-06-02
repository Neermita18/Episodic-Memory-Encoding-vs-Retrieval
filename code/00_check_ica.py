import os
import warnings
import matplotlib.pyplot as plt

import mne
import pandas as pd

from mne.preprocessing import ICA

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



CHANNELS_TO_DROP = []
ICA_DIR = "ica_review"

os.makedirs(ICA_DIR, exist_ok=True)

# ==========================================================
# MAIN
# ==========================================================

for subject in SUBJECTS:

    print("\n" + "=" * 60)
    print(subject)
    print("=" * 60)

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
            # LOAD
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
            # ICA PREP
            # ==================================================
            raw_for_ica = raw.copy()
            raw_for_ica.resample(250, verbose=False)
            print("Fitting ICA...")
            raw_for_ica.filter(l_freq=4.0, h_freq=12.0, verbose=False)

            ica = ICA(n_components=15, random_state=42, max_iter="auto")
            ica.fit(raw_for_ica)

          

            # ==================================================
            # SAVE COMPONENT GRID
            # ==================================================

            fig = ica.plot_components(
                show=False
            )

            fig.savefig(
                os.path.join(
                    ICA_DIR,
                    f"{subject}_ses{ses}_components.png"
                ),
                dpi=300,
                bbox_inches="tight"
            )

            plt.close(fig)

            # ==================================================
            # SAVE ALL COMPONENT PROPERTIES
            # ==================================================

            comp_dir = os.path.join(
                ICA_DIR,
                f"{subject}_ses{ses}"
            )

            os.makedirs(
                comp_dir,
                exist_ok=True
            )

            for comp in range(15):

                figs = ica.plot_properties(
                    raw,
                    picks=[comp],
                    show=False
                )

                figs[0].savefig(
                    os.path.join(
                        comp_dir,
                        f"ICA{comp:03d}.png"
                    ),
                    dpi=200,
                    bbox_inches="tight"
                )

                plt.close("all")

            print("Saved ICA plots")

        except Exception as e:

            print(
                f"Session {ses} failed:"
            )

            print(e)

print("\nDone.")