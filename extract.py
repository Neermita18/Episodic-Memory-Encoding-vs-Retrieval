import mne
import matplotlib.pyplot as plt

raw = mne.io.read_raw_edf(
    r"C:\Users\Neermita\Desktop\memory_and_task\ds004395\sub-LTP063\ses-0\eeg\sub-LTP063_ses-0_task-ltpFR_eeg.edf",
    preload=True
)

print(raw)


# Open interactive EEG viewer
raw.plot(
    n_channels=20,
    duration=10,
    scalings='auto',
    block=True   # IMPORTANT
)

plt.show()