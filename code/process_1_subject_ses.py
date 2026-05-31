import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from mne.decoding import CSP
from sklearn.decomposition import PCA
from mne.decoding import get_spatial_filter_from_estimator
from mne.preprocessing import ICA  # <-- NEW IMPORT

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
subject = "LTP065"
session = "0"
base_path = r"C:\Users\Neermita\Desktop\memory_and_task\ds004395"

sub_dir = f"sub-{subject}"
ses_dir = f"ses-{session}"

edf_path = os.path.join(base_path, sub_dir, ses_dir, "eeg", f"{sub_dir}_{ses_dir}_task-ltpFR_eeg.edf")
events_path = os.path.join(base_path, sub_dir, ses_dir, "eeg", f"{sub_dir}_{ses_dir}_task-ltpFR_events.tsv")
elec_path = os.path.join(base_path, sub_dir, ses_dir, "eeg", f"{sub_dir}_{ses_dir}_space-CapTrak_electrodes.tsv")

raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

# Montage Setup
electrodes = pd.read_csv(elec_path, sep='\t')
electrodes = electrodes[electrodes['x'] != 'n/a'].copy()
ch_pos = {row['name']: [float(row['x']), float(row['y']), float(row['z'])] 
          for _, row in electrodes.iterrows() if row['name'] in raw.ch_names}
raw.set_montage(mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head'), on_missing='ignore')

# Preload, Filter (Theta/Alpha), and drop noisy channel
print("\nFiltering raw data (4-12 Hz)...")
raw.load_data(verbose=False) 
raw.filter(l_freq=4.0, h_freq=12.0, verbose=False) 
raw.info['bads'] = ['E121']

# --- NEW FIX: Drop overlapping face/neck channels permanently ---
overlap = [ch for ch in ['E8', 'E25', 'E126', 'E127', 'E129'] if ch in raw.ch_names]
if overlap:
    raw.drop_channels(overlap)

# ==========================================
# 1.5 ARTIFACT REMOVAL (ICA) - NEW ADDITION
# ==========================================
print("\nRunning ICA to isolate eye blinks...")
ica = ICA(n_components=15, random_state=97, max_iter="auto")
ica.fit(raw)

# Step 1: Run this and look at the plot to find the blink (e.g., component 0 or 1)
ica.plot_components(show=True)

# Step 2: Once you know the number, uncomment the next two lines, 
# put the number in the brackets, and run the script again!
ica.exclude = [0]  # <--- Change this number to your blink component
ica.apply(raw)


# ==========================================
# 2. EVENT PARSING 
# ==========================================
print("\nParsing Events for Both Paradigms...")
events_data = pd.read_csv(events_path, sep='\t')
words_presented = events_data[events_data['trial_type'] == 'WORD'].copy()
words_recalled = events_data[events_data['trial_type'] == 'REC_WORD'].copy()

# --- PIPELINE A: Remembered vs Forgotten (SME) ---
recalled_pairs = set(zip(words_recalled['trial'], words_recalled['item_name']))
words_presented['is_remembered'] = words_presented.apply(
    lambda row: 1 if (row['trial'], row['item_name']) in recalled_pairs else 0, axis=1)

events_sme = np.zeros((len(words_presented), 3), dtype=int)
events_sme[:, 0] = (words_presented['onset'].astype(float) * raw.info['sfreq']).astype(int)
events_sme[:, 2] = words_presented['is_remembered'].to_numpy()

id_sme = {}
if 0 in np.unique(events_sme[:, 2]): id_sme['Forgotten'] = 0
if 1 in np.unique(events_sme[:, 2]): id_sme['Remembered'] = 1

# --- PIPELINE B: Silent Encoding vs Silent Retrieval (ERS) ---
words_presented['label'] = 0 # 0 = Encoding

# FIX: Shifted back to 2 seconds to avoid motor preparation artifacts!
words_recalled_shifted = words_recalled.copy()
words_recalled_shifted['onset'] = words_recalled_shifted['onset'].astype(float) - 2.0
words_recalled_shifted['label'] = 1  # 1 = Retrieval

combined_events = pd.concat([words_presented, words_recalled_shifted]).sort_values(by='onset')

events_ers = np.zeros((len(combined_events), 3), dtype=int)
events_ers[:, 0] = (combined_events['onset'].astype(float) * raw.info['sfreq']).astype(int)
events_ers[:, 2] = combined_events['label'].to_numpy()

id_ers = {'Encoding': 0, 'Retrieval': 1}

# ==========================================
# 3. EPOCH CREATION
# ==========================================
print("\nExtracting Epochs...")
epochs_sme = mne.Epochs(raw, events_sme, event_id=id_sme, tmin=-0.2, tmax=1.0, 
                        baseline=(None, 0), preload=True, reject=dict(eeg=200e-6), on_missing='ignore', verbose=False)

epochs_ers = mne.Epochs(raw, events_ers, event_id=id_ers, tmin=-0.2, tmax=1.0, 
                        baseline=(None, 0), preload=True, reject=dict(eeg=200e-6), on_missing='ignore', verbose=False)

# ==========================================
# 4. CLASSIFICATION 
# ==========================================
def train_classifier(epochs, name):
    if len(epochs.event_id) < 2 or len(epochs.get_data(copy=False)) == 0:
        print(f"\n[WARNING] Not enough data to classify {name}.")
        return
    
    X = epochs.get_data(copy=False)
    y = epochs.events[:, 2]
    clf = make_pipeline(CSP(n_components=4, reg=None, log=True, norm_trace=False), StandardScaler(), SVC(kernel='rbf'))
    scores = cross_val_score(clf, X, y, cv=5)
    print(f"{name} Accuracy: {np.mean(scores):.2f} +/- {np.std(scores):.2f}")

print("\n--- CLASSIFICATION RESULTS ---")
train_classifier(epochs_sme, "Encoding: Remembered vs Forgotten (SME)")
train_classifier(epochs_ers, "State: Silent Encoding vs Silent Retrieval (ERS)")

# ==========================================
# 5. VISUALIZATION: TRAJECTORIES
# ==========================================
print("\nGenerating Plots...")

def plot_trajectory(epochs, cond1, cond2, title, color1, color2):
    if cond1 in epochs.event_id and cond2 in epochs.event_id:
        ev1 = epochs[cond1].average().data
        ev2 = epochs[cond2].average().data
        pca = PCA(n_components=2).fit(np.hstack([ev1, ev2]).T)
        t1, t2 = pca.transform(ev1.T), pca.transform(ev2.T)

        plt.figure(figsize=(7, 5))
        plt.plot(t1[:, 0], t1[:, 1], label=cond1, color=color1, linewidth=2)
        plt.plot(t2[:, 0], t2[:, 1], label=cond2, color=color2, linewidth=2)
        plt.scatter(t1[0, 0], t1[0, 1], marker='o', color=color1, s=100)
        plt.scatter(t2[0, 0], t2[0, 1], marker='o', color=color2, s=100)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show(block=False)

plot_trajectory(epochs_sme, 'Remembered', 'Forgotten', "Trajectory: Remembered vs Forgotten", 'green', 'red')
plot_trajectory(epochs_ers, 'Encoding', 'Retrieval', "Trajectory: Silent Encoding vs Silent Retrieval", 'blue', 'orange')

# ==========================================
# 6. VISUALIZATION: TOPOMAPS
# ==========================================
def plot_topomap_grid(epochs, cond1, cond2, title):
    if cond1 in epochs.event_id and cond2 in epochs.event_id:
        ev1 = epochs[cond1].average()
        ev2 = epochs[cond2].average()
        
        # Drop face/eye electrodes to prevent map distortion
        overlap = [ch for ch in ['E8', 'E25', 'E126', 'E127', 'E129'] if ch in ev1.ch_names]
        if overlap:
            ev1.drop_channels(overlap)
            ev2.drop_channels(overlap)
            
        ev_diff = mne.combine_evoked([ev1, ev2], weights=[1, -1])
        
        times = [0.3, 0.5, 0.8] 
        fig, axes = plt.subplots(3, 4, figsize=(10, 7), gridspec_kw={'width_ratios': [1, 1, 1, 0.15]})
        axes[0, 3].axis('off')
        axes[1, 3].axis('off')
        
        fig.text(0.02, 0.80, cond1, va='center', rotation='vertical', fontsize=12, fontweight='bold')
        fig.text(0.02, 0.50, cond2, va='center', rotation='vertical', fontsize=12, fontweight='bold')
        fig.text(0.02, 0.20, f'Difference\n({cond1[:3]} - {cond2[:3]})', va='center', rotation='vertical', fontsize=12, fontweight='bold')
        
        ev1.plot_topomap(times, ch_type='eeg', axes=axes[0, :3], show=False, colorbar=False)
        ev2.plot_topomap(times, ch_type='eeg', axes=axes[1, :3], show=False, colorbar=False)
        ev_diff.plot_topomap(times, ch_type='eeg', axes=axes[2, :], show=False, colorbar=True)
        
        plt.suptitle(title, fontsize=14)
        plt.subplots_adjust(left=0.12, right=0.95, hspace=0.3)
        plt.show(block=False)

# ==========================================
# 7. VISUALIZATION: CSP PATTERNS & SCREE
# ==========================================
def plot_csp_components(epochs, title):
    if len(epochs.event_id) < 2 or len(epochs.get_data(copy=False)) == 0:
        return
    
    epochs_to_plot = epochs.copy()
    overlap = [ch for ch in ['E8', 'E25', 'E126', 'E127', 'E129'] if ch in epochs_to_plot.ch_names]
    if overlap:
        epochs_to_plot.drop_channels(overlap)
        
    X = epochs_to_plot.get_data(copy=False)
    y = epochs_to_plot.events[:, 2]
    
    csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
    csp.fit_transform(X, y)
    
    spf = get_spatial_filter_from_estimator(csp, info=epochs_to_plot.info)
    
    print(f"\nPlotting Scree for: {title}")
    spf.plot_scree()
    print(f"Plotting Patterns for: {title}")
    spf.plot_patterns(components=np.arange(4), show=True)


# ==========================================
# 8. EXECUTE ALL PLOTS
# ==========================================
# (Removed the ERS topomap grid because the timelines don't align!)
plot_topomap_grid(epochs_sme, 'Remembered', 'Forgotten', "Subsequent Memory Effect (SME)")

plot_csp_components(epochs_sme, "CSP: Remembered vs Forgotten (SME)")
plot_csp_components(epochs_ers, "CSP: Silent Encoding vs Silent Retrieval (ERS)")

plt.show()