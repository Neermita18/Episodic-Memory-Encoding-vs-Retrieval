# run_multi_subject.py
import os
import mne
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from mne.decoding import CSP
from scipy.stats import permutation_test_score
import warnings
warnings.filterwarnings('ignore')

from config import BASE_PATH, SUBJECTS, FILTER_LOW, FILTER_HIGH, TMIN, TMAX, REJECT_CRITERION

def process_one_subject(subject, session="0"):
    """Process a single subject and return decoding results"""
    
    print(f"\n{'='*50}")
    print(f"Processing: {subject}")
    print(f"{'='*50}")
    
    # Build file paths
    sub_dir = f"sub-{subject}"
    ses_dir = f"ses-{session}"
    
    edf_path = os.path.join(BASE_PATH, sub_dir, ses_dir, "eeg", 
                            f"{sub_dir}_{ses_dir}_task-ltpFR_eeg.edf")
    events_path = os.path.join(BASE_PATH, sub_dir, ses_dir, "eeg", 
                               f"{sub_dir}_{ses_dir}_task-ltpFR_events.tsv")
    elec_path = os.path.join(BASE_PATH, sub_dir, ses_dir, "eeg", 
                             f"{sub_dir}_{ses_dir}_space-CapTrak_electrodes.tsv")
    
    # Check files exist
    if not all(os.path.exists(p) for p in [edf_path, events_path, elec_path]):
        print(f"Missing files for {subject}, skipping...")
        return None
    
    # Load raw data
    print("Loading EEG...")
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)
    
    # Load and set montage
    electrodes = pd.read_csv(elec_path, sep='\t')
    electrodes = electrodes[electrodes['x'] != 'n/a'].copy()
    ch_pos = {row['name']: [float(row['x']), float(row['y']), float(row['z'])] 
              for _, row in electrodes.iterrows() if row['name'] in raw.ch_names}
    raw.set_montage(mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head'), 
                    on_missing='ignore')
    
    # Load data and filter
    print(f"Filtering {FILTER_LOW}-{FILTER_HIGH} Hz...")
    raw.load_data(verbose=False)
    raw.filter(l_freq=FILTER_LOW, h_freq=FILTER_HIGH, verbose=False)
    
    # Mark noisy channel if exists
    if 'E121' in raw.ch_names:
        raw.info['bads'] = ['E121']
    
    # Load events
    events_data = pd.read_csv(events_path, sep='\t')
    words_presented = events_data[events_data['trial_type'] == 'WORD'].copy()
    words_recalled = events_data[events_data['trial_type'] == 'REC_WORD'].copy()
    
    if len(words_presented) == 0 or len(words_recalled) == 0:
        print(f"Insufficient events for {subject}, skipping...")
        return None
    
    # ==========================================
    # CONDITION 1: SME (Remembered vs Forgotten)
    # ==========================================
    recalled_pairs = set(zip(words_recalled['trial'], words_recalled['item_name']))
    words_presented['is_remembered'] = words_presented.apply(
        lambda row: 1 if (row['trial'], row['item_name']) in recalled_pairs else 0, axis=1)
    
    events_sme = np.zeros((len(words_presented), 3), dtype=int)
    events_sme[:, 0] = (words_presented['onset'].astype(float) * raw.info['sfreq']).astype(int)
    events_sme[:, 2] = words_presented['is_remembered'].to_numpy()
    
    # ==========================================
    # CONDITION 2: ERS (Encoding vs Retrieval)
    # ==========================================
    words_presented['label'] = 0
    words_recalled_shifted = words_recalled.copy()
    words_recalled_shifted['onset'] = words_recalled_shifted['onset'].astype(float) - 1.0
    words_recalled_shifted['label'] = 1
    combined_events = pd.concat([words_presented, words_recalled_shifted]).sort_values(by='onset')
    
    events_ers = np.zeros((len(combined_events), 3), dtype=int)
    events_ers[:, 0] = (combined_events['onset'].astype(float) * raw.info['sfreq']).astype(int)
    events_ers[:, 2] = combined_events['label'].to_numpy()
    
    # Create epochs
    epochs_sme = mne.Epochs(raw, events_sme, event_id={'Forgotten':0, 'Remembered':1}, 
                            tmin=TMIN, tmax=TMAX, baseline=(None, 0), preload=True, 
                            reject=dict(eeg=REJECT_CRITERION), on_missing='ignore', verbose=False)
    
    epochs_ers = mne.Epochs(raw, events_ers, event_id={'Encoding':0, 'Retrieval':1}, 
                            tmin=TMIN, tmax=TMAX, baseline=(None, 0), preload=True, 
                            reject=dict(eeg=REJECT_CRITERION), on_missing='ignore', verbose=False)
    
    def get_decoding_scores(epochs, name):
        if len(epochs) < 10 or len(np.unique(epochs.events[:, 2])) < 2:
            print(f"  {name}: Insufficient trials")
            return None
        
        X = epochs.get_data()
        y = epochs.events[:, 2]
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        clf = make_pipeline(CSP(n_components=4, log=True), 
                           StandardScaler(), 
                           SVC(kernel='rbf'))
        
        # Cross-validation score
        scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
        
        # Permutation test (100 permutations for speed, increase later)
        score_perm, perm_scores, p_val = permutation_test_score(
            clf, X, y, cv=cv, n_permutations=100, 
            scoring='accuracy', n_jobs=-1, random_state=42
        )
        
        return {
            'mean_acc': np.mean(scores),
            'std_acc': np.std(scores),
            'p_value': p_val,
            'n_trials': len(X)
        }
    
    sme_results = get_decoding_scores(epochs_sme, "SME")
    ers_results = get_decoding_scores(epochs_ers, "ERS")
    
    # Compute recall rate
    recall_rate = len(words_recalled) / len(words_presented) if len(words_presented) > 0 else 0
    
    return {
        'subject': subject,
        'sme_mean': sme_results['mean_acc'] if sme_results else None,
        'sme_std': sme_results['std_acc'] if sme_results else None,
        'sme_p': sme_results['p_value'] if sme_results else None,
        'sme_n': sme_results['n_trials'] if sme_results else None,
        'ers_mean': ers_results['mean_acc'] if ers_results else None,
        'ers_std': ers_results['std_acc'] if ers_results else None,
        'ers_p': ers_results['p_value'] if ers_results else None,
        'ers_n': ers_results['n_trials'] if ers_results else None,
        'recall_rate': recall_rate
    }

# ==========================================
# MAIN: Process all subjects
# ==========================================
if __name__ == "__main__":
    print(f"Found {len(SUBJECTS)} subjects to process")
    print(f"Subjects: {SUBJECTS}")
    
    all_results = []
    for subj in SUBJECTS:
        result = process_one_subject(subj)
        if result:
            all_results.append(result)
    
    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv('memory_decoding_results.csv', index=False)
    
    print("\n" + "="*50)
    print("COMPLETE RESULTS")
    print("="*50)
    print(df[['subject', 'sme_mean', 'ers_mean', 'recall_rate']].to_string())
    
    # Summary stats
    sme_vals = df['sme_mean'].dropna()
    ers_vals = df['ers_mean'].dropna()
    
    print(f"\nSME: mean={sme_vals.mean():.3f} ± {sme_vals.std():.3f}, n={len(sme_vals)}")
    print(f"ERS: mean={ers_vals.mean():.3f} ± {ers_vals.std():.3f}, n={len(ers_vals)}")