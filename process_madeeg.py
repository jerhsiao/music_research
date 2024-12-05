# This script processes EEG data from the MAD-EEG dataset, extracts features,
# integrates behavioral and musical features, and saves results to a CSV file.

import os
import sys
import logging
import yaml
import h5py
import pandas as pd
import numpy as np
import mne
import ast
import re
from tqdm import tqdm

# Expected columns from user's specification
EXPECTED_COLUMNS = [
    'F1_Delta','F2_Delta','F3_Delta','F4_Delta',
    'F1_Theta','F2_Theta','F3_Theta','F4_Theta',
    'F1_Alpha','F2_Alpha','F3_Alpha','F4_Alpha',
    'F1_Beta','F2_Beta','F3_Beta','F4_Beta',
    'Frontal_Alpha_Asymmetry',
    'F1_Activity','F1_Mobility','F1_Complexity',
    'F2_Activity','F2_Mobility','F2_Complexity',
    'F3_Activity','F3_Mobility','F3_Complexity',
    'F4_Activity','F4_Mobility','F4_Complexity',
    'Subject','Stimulus','Genre','Song','Ensemble','Instruments','Theme','Spatial','Target_Instrument',
    'Tempo','Key','Spectral_Centroid','Spectral_Bandwidth','Spectral_Contrast','Spectral_Rolloff',
    'MFCC_1_Mean','MFCC_1_Std','MFCC_2_Mean','MFCC_2_Std','MFCC_3_Mean','MFCC_3_Std','MFCC_4_Mean','MFCC_4_Std','MFCC_5_Mean','MFCC_5_Std','MFCC_6_Mean','MFCC_6_Std',
    'MFCC_7_Mean','MFCC_7_Std','MFCC_8_Mean','MFCC_8_Std','MFCC_9_Mean','MFCC_9_Std','MFCC_10_Mean','MFCC_10_Std','MFCC_11_Mean','MFCC_11_Std','MFCC_12_Mean','MFCC_12_Std','MFCC_13_Mean','MFCC_13_Std',
    'Chroma_0','Chroma_1','Chroma_2','Chroma_3','Chroma_4','Chroma_5','Chroma_6','Chroma_7','Chroma_8','Chroma_9','Chroma_10','Chroma_11',
    'Attention_Label'
]

from mne import time_frequency

HDF5_PATH = 'datasets/madeeg_preprocessed.hdf5'
YAML_PATH = 'datasets/madeeg_preprocessed.yaml'
BEHAVIORAL_DATA_PATH = 'datasets/behavioural_data.xlsx'
OUTPUT_CSV = 'outputs/csv/madeeg_features_dataset.csv'
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

LOG_DIR = "outputs/logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "process_madeeg.log")),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

def normalize_stimulus_name(stim_name):
    name = re.sub(r'_(?:lcr|mono)', '', stim_name)
    name = re.sub(r'_$', '', name)
    return name

def compute_frontal_alpha_asymmetry(epoch_features):
    f1_alpha = epoch_features.get('F1_Alpha', np.nan)
    f2_alpha = epoch_features.get('F2_Alpha', np.nan)
    if not np.isnan(f1_alpha) and not np.isnan(f2_alpha):
        return f1_alpha - f2_alpha
    return np.nan

def compute_hjorth_parameters(epoch_data, selected_channels):
    hjorth = {}
    for idx, ch in enumerate(selected_channels):
        ch_data = epoch_data[idx]
        if np.std(ch_data) == 0:
            hjorth[f"{ch}_Activity"] = 0
            hjorth[f"{ch}_Mobility"] = 0
            hjorth[f"{ch}_Complexity"] = 0
        else:
            activity = np.var(ch_data)
            diff1 = np.diff(ch_data)
            mobility = np.std(diff1)/np.std(ch_data) if np.std(ch_data)!=0 else 0
            diff2 = np.diff(diff1)
            complexity = (np.std(diff2)/np.std(diff1))/mobility if mobility!=0 and np.std(diff1)!=0 else 0
            hjorth[f"{ch}_Activity"] = activity
            hjorth[f"{ch}_Mobility"] = mobility
            hjorth[f"{ch}_Complexity"] = complexity
    return hjorth

def preprocess_behavioral_data(stimuli_order_input, concentration_levels_input):
    def parse_list(input_data):
        if isinstance(input_data, str):
            try:
                parsed = ast.literal_eval(input_data)
                if isinstance(parsed, list):
                    parsed = [item for sublist in parsed for item in (sublist if isinstance(sublist, list) else [sublist])]
                return parsed
            except:
                logger.error("Error parsing list.")
                return None
        elif isinstance(input_data, list):
            if all(isinstance(item, list) for item in input_data):
                return [stim for sublist in input_data for stim in sublist]
            else:
                return input_data
        else:
            logger.error(f"Unexpected type: {type(input_data)}")
            return None

    stimuli_order_list = parse_list(stimuli_order_input)
    if stimuli_order_list is None:
        logger.error("Failed to parse stimuli_order_input.")
        return None, None

    concentration_levels_list = parse_list(concentration_levels_input)
    if concentration_levels_list is None:
        logger.error("Failed to parse concentration_levels_input.")
        return None, None

    try:
        concentration_levels_list = [float(score) for score in concentration_levels_list]
    except:
        logger.error("Error converting concentration levels to float.")
        return None, None

    if len(stimuli_order_list) != len(concentration_levels_list):
        logger.error("Length mismatch between stimuli and concentration levels.")
        return None, None

    return stimuli_order_list, concentration_levels_list

def extract_madeeg_features(subject_id, stimulus_name, data, metadata, selected_channels, freq_bands, stimuli_features_dict, behavioral_dict):
    logger.info(f"  Extracting features for {subject_id}, {stimulus_name}")
    norm_name = normalize_stimulus_name(stimulus_name)
    eeg_response = data[subject_id][norm_name]['response'][:]
    eeg_info = metadata[subject_id][norm_name].get('eeg_info', {})
    sfreq = eeg_info.get('sfreq', None)
    ch_names = eeg_info.get('ch_names', [])

    if sfreq is None or not ch_names:
        logger.warning(f"    Missing EEG info for {subject_id}, {norm_name}.")
        return None

    selected_indices = [ch_names.index(ch) for ch in selected_channels if ch in ch_names]
    available_channels = [ch_names[i] for i in selected_indices]
    eeg_response = eeg_response[selected_indices, :]

    if len(available_channels) == 0:
        logger.warning(f"    No selected channels available for {subject_id}, {norm_name}.")
        return None

    if np.all(eeg_response == eeg_response[0, :]):
        logger.warning(f"    Zero variance EEG for {subject_id}, {norm_name}.")
        return None

    info = mne.create_info(ch_names=available_channels, ch_types=['eeg']*len(available_channels), sfreq=sfreq)
    raw = mne.io.RawArray(eeg_response, info, verbose=False)
    raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
    raw.notch_filter(freqs=50, verbose=False)

    psds, freqs = raw.compute_psd(method='welch', fmin=1.0, fmax=40.0, n_fft=256, verbose=False).get_data(return_freqs=True)

    # Initialize all expected columns with NaN
    features = {col: np.nan for col in EXPECTED_COLUMNS}

    # Fill in EEG band power features
    for band_name, (fmin, fmax) in freq_bands.items():
        freq_mask = (freqs>=fmin)&(freqs<=fmax)
        if np.any(freq_mask):
            band_power = psds[:, freq_mask].mean(axis=1)
            for i, ch_name in enumerate(available_channels):
                features[f"{ch_name}_{band_name}"] = band_power[i]

    # Frontal Alpha Asymmetry
    features['Frontal_Alpha_Asymmetry'] = compute_frontal_alpha_asymmetry(features)

    # Hjorth parameters
    raw_data = raw.get_data()
    hjorth_features = compute_hjorth_parameters(raw_data, available_channels)
    for k,v in hjorth_features.items():
        features[k] = v

    # Subject/Stimulus info
    features['Subject'] = subject_id
    features['Stimulus'] = norm_name

    # Metadata from YAML
    stim_metadata = metadata[subject_id][norm_name]
    features['Genre'] = stim_metadata.get('genre', 'Unknown')
    features['Song'] = stim_metadata.get('song', 'Unknown')
    features['Ensemble'] = stim_metadata.get('ensemble', 'Unknown')
    features['Instruments'] = stim_metadata.get('instruments', 'Unknown')
    features['Theme'] = stim_metadata.get('theme', 'Unknown')
    features['Spatial'] = stim_metadata.get('spatial', 'Unknown')
    features['Target_Instrument'] = stim_metadata.get('target', 'Unknown')

    # Stimuli features
    if norm_name in stimuli_features_dict:
        stimuli_feat = stimuli_features_dict[norm_name]
        features['Tempo'] = stimuli_feat.get('tempo', np.nan)
        features['Key'] = stimuli_feat.get('key', 'Unknown')
        features['Spectral_Centroid'] = stimuli_feat.get('spectral_centroid', np.nan)
        features['Spectral_Bandwidth'] = stimuli_feat.get('spectral_bandwidth', np.nan)
        features['Spectral_Contrast'] = stimuli_feat.get('spectral_contrast', np.nan)
        features['Spectral_Rolloff'] = stimuli_feat.get('spectral_rolloff', np.nan)
        for i in range(1, 14):
            features[f'MFCC_{i}_Mean'] = stimuli_feat.get(f'mfcc_{i}_mean', np.nan)
            features[f'MFCC_{i}_Std'] = stimuli_feat.get(f'mfcc_{i}_std', np.nan)
        for i in range(12):
            features[f'Chroma_{i}'] = stimuli_feat.get(f'chroma_{i}', np.nan)

    concentration_level = behavioral_dict.get(subject_id, {}).get(norm_name, np.nan)
    features['Attention_Label'] = 'High Attention' if not pd.isna(concentration_level) and concentration_level >= 4.0 else 'Low Attention'

    features_df = pd.DataFrame([features])
    logger.info(f"    Extracted features for {subject_id}, {norm_name}")
    return features_df

def list_unique_stimuli(data, behavioral_dict):
    for subject_id in data.keys():
        eeg_stimuli = set([normalize_stimulus_name(stim) for stim in data[subject_id].keys() if '_stereo' in stim])
        behavioral_stimuli = set([normalize_stimulus_name(stim) for stim in behavioral_dict.get(subject_id, {}).keys() if '_stereo' in stim])

        missing_in_behavioral = eeg_stimuli - behavioral_stimuli
        extra_in_behavioral = behavioral_stimuli - eeg_stimuli

        if missing_in_behavioral:
            logger.warning(f"Subject {subject_id}: {len(missing_in_behavioral)} '_stereo' in EEG missing in behavioral.")
        if extra_in_behavioral:
            logger.warning(f"Subject {subject_id}: {len(extra_in_behavioral)} '_stereo' in behavioral not in EEG.")

def main():
    try:
        with open(YAML_PATH, 'r') as yaml_file:
            metadata = yaml.safe_load(yaml_file)
        logger.info(f"Loaded metadata from {YAML_PATH}")
    except Exception as e:
        logger.error(f"Failed to load YAML: {e}")
        sys.exit(1)

    try:
        data = h5py.File(HDF5_PATH, 'r')
        logger.info(f"Opened HDF5: {HDF5_PATH}")
    except Exception as e:
        logger.error(f"Failed to open HDF5: {e}")
        sys.exit(1)

    STIMULI_FEATURES_CSV = 'outputs/csv/stimuli_features.csv'
    if os.path.exists(STIMULI_FEATURES_CSV):
        try:
            stimuli_features_df = pd.read_csv(STIMULI_FEATURES_CSV)
            stimuli_features_df['stimulus_name'] = stimuli_features_df['stimulus_name'].apply(normalize_stimulus_name)
            stimuli_features_dict = stimuli_features_df.set_index('stimulus_name').to_dict('index')
            logger.info(f"Loaded stimulus features from {STIMULI_FEATURES_CSV}")
        except Exception as e:
            logger.error(f"Failed to load stimuli features CSV: {e}")
            sys.exit(1)
    else:
        logger.error(f"No stimuli features file at {STIMULI_FEATURES_CSV}.")
        sys.exit(1)

    if os.path.exists(BEHAVIORAL_DATA_PATH):
        try:
            behavioral_df = pd.read_excel(BEHAVIORAL_DATA_PATH)
            logger.info("Behavioral DataFrame dtypes:")
            logger.info(behavioral_df.dtypes)
            logger.info("\nSample behavioral data:")
            logger.info(behavioral_df.head())

            behavioral_dict = {}
            for idx, row in behavioral_df.iterrows():
                subject_id_raw = row['subject ID']
                if pd.isnull(subject_id_raw):
                    logger.warning(f"Row {idx}: missing 'subject ID'.")
                    continue
                try:
                    subject_id = str(int(subject_id_raw)).zfill(4)
                except:
                    logger.error(f"Invalid 'subject ID' in row {idx}: {subject_id_raw}")
                    continue

                stimuli_order_str = row['stimuli order']
                concentration_levels_str = row['concentration levels']
                stimuli_order_list, concentration_levels_list = preprocess_behavioral_data(stimuli_order_str, concentration_levels_str)
                if stimuli_order_list is None or concentration_levels_list is None:
                    logger.error(f"Parsing behavioral data failed for {subject_id}, row {idx}.")
                    continue

                if len(stimuli_order_list)!=len(concentration_levels_list):
                    logger.error(f"Mismatch stimuli vs levels for {subject_id}, row {idx}.")
                    continue

                for stim_name, concentration in zip(stimuli_order_list, concentration_levels_list):
                    if isinstance(stim_name, list):
                        stim_name = '_'.join(stim_name)
                    stim_name = normalize_stimulus_name(stim_name)
                    if '_stereo' not in stim_name:
                        continue
                    if subject_id not in behavioral_dict:
                        behavioral_dict[subject_id] = {}
                    behavioral_dict[subject_id][stim_name] = concentration

            logger.info(f"Loaded behavioral data from {BEHAVIORAL_DATA_PATH}")
            logger.info(f"Subjects in behavioral data: {len(behavioral_dict)}")
            for sid, stim_dict in behavioral_dict.items():
                logger.info(f"Subject {sid}: {len(stim_dict)} '_stereo' stimuli")

            for subject_id in behavioral_dict:
                for stim in list(behavioral_dict[subject_id].keys()):
                    normalized_stim = normalize_stimulus_name(stim)
                    if normalized_stim != stim:
                        behavioral_dict[subject_id][normalized_stim] = behavioral_dict[subject_id].pop(stim)
        except Exception as e:
            logger.error(f"Failed to load/parse behavioral data: {e}")
            sys.exit(1)
    else:
        logger.error(f"No behavioral file at {BEHAVIORAL_DATA_PATH}.")
        sys.exit(1)

    list_unique_stimuli(data, behavioral_dict)

    selected_channels = ['F1', 'F2', 'F3', 'F4']
    freq_bands = {
        'Delta': (1,4),
        'Theta': (4,8),
        'Alpha': (8,12),
        'Beta': (12,30)
    }

    all_features = []
    subjects = list(data.keys())
    logger.info(f"Processing Subjects: {subjects}")

    for subject_id in tqdm(subjects, desc="Processing Subjects"):
        stimuli = list(data[subject_id].keys())
        logger.info(f"Subject {subject_id}: {len(stimuli)} stimuli")
        for stimulus_name in stimuli:
            if '_stereo' not in stimulus_name:
                continue
            features_df = extract_madeeg_features(
                subject_id=subject_id,
                stimulus_name=stimulus_name,
                data=data,
                metadata=metadata,
                selected_channels=selected_channels,
                freq_bands=freq_bands,
                stimuli_features_dict=stimuli_features_dict,
                behavioral_dict=behavioral_dict
            )
            if features_df is not None:
                all_features.append(features_df)

    if all_features:
        final_df = pd.concat(all_features, ignore_index=True)
        logger.info(f"Final DataFrame shape: {final_df.shape}")
        attention_counts = final_df['Attention_Label'].value_counts(dropna=False)
        logger.info(f"Attention Label Distribution:\n{attention_counts}")

        logger.info("\nSample data after concatenation:")
        logger.info(final_df.head())

        final_processed = final_df.copy()
        final_processed.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Saved extracted features to {OUTPUT_CSV}")
    else:
        logger.warning("No features extracted.")

    data.close()
    logger.info("Processing completed.")

if __name__ == "__main__":
    main()
