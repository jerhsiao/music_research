#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# This script analyzes a set of .wav audio files (stimuli), extracts various audio features,
# and saves the extracted features as a CSV file for later use.

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
from mne import time_frequency
import librosa

import essentia.standard as es

# Paths
STIMULI_PATH = 'datasets/stimuli/'
OUTPUT_CSV = 'outputs/csv/stimuli_features.csv'
os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("outputs/process_madeeg.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger()

def normalize_stimulus_name(stim_name):
    name = re.sub(r'_(?:lcr|mono)', '', stim_name)
    name = re.sub(r'_$', '', name)
    return name

def estimate_key_essentia(audio_path):
    try:
        loader = es.MonoLoader(filename=audio_path)
        audio = loader()
        key_extractor = es.KeyExtractor()
        key, scale, strength = key_extractor(audio)
        return f"{key} {scale}"
    except Exception as e:
        logger.error(f"Error estimating key for {audio_path}: {e}")
        return 'Unknown'

def analyze_stimuli():
    stimuli_info = {}
    processed = set()
    audio_files = sorted(os.listdir(STIMULI_PATH))

    for audio_file in tqdm(audio_files, desc='Analyzing Stimuli'):
        if not audio_file.endswith('.wav'):
            continue
        full_name = os.path.splitext(audio_file)[0]
        name_no_number = re.sub(r'_\d+$', '', full_name)
        stimulus_name = name_no_number.replace('_stereo_lcr', '_stereo')

        if stimulus_name in processed:
            continue
        processed.add(stimulus_name)

        audio_path = os.path.join(STIMULI_PATH, audio_file)
        y, sr = librosa.load(audio_path, sr=None)
        features = {'stimulus_name': stimulus_name}

        tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = tempo

        chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_all = np.concatenate((chroma_cqt, chroma_cens, chroma_stft), axis=1)
        chroma_mean = np.mean(chroma_all, axis=1)

        key = estimate_key_essentia(audio_path)
        features['key'] = key

        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
        spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
        spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))

        features['spectral_centroid'] = spectral_centroid
        features['spectral_bandwidth'] = spectral_bandwidth
        features['spectral_contrast'] = spectral_contrast
        features['spectral_rolloff'] = spectral_rolloff

        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        for i in range(13):
            features[f'mfcc_{i+1}_mean'] = mfccs_mean[i]
            features[f'mfcc_{i+1}_std'] = mfccs_std[i]

        for i in range(12):
            features[f'chroma_{i}'] = chroma_mean[i]

        stimuli_info[stimulus_name] = features

    return pd.DataFrame.from_dict(stimuli_info, orient='index').reset_index(drop=True)

if __name__ == '__main__':
    df = analyze_stimuli()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Stimuli analysis completed. Features saved to '{OUTPUT_CSV}'.")
