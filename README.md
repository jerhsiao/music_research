# EEG Music Attention Decoding Project

This project includes three main scripts for processing EEG data and analyzing audio stimuli:

- **`analyze_stimuli.py`**: Extracts musical features (tempo, key, spectral properties, MFCCs, chroma) from audio stimuli.
- **`process_madeeg.py`**: Processes EEG data from the MAD-EEG dataset, integrates behavioral and musical features, and generates a comprehensive CSV file with attention labels. 
- **`process_physionet.py`**: Processes PhysioNet EEG data recorded during mental arithmetic tasks, extracts features, and trains a predictive model for attention and performance quality. Creates visualizations and pkl files for the machine learning models.

To use this program, clone the repository. Navigate to project directory. 

Install the required packages:

```pip install -r requirements.txt```

## Directory Setup and Data Download

Before running the scripts, you need to download the correct datasets and place them into the right folders. 

Download and place these two files in 'music_research/datasets'

https://zenodo.org/records/4537751#.YS5MOI4zYuU:~:text=madeeg_preprocessed.hdf5

https://zenodo.org/records/4537751#.YS5MOI4zYuU:~:text=madeeg_preprocessed.yaml

In '/datasets/eeg-during-mental-arithmetic-tasks-1.0.0', download and place contents of this data: https://physionet.org/content/eegmat/1.0.0/#:~:text=Download%20the%20ZIP%20file

In '/datasets/stimuli', download and place contents of these stimuli: https://zenodo.org/records/4537751#.YS5MOI4zYuU:~:text=Files-,stimuli,-.zip
