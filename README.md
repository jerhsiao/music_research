# EEG Music Attention Decoding Project

This project includes three main scripts for processing EEG data and analyzing audio stimuli:

- **`analyze_stimuli.py`**: Extracts musical features (tempo, key, spectral properties, MFCCs, chroma) from audio stimuli.
- **`process_madeeg.py`**: Processes EEG data from the MAD-EEG dataset, integrates behavioral and musical features, and generates a comprehensive CSV file with attention labels. 
- **`process_physionet.py`**: Processes PhysioNet EEG data recorded during mental arithmetic tasks, extracts features, and trains a predictive model for attention and performance quality. Creates visualizations and pkl files for the machine learning models.

This project also includes a script for an attention decoder based on the MAD EEG paper's attention decoder to attended instrument vs unattended instrument. 

- **`music_attention_decoder.py`**: 
The `MusicAttentionDecoder` class implements a backward temporal response function model to predict and evaluate listeners' attention toward attended versus unattended musical instruments using EEG data and extracted audio features.


## Datasets used:

### MAD-EEG Dataset
[MAD-EEG: an EEG dataset for decoding auditory attention to a target instrument in polyphonic music](https://zenodo.org/records/4537751#.YS5MOI4zYuU)
- EEG data from subjects listening to polyphonic music
- Focus on auditory attention decoding
- Used to validate our attention decoding approach

### PhysioNet Mental Arithmetic Dataset  
[EEG During Mental Arithmetic Tasks](https://physionet.org/content/eegmat/1.0.0/)
- EEG recordings during arithmetic problem solving
- Provides baseline cognitive task data
- Used to understand EEG patterns during analytical tasks

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
