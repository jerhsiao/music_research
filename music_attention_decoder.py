import numpy as np
import h5py
import yaml
import librosa
from mtrf.model import TRF
from scipy.stats import pearsonr
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from scipy import stats

class MusicAttentionDecoder:
    def __init__(self, data_path='./datasets/madeeg_preprocessed.hdf5',
                 metadata_path='./datasets/madeeg_preprocessed.yaml'):
        self.data_path = Path(data_path)
        self.metadata_path = Path(metadata_path)
        self.trf = TRF(direction= -1)  # Backward model for attention decoding
        self.tmin = 0
        self.tmax = 0.25
        self.fs = 256  # EEG sampling frequency

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def compute_features(self, audio, sr, target_len=None):
        """Compute spectrogram features."""
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=0)

        # Compute spectrogram
        D = librosa.stft(audio, n_fft=2048, hop_length=int(sr / self.fs))
        mag = np.abs(D)

        # Convert to mel scale and normalize
        mel_basis = librosa.filters.mel(sr=sr, n_fft=2048, n_mels=40)
        mel_spec = np.dot(mel_basis, mag)

        # Convert to dB scale
        features = librosa.power_to_db(mel_spec, ref=np.max)
        features = features.T  # Shape: (time, features)

        if target_len is not None:
            if features.shape[0] > target_len:
                features = features[:target_len]
            elif features.shape[0] < target_len:
                pad_width = ((0, target_len - features.shape[0]), (0, 0))
                features = np.pad(features, pad_width, mode='constant')

        return features

    def load_data(self, subject_id):
        """Load duet trials and prepare data for classification."""
        with h5py.File(self.data_path, 'r') as f:
            with open(self.metadata_path, 'r') as stream:
                metadata = yaml.safe_load(stream)

            subject = f[subject_id]
            subject_meta = metadata[subject_id]

            attended_features = []
            unattended_features = []
            eeg_responses = []

            for stim_name in subject.keys():
                stim_meta = subject_meta[stim_name]

                # Only process duet trials
                if 'duo' not in stim_meta['ensemble'].lower():
                    continue

                # Get individual instrument stimuli
                if 'soli' in subject[stim_name]:
                    soli = subject[stim_name]['soli'][()]  # Shape: (2, samples)
                else:
                    self.logger.warning(f"'soli' data not found for {stim_name}. Skipping trial.")
                    continue

                resp = subject[stim_name]['response'][()]  # Shape: (channels, samples)
                sr = stim_meta['wav_info']['sfreq']

                # Get attended instrument index
                target = stim_meta['target']
                instruments = stim_meta['instruments']
                if target in instruments:
                    attended_idx = instruments.index(target)
                    unattended_idx = 1 - attended_idx  # Other instrument in duo
                else:
                    self.logger.warning(f"Target instrument not found in instruments list for {stim_name}. Skipping trial.")
                    continue

                # Compute features for both instruments
                attended = self.compute_features(soli[attended_idx], sr)
                unattended = self.compute_features(soli[unattended_idx], sr)

                # Align lengths
                min_len = min(attended.shape[0], unattended.shape[0], resp.shape[1])
                attended = attended[:min_len]
                unattended = unattended[:min_len]
                resp = resp[:, :min_len]  # Shape: (n_channels, n_samples)

                # Verify shapes
                if attended.shape[0] != resp.shape[1]:
                    self.logger.warning(f"Feature and EEG sample counts do not match for trial {stim_name}. Skipping trial.")
                    continue

                attended_features.append(attended)
                unattended_features.append(unattended)
                eeg_responses.append(resp)

        return eeg_responses, attended_features, unattended_features

    def train_decoder(self, subject_id):
        """Train decoder with multiple trials."""
        eeg_responses, attended_features, _ = self.load_data(subject_id)

        if not attended_features:
            self.logger.error(f"No valid data for subject {subject_id}")
            return

        # Truncate to minimum length
        min_samples = min(eeg.shape[1] for eeg in eeg_responses)
        
        # Prepare data in correct format
        response = [eeg[:, :min_samples].T for eeg in eeg_responses]  # Shape: (trials, samples, channels)
        stimulus = [feat[:min_samples, :] for feat in attended_features]  # Shape: (trials, samples, features)
        
        self.logger.info(f"Number of trials: {len(stimulus)}")
        self.logger.info(f"Stimulus shape: {stimulus[0].shape}")
        self.logger.info(f"Response shape: {response[0].shape}")

        # Train with regularization range
        regularization = np.logspace(0, 4, 20)
        self.trf.train(response, stimulus, self.fs, self.tmin, self.tmax, regularization)

    def evaluate(self, subject_id):
        """Compare correlations for attended vs unattended instruments."""
        eeg_responses, attended_features, unattended_features = self.load_data(subject_id)

        if not attended_features:
            self.logger.warning(f"No valid duet trials found for subject {subject_id}.")
            return np.array([]), np.array([])

        # Truncate all trials to minimum length
        min_samples = min(eeg.shape[1] for eeg in eeg_responses)
        eeg_responses = [eeg[:, :min_samples] for eeg in eeg_responses]
        attended_features = [feat[:min_samples, :] for feat in attended_features]
        unattended_features = [feat[:min_samples, :] for feat in unattended_features]

        attended_correlations = []
        unattended_correlations = []

        for idx, (eeg, att_feat, unatt_feat) in enumerate(zip(eeg_responses, attended_features, unattended_features)):
            # Normalize
            eeg = (eeg - np.mean(eeg)) / np.std(eeg)
            att_feat = (att_feat - np.mean(att_feat)) / np.std(att_feat)
            unatt_feat = (unatt_feat - np.mean(unatt_feat)) / np.std(unatt_feat)

            # For backward model, predict using both EEG and features
            try:
                # Predict using attended features
                pred_att, r_att = self.trf.predict([eeg.T], [att_feat])
                
                # Predict using unattended features
                pred_unatt, r_unatt = self.trf.predict([eeg.T], [unatt_feat])
                
                attended_correlations.append(r_att)
                unattended_correlations.append(r_unatt)
                
                self.logger.debug(f"Trial {idx+1}: r_att={r_att:.3f}, r_unatt={r_unatt:.3f}")
                
            except ValueError as ve:
                self.logger.error(f"Prediction failed for trial {idx+1}: {ve}")
                continue

        return np.array(attended_correlations), np.array(unattended_correlations)

def main():
    decoder = MusicAttentionDecoder()

    with h5py.File(decoder.data_path, 'r') as f:
        subject_ids = list(f.keys())

    all_attended = []
    all_unattended = []

    for subject in subject_ids:
        print(f"Processing subject {subject}")
        print("Training decoder...")
        decoder.train_decoder(subject)
        print("Evaluating...")
        att_corr, unatt_corr = decoder.evaluate(subject)

        if att_corr.size == 0:
            print(f"No valid data for subject {subject}.")
            continue

        all_attended.extend(att_corr)
        all_unattended.extend(unatt_corr)

        print(f"Attended correlation: {np.mean(att_corr):.3f} ± {np.std(att_corr):.3f}")
        print(f"Unattended correlation: {np.mean(unatt_corr):.3f} ± {np.std(unatt_corr):.3f}")

    # Convert to numpy arrays
    all_attended = np.array(all_attended)
    all_unattended = np.array(all_unattended)

    # Statistical test
    t_stat, p_value = stats.ttest_rel(all_attended, all_unattended)
    print(f"\nStatistical Analysis:")
    print(f"Paired t-test: t = {t_stat:.3f}, p = {p_value:.3e}")

    # Compute effect size (Cohen's d)
    mean_diff = np.mean(all_attended) - np.mean(all_unattended)
    pooled_std = np.sqrt((np.std(all_attended) ** 2 + np.std(all_unattended) ** 2) / 2)
    d = mean_diff / pooled_std
    print(f"Cohen's d: {d:.3f}")

    # Create Figure 3 style plot
    fig = plt.figure(figsize=(12, 8))

    # Main scatter plot
    ax_scatter = plt.axes([0.1, 0.1, 0.6, 0.6])
    ax_scatter.scatter(all_attended, all_unattended, alpha=0.5, color='blue')

    # Plot diagonal line
    min_val = min(np.min(all_attended), np.min(all_unattended))
    max_val = max(np.max(all_attended), np.max(all_unattended))
    ax_scatter.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)

    # Add grid
    ax_scatter.grid(True, alpha=0.3)

    # Labels
    ax_scatter.set_xlabel('Correlation with Attended')
    ax_scatter.set_ylabel('Correlation with Unattended')

    # Histograms
    ax_hist_x = plt.axes([0.1, 0.7, 0.6, 0.2])
    ax_hist_y = plt.axes([0.7, 0.1, 0.2, 0.6])

    # Plot histograms
    ax_hist_x.hist(all_attended, bins=20, density=True, alpha=0.5, color='blue')
    ax_hist_y.hist(all_unattended, bins=20, density=True, alpha=0.5,
                  color='blue', orientation='horizontal')

    # Remove some tick labels
    ax_hist_x.set_xticklabels([])
    ax_hist_y.set_yticklabels([])

    # Add title with statistics
    plt.suptitle('Attention Decoding Performance\n' +
                 f'Attended: {np.mean(all_attended):.3f}±{np.std(all_attended):.3f}, ' +
                 f'Unattended: {np.mean(all_unattended):.3f}±{np.std(all_unattended):.3f}\n' +
                 f'p = {p_value:.3e}, Cohen\'s d = {d:.3f}', y=0.95)
    # Save and show
    plt.savefig('outputs/visualizations/attention_decoding_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
