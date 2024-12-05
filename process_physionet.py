"""
Process EEG data from the PhysioNet "EEG During Mental Arithmetic Tasks" dataset:
- Extract PSD-based EEG features (band powers).
- Compute Frontal Alpha Asymmetry and Hjorth parameters.
- Label data by "Count quality" (good/bad) and "Attention" (low/high) based on session.
- Save extracted features and train XGBoost models for attention level and performance quality prediction.
"""

import os
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_selection import RFE
from xgboost import XGBClassifier

import mne
import joblib
import logging
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    random.seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

def setup_logging(log_file='outputs/logs/process_physionet.log', verbose=True):
    logger = logging.getLogger('PhysioNetProcessor')
    logger.setLevel(logging.DEBUG)
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO if verbose else logging.WARNING)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    return logger

def verify_data_path(data_path):
    required_files = ['subject-info.csv']
    missing = [f for f in required_files if not os.path.isfile(os.path.join(data_path, f))]
    if missing:
        logger.error(f"Missing required files in '{data_path}': {missing}")
        return False
    logger.info(f"All required files are present in '{data_path}'.")
    return True

def load_group_labels(info_file_path):
    try:
        df = pd.read_csv(info_file_path, sep=',')
        if 'Subject' not in df.columns or 'Count quality' not in df.columns:
            logger.error("Missing required columns in subject-info.csv.")
            return None
        if df['Count quality'].isnull().any():
            logger.warning("Some 'Count quality' entries are missing.")
        return df[['Subject', 'Count quality']]
    except Exception as e:
        logger.error(f"Error loading group mapping: {e}")
        return None

def process_subject_session(subject_id, session, data_path, selected_channels, freq_bands, group_mapping_df):
    file_name = f"{subject_id}_{session}.edf"
    file_path = os.path.join(data_path, file_name)
    if not os.path.isfile(file_path):
        logger.warning(f"File not found: {file_path}")
        return None

    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        logger.debug(f"Loaded file: {file_path}")
    except Exception as e:
        logger.error(f"Error loading {file_path}: {e}")
        return None

    raw.pick(selected_channels)
    logger.debug(f"Selected channels: {selected_channels}")

    try:
        raw.filter(l_freq=1.0, h_freq=40.0, verbose=False)
        raw.notch_filter(freqs=50, verbose=False)
    except Exception as e:
        logger.error(f"Error filtering data for {file_path}: {e}")
        return None

    try:
        epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True, verbose=False)
    except Exception as e:
        logger.error(f"Error creating epochs for {file_path}: {e}")
        return None

    try:
        psd_result = epochs.compute_psd(method='welch', fmin=1.0, fmax=40.0, n_fft=256, verbose=False)
        psds = psd_result.get_data()
        freqs = psd_result.freqs
        logger.debug(f"Computed PSD for {file_path}")
    except Exception as e:
        logger.error(f"Error computing PSD for {file_path}: {e}")
        return None

    try:
        quality_label = group_mapping_df.loc[group_mapping_df['Subject'] == subject_id, 'Count quality'].values[0]
    except IndexError:
        logger.error(f"No 'Count quality' found for {subject_id}.")
        return None

    quality_label = 'Bad Quality' if quality_label == 0 else 'Good Quality' if quality_label == 1 else 'Unknown'
    attention_label = 'Low Attention' if session == '1' else 'High Attention'

    features_list = []
    for epoch_idx in range(len(epochs)):
        epoch_psd = psds[epoch_idx]
        epoch_features = {}
        for band_name, (fmin, fmax) in freq_bands.items():
            freq_mask = (freqs >= fmin) & (freqs <= fmax)
            if not np.any(freq_mask):
                logger.warning(f"No freq found in {band_name} for {file_path}")
                continue
            band_power = epoch_psd[:, freq_mask].mean(axis=1)
            for ch_idx, ch_name in enumerate(selected_channels):
                epoch_features[f"{ch_name}_{band_name}"] = band_power[ch_idx]

        try:
            alpha_fp1 = epoch_features['EEG Fp1_Alpha']
            alpha_fp2 = epoch_features['EEG Fp2_Alpha']
            epoch_features['Frontal_Alpha_Asymmetry'] = alpha_fp1 - alpha_fp2
        except KeyError:
            epoch_features['Frontal_Alpha_Asymmetry'] = np.nan
            logger.warning(f"Missing Alpha band in {file_path}, epoch {epoch_idx}")

        epoch_data = epochs.get_data()[epoch_idx]
        for ch_idx, ch_name in enumerate(selected_channels):
            ch_data = epoch_data[ch_idx]
            activity = np.var(ch_data)
            diff1 = np.diff(ch_data)
            mobility = np.std(diff1)/np.std(ch_data) if np.std(ch_data)!=0 else 0
            diff2 = np.diff(diff1)
            complexity = (np.std(diff2)/np.std(diff1))/mobility if mobility!=0 and np.std(diff1)!=0 else 0
            epoch_features[f'{ch_name}_Activity'] = activity
            epoch_features[f'{ch_name}_Mobility'] = mobility
            epoch_features[f'{ch_name}_Complexity'] = complexity

        epoch_features['Subject'] = subject_id
        epoch_features['Session'] = session
        epoch_features['Attention_Label'] = attention_label
        epoch_features['Quality'] = quality_label
        features_list.append(epoch_features)

    features_df = pd.DataFrame(features_list)
    return features_df

def process_all_subjects(data_path, group_info_path):
    group_mapping_df = load_group_labels(group_info_path)
    if group_mapping_df is None:
        logger.error("Failed to load group labels.")
        return None

    subject_ids = group_mapping_df['Subject'].tolist()
    selected_channels = ['EEG Fp1', 'EEG Fp2', 'EEG F7', 'EEG F8']
    freq_bands = {'Delta': (1,4), 'Theta': (4,8), 'Alpha': (8,12), 'Beta': (12,30)}
    dataset_list = []

    for subject_id in tqdm(subject_ids, desc="Processing Subjects"):
        for session in ['1', '2']:
            logger.info(f"Processing {subject_id}, Session {session}")
            features_df = process_subject_session(subject_id, session, data_path, selected_channels, freq_bands, group_mapping_df)
            if features_df is not None:
                dataset_list.append(features_df)
            else:
                logger.warning(f"No data for {subject_id}, Session {session}")

    if dataset_list:
        dataset = pd.concat(dataset_list, ignore_index=True)
        logger.info("Combined all subject data into a single DataFrame")
        return dataset
    else:
        logger.error("No data processed for any subject.")
        return None

def train_and_evaluate_model(X_train, y_train, X_test, y_test, label_encoder, model_name):
    class_counts = Counter(y_train)
    logger.info(f'Training set class distribution: {class_counts}')

    if len(class_counts) == 2:
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        scale_pos_weight = class_counts[majority_class]/class_counts[minority_class]
        logger.info(f"scale_pos_weight: {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = None
        logger.warning("Multi-class classification: scale_pos_weight not used.")

    logger.info("Feature selection with RFE.")
    selector = RFE(estimator=XGBClassifier(eval_metric='logloss', random_state=42), n_features_to_select=15)
    selector.fit(X_train, y_train)
    selected_features = X_train.columns[selector.support_]
    logger.info(f"Selected features: {list(selected_features)}")

    X_train_selected = X_train[selected_features]
    X_test_selected = X_test[selected_features]
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    xgb_param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 10],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.7, 0.8, 1.0],
        'colsample_bytree': [0.7, 0.8, 1.0]
    }

    logger.info("XGBoost hyperparameter tuning.")
    xgb_classifier = XGBClassifier(eval_metric='logloss', random_state=42,
                                   scale_pos_weight=scale_pos_weight if scale_pos_weight else 1)
    xgb_random_search = RandomizedSearchCV(
        estimator=xgb_classifier,
        param_distributions=xgb_param_grid,
        n_iter=30,
        scoring='f1_macro',
        cv=3,
        random_state=42,
        n_jobs=-1
    )
    xgb_random_search.fit(X_train_scaled, y_train)
    best_xgb = xgb_random_search.best_estimator_
    logger.info(f"Best XGBoost params: {xgb_random_search.best_params_}")

    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(best_xgb, X_train_scaled, y_train, cv=skf, scoring='f1_macro')
    logger.info(f"CV F1 scores: {cv_scores}, Mean: {cv_scores.mean():.4f}")

    best_xgb.fit(X_train_scaled, y_train)
    y_pred = best_xgb.predict(X_test_scaled)

    logger.info("Classification Report:")
    target_names = [str(cls) for cls in label_encoder.classes_]
    logger.info("\n" + classification_report(y_test, y_pred, target_names=target_names))

    cm = confusion_matrix(y_test, y_pred)
    plot_confusion_matrix(cm, target_names, save_path=f'outputs/visualizations/confusion_matrix_{model_name}.png')
    plot_roc_curve(best_xgb, X_test_scaled, y_test, save_path=f'outputs/visualizations/roc_curve_{model_name}.png')

    if hasattr(best_xgb, 'feature_importances_'):
        feature_importances = best_xgb.feature_importances_
        fi_df = pd.DataFrame({'Feature': selected_features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
        fi_df.to_csv(f'outputs/csv/feature_importances_{model_name}.csv', index=False)
        logger.info("Feature Importances:\n" + str(fi_df))
        plot_feature_importances(best_xgb, selected_features, top_n=15, save_path=f'outputs/visualizations/feature_importances_{model_name}.png')
    else:
        fi_df = None
        logger.warning("No feature_importances_ in model.")

    joblib.dump(best_xgb, f'outputs/pkl/{model_name}_classifier.pkl')
    joblib.dump(scaler, f'outputs/pkl/{model_name}_scaler.pkl')
    joblib.dump(selected_features, f'outputs/pkl/{model_name}_selected_features.pkl')
    joblib.dump(label_encoder, f'outputs/pkl/{model_name}_label_encoder.pkl')
    logger.info(f"Saved {model_name} model and artifacts.")

    return best_xgb, fi_df

def plot_confusion_matrix(cm, class_names, save_path=None):
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()

def plot_roc_curve(model, X_test, y_test, save_path=None):
    y_pred_proba = model.predict_proba(X_test)
    y_test_binarized = label_binarize(y_test, classes=np.unique(y_test))
    if y_pred_proba.shape[1] == 2:
        y_scores = y_pred_proba[:, 1]
    else:
        y_scores = y_pred_proba.ravel()
    fpr, tpr, _ = roc_curve(y_test_binarized, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0,1], [0,1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()

def plot_feature_importances(model, feature_names, top_n=15, save_path=None):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_features = [feature_names[i] for i in indices[:top_n]]

    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances[indices[:top_n]], y=top_features, palette='viridis')
    plt.title("Top Feature Importances")
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()

if __name__ == "__main__":
    set_random_seeds(42)
    logger = setup_logging(log_file='outputs/logs/process_physionet.log', verbose=True)
    logger.info("Starting PhysioNet EEG Feature Extraction and Model Training")
    logger.info(f"MNE version: {mne.__version__}")

    data_path = 'datasets/eeg-during-mental-arithmetic-tasks-1.0.0/'
    group_info_path = os.path.join(data_path, 'subject-info.csv')
    if not verify_data_path(data_path):
        logger.error("Required files missing.")
        exit(1)

    dataset = process_all_subjects(data_path, group_info_path)
    if dataset is not None:
        logger.info(f"Data processing complete. Total samples: {len(dataset)}")
        os.makedirs('outputs/csv', exist_ok=True)
        dataset.to_csv('outputs/csv/physionet_features.csv', index=False)
        os.makedirs('outputs/pkl', exist_ok=True)
        os.makedirs('outputs/visualizations', exist_ok=True)

        X = dataset.drop(['Subject', 'Session', 'Attention_Label', 'Quality'], axis=1)

        # Attention prediction
        logger.info("Training model to predict Attention Levels")
        y_attention = dataset['Attention_Label']
        le_attention = LabelEncoder()
        y_attention_encoded = le_attention.fit_transform(y_attention)
        X_train_attn, X_test_attn, y_train_attn, y_test_attn = train_test_split(
            X, y_attention_encoded, test_size=0.2, random_state=42, stratify=y_attention_encoded
        )
        best_model_attention, feature_importances_attention = train_and_evaluate_model(
            X_train_attn, y_train_attn, X_test_attn, y_test_attn, le_attention, model_name='attention'
        )

        # Quality prediction
        logger.info("Training model to predict Performance Quality")
        y_quality = dataset['Quality']
        le_quality = LabelEncoder()
        y_quality_encoded = le_quality.fit_transform(y_quality)
        X_train_qual, X_test_qual, y_train_qual, y_test_qual = train_test_split(
            X, y_quality_encoded, test_size=0.2, random_state=42, stratify=y_quality_encoded
        )
        best_model_quality, feature_importances_quality = train_and_evaluate_model(
            X_train_qual, y_train_qual, X_test_qual, y_test_qual, le_quality, model_name='quality'
        )
    else:
        logger.error("Data processing failed.")
