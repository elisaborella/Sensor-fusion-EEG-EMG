import os
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.impute import SimpleImputer
from scipy.stats import mode
import re
from collections import Counter

# Define paths and constants
emg_features_directory = 'EMG_features'
eeg_features_directory = 'EEG_features'

def extract_label_from_filename(filename):
    # Extract the class label from the filename using regex
    match = re.search(r'_G(\d+)_', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Filename {filename} does not contain a valid class label.")

def load_features_and_labels(directory):
    X = []
    y = []
    filenames = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.mat'):
            file_path = os.path.join(directory, file_name)
            data = scipy.io.loadmat(file_path)
            features = data['features']
            labels = [extract_label_from_filename(file_name)] * features.shape[0]
            X.append(features)
            y.extend(labels)
            filenames.append(file_name)
    return np.vstack(X), np.array(y), filenames

# Load EMG features and labels
X_emg, y_emg, emg_filenames = load_features_and_labels(emg_features_directory)

# Load EEG features and labels
X_eeg, y_eeg, eeg_filenames = load_features_and_labels(eeg_features_directory)

# Count label occurrences for EMG and EEG
emg_label_counts = Counter(y_emg)
eeg_label_counts = Counter(y_eeg)

# Print label distributions
print("EMG Label Distribution:")
for label, count in emg_label_counts.items():
    print(f"Class {label}: {count} samples")

print("\nEEG Label Distribution:")
for label, count in eeg_label_counts.items():
    print(f"Class {label}: {count} samples")

# Ensure the labels match for both modalities and print mismatches
mismatched_files = []
for i, (emg_file, eeg_file) in enumerate(zip(emg_filenames, eeg_filenames)):
    if not np.array_equal(emg_file, eeg_file):
        mismatched_files.append((emg_file, eeg_file))

if mismatched_files:
    print("Filenames with label mismatches:")
    for emg_file, eeg_file in mismatched_files:
        print(f"EMG file: {emg_file} | EEG file: {eeg_file}")
else:
    print("Filenames match for both EMG and EEG datasets.")

# Impute NaN values with the mean of each feature
imputer = SimpleImputer(strategy='mean')
X_emg = imputer.fit_transform(X_emg)
X_eeg = imputer.fit_transform(X_eeg)

# Normalize features
scaler_emg = StandardScaler()
scaler_eeg = StandardScaler()
X_emg = scaler_emg.fit_transform(X_emg)
X_eeg = scaler_eeg.fit_transform(X_eeg)

# Split data into training (80%) and testing (20%) sets
X_train_emg, X_test_emg, y_train_emg, y_test_emg = train_test_split(X_emg, y_emg, test_size=0.2, random_state=42)
X_train_eeg, X_test_eeg, y_train_eeg, y_test_eeg = train_test_split(X_eeg, y_eeg, test_size=0.2, random_state=42)

# Train SVM classifiers
svm_emg = SVC(kernel='linear', probability=True)
svm_eeg = SVC(kernel='linear', probability=True)

svm_emg.fit(X_train_emg, y_train_emg)
svm_eeg.fit(X_train_eeg, y_train_eeg)

# Predictions
y_pred_emg = svm_emg.predict(X_test_emg)
y_pred_eeg = svm_eeg.predict(X_test_eeg)

# Metrics
accuracy_emg = accuracy_score(y_test_emg, y_pred_emg)
precision_emg = precision_score(y_test_emg, y_pred_emg, average='macro')
recall_emg = recall_score(y_test_emg, y_pred_emg, average='macro')
f1_emg = f1_score(y_test_emg, y_pred_emg, average='macro')

accuracy_eeg = accuracy_score(y_test_eeg, y_pred_eeg)
precision_eeg = precision_score(y_test_eeg, y_pred_eeg, average='macro')
recall_eeg = recall_score(y_test_eeg, y_pred_eeg, average='macro')
f1_eeg = f1_score(y_test_eeg, y_pred_eeg, average='macro')

print("\nEMG SVM Classifier Metrics:")
print(f"Accuracy: {accuracy_emg:.4f}, Precision: {precision_emg:.4f}, Recall: {recall_emg:.4f}, F1-score: {f1_emg:.4f}")
print("\nEEG SVM Classifier Metrics:")
print(f"Accuracy: {accuracy_eeg:.4f}, Precision: {precision_eeg:.4f}, Recall: {recall_eeg:.4f}, F1-score: {f1_eeg:.4f}")
