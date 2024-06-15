import os
import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Define paths and constants
emg_features_directory = 'EMG_features'
eeg_features_directory = 'EEG_features'

def extract_label_from_filename(filename):
    # Extract class label from filename, assuming format 'S1_R1_G1_filtered_features.mat'
    label_str = filename.split('_')[2]  # Assuming label is the third part after splitting by '_'
    label = int(label_str[1:])  # Remove 'G' and convert to integer
    return label

def load_features(directory):
    X = []
    y = []
    filenames = []
    
    for file_name in os.listdir(directory):
        if file_name.endswith('.mat'):
            file_path = os.path.join(directory, file_name)
            data = scipy.io.loadmat(file_path)
            features = data['features']
            features = features.T  # Transpose to have features as rows and channels as columns
            X.append(features)
            
            # Extract label from filename
            label = extract_label_from_filename(file_name)
            y.append(label)
            filenames.append(file_name)
    
    return np.array(X), np.array(y), filenames

# Load EMG features
X_emg, y_emg, emg_filenames = load_features(emg_features_directory)

# Load EEG features
X_eeg, y_eeg, eeg_filenames = load_features(eeg_features_directory)

# Normalize the features
scaler = StandardScaler()
X_emg_normalized = scaler.fit_transform(X_emg.reshape(-1, X_emg.shape[-1])).reshape(X_emg.shape)
X_eeg_normalized = scaler.fit_transform(X_eeg.reshape(-1, X_eeg.shape[-1])).reshape(X_eeg.shape)

# Reshape normalized features to 2D arrays
X_emg_normalized = X_emg_normalized.reshape(X_emg_normalized.shape[0], -1)
X_eeg_normalized = X_eeg_normalized.reshape(X_eeg_normalized.shape[0], -1)

# Impute NaN values
imputer = SimpleImputer(strategy='mean')
X_emg_normalized = imputer.fit_transform(X_emg_normalized)
X_eeg_normalized = imputer.fit_transform(X_eeg_normalized)

# Split data into training (80%) and testing (20%) sets
X_train_emg, X_test_emg, y_train_emg, y_test_emg = train_test_split(X_emg_normalized, y_emg, test_size=0.2, random_state=42)
X_train_eeg, X_test_eeg, y_train_eeg, y_test_eeg = train_test_split(X_eeg_normalized, y_eeg, test_size=0.2, random_state=42)

common_filenames = list(set(emg_filenames).intersection(eeg_filenames))
common_indices_emg = [emg_filenames.index(filename) for filename in common_filenames]
common_indices_eeg = [eeg_filenames.index(filename) for filename in common_filenames]

# Train the SVM on EMG data
svm_emg = SVC(kernel='linear', C=1, random_state=42)
svm_emg.fit(X_train_emg, y_train_emg)

# Predict the test set for EMG
y_pred_emg = svm_emg.predict(X_test_emg)

# Evaluate the EMG model
accuracy_emg = accuracy_score(y_test_emg, y_pred_emg)
precision_emg = precision_score(y_test_emg, y_pred_emg, average='weighted')
recall_emg = recall_score(y_test_emg, y_pred_emg, average='weighted')
f1_emg = f1_score(y_test_emg, y_pred_emg, average='weighted')

print(f'EMG Model Evaluation:')
print(f'Accuracy: {accuracy_emg:.2f}')
print(f'Precision: {precision_emg:.2f}')
print(f'Recall: {recall_emg:.2f}')
print(f'F1-Score: {f1_emg:.2f}')
print('Classification Report EMG:')
print(classification_report(y_test_emg, y_pred_emg))

# Confusion matrix for EMG
labels = ['LDG', 'MRDG', 'TFSG', 'PPG', 'PG', 'Cut', 'Rest']
cm_emg = confusion_matrix(y_test_emg, y_pred_emg)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_emg, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - EMG')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Train the SVM on EEG data
svm_eeg = SVC(kernel='linear', C=1, random_state=42)
svm_eeg.fit(X_train_eeg, y_train_eeg)

# Predict the test set for EEG
y_pred_eeg = svm_eeg.predict(X_test_eeg)

# Evaluate the EEG model
accuracy_eeg = accuracy_score(y_test_eeg, y_pred_eeg)
precision_eeg = precision_score(y_test_eeg, y_pred_eeg, average='weighted')
recall_eeg = recall_score(y_test_eeg, y_pred_eeg, average='weighted')
f1_eeg = f1_score(y_test_eeg, y_pred_eeg, average='weighted')

print(f'EEG Model Evaluation:')
print(f'Accuracy: {accuracy_eeg:.2f}')
print(f'Precision: {precision_eeg:.2f}')
print(f'Recall: {recall_eeg:.2f}')
print(f'F1-Score: {f1_eeg:.2f}')
print('Classification Report EEG:')
print(classification_report(y_test_eeg, y_pred_eeg))

# Confusion matrix for EEG
cm_eeg = confusion_matrix(y_test_eeg, y_pred_eeg)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_eeg, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - EEG')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


def majority_vote(predictions):
    fused_predictions = []
    for i in range(len(predictions)):
        combined_prediction = np.argmax(np.bincount(predictions[i]))
        fused_predictions.append(combined_prediction)
    return np.array(fused_predictions)

# Perform decision-level fusion on common filenames
predictions_emg = svm_emg.predict(X_emg_normalized[common_indices_emg])
predictions_eeg = svm_eeg.predict(X_eeg_normalized[common_indices_eeg])
predictions = np.column_stack((predictions_emg, predictions_eeg))
y_pred_fused = majority_vote(predictions)

y_test = y_eeg[common_indices_eeg]

# Evaluate the fused model
accuracy_fused = accuracy_score(y_test, y_pred_fused)
precision_fused = precision_score(y_test, y_pred_fused, average='weighted')
recall_fused = recall_score(y_test, y_pred_fused, average='weighted')
f1_fused = f1_score(y_test, y_pred_fused, average='weighted')

print(f'Fusion Model Evaluation:')
print(f'Accuracy: {accuracy_fused:.2f}')
print(f'Precision: {precision_fused:.2f}')
print(f'Recall: {recall_fused:.2f}')
print(f'F1-Score: {f1_fused:.2f}')
print('Classification Report - Fusion:')
print(classification_report(y_test, y_pred_fused))

# Confusion matrix for fused model
cm_fused = confusion_matrix(y_test, y_pred_fused)
plt.figure(figsize=(10, 7))
sns.heatmap(cm_fused, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_emg), yticklabels=np.unique(y_emg))
plt.title('Confusion Matrix - Fused Model')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Print number of samples used for fusion
print(f'Number of samples used for fusion: {len(y_test)}')