import os
import numpy as np
import scipy.io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, \
    confusion_matrix
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt

# Define paths and constants
emg_features_directory = '../Data/EMG_features'
eeg_features_directory = '../Data/EEG_features'


def extract_label_from_filename(filename):
    # Extract class label from filename, assuming format 'S1_R1_G1_filtered_features.mat'
    label_str = filename.split('_')[2]  # Assuming label is the third part after splitting by '_'
    label = int(label_str[1:])  # Remove 'G' and convert to integer
    return label


def load_features(directory):
    X = []
    y = []
    filenames = []

    for file_name in sorted(os.listdir(directory)):
        if file_name.endswith('.mat'):
            file_path = os.path.join(directory, file_name)
            data = scipy.io.loadmat(file_path)
            features = data['features']  # Assuming the feature matrix is named 'features' in the .mat files
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

# Find the common filenames between EEG and EMG
common_filenames = sorted(set(emg_filenames).intersection(eeg_filenames))

# Initialize lists to store combined features and labels
X_combined = []
y_combined = []

# Iterate over common filenames and combine features and labels
for filename in common_filenames:
    emg_idx = emg_filenames.index(filename)
    eeg_idx = eeg_filenames.index(filename)

    X_combined.append(np.concatenate([X_emg[emg_idx], X_eeg[eeg_idx]], axis=1))
    y_combined.append(y_emg[emg_idx])  # Using EMG label for combined data

# Convert combined lists to numpy arrays
X_combined = np.array(X_combined)
y_combined = np.array(y_combined)

# Normalize the features
scaler = StandardScaler()
X_combined_reshaped = X_combined.reshape(-1, X_combined.shape[-1])
X_combined_normalized = scaler.fit_transform(X_combined_reshaped)
X_combined_normalized = X_combined_normalized.reshape(X_combined.shape)

# Impute NaN values
imputer = SimpleImputer(strategy='mean')
X_combined_normalized = imputer.fit_transform(X_combined_normalized.reshape(X_combined_normalized.shape[0], -1))

# Split data into training (80%) and testing (20%) sets
random_seed = 42
X_train, X_test, y_train, y_test = train_test_split(X_combined_normalized, y_combined, test_size=0.2,
                                                    random_state=random_seed)

# Train the SVM
svm = SVC(kernel='linear', C=10, random_state=random_seed)
svm.fit(X_train, y_train)

# Predict the test set
y_pred = svm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')

# Print detailed classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Confusion matrix
labels = ['LDG', 'MRDG', 'TFSG', 'PPG', 'PG', 'Cut', 'Rest']
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()