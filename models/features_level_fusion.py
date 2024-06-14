import os
import numpy as np
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.impute import SimpleImputer

# Define paths and constants
emg_features_directory = 'EMG_features'
eeg_features_directory = 'EEG_features'

def load_features(directory, label):
    X = []
    y = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.mat'):
            file_path = os.path.join(directory, file_name)
            data = scipy.io.loadmat(file_path)
            features = data['features']
            features_2d = features.reshape(features.shape[0], -1)  # Reshape to 2D
            X.append(features_2d)
            y.append(np.full(features_2d.shape[0], label))
    return np.vstack(X), np.hstack(y)

# Load EMG features (label=1 for EMG)
X_emg, y_emg = load_features(emg_features_directory, label=1)

# Load EEG features (label=2 for EEG)
X_eeg, y_eeg = load_features(eeg_features_directory, label=2)

# Combine EMG and EEG features
X = np.vstack([X_emg, X_eeg])
y = np.hstack([y_emg, y_eeg])

# Normalize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)


# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SVM
svm = SVC(kernel='rbf', C=1, random_state=42)
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
