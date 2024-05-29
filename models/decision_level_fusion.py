#currently working but with very low performances, to be improved
import scipy.io
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.model_selection import GridSearchCV

np.random.seed(42)

# Function to load EEG features and select the first 19 features
def load_eeg_features(file_path):
    try:
        mat = scipy.io.loadmat(file_path)
        features = mat['features'].T  # Transpose to have 19 rows and 8 columns
        eeg_features = np.array(features)  # Convert to numpy array
        return eeg_features
    except Exception as e:
        print(f"Error loading EEG features from {file_path}: {e}")
        return None

# Function to load EMG features
def load_emg_features(file_path):
    try:
        mat = scipy.io.loadmat(file_path)
        features = mat['features'].T  # Transpose to have 19 rows and 8 columns
        emg_features = np.array(features)  # Convert to numpy array
        return emg_features
    except Exception as e:
        print(f"Error loading EMG features from {file_path}: {e}")
        return None

# Function to normalize features
def normalize_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler

# Function to extract label from file path
def extract_label(file_path):
    filename = os.path.basename(file_path)
    # Find the part of the filename that starts with 'G' and extract the number
    for part in filename.split('_'):
        if part.startswith('G') and part[1:].isdigit():
            label = int(part[1:]) - 1  # Convert to zero-based index
            return label
    raise ValueError(f"Filename does not contain a valid label: {filename}")

# Paths to the EEG and EMG feature directories
eeg_feature_dir = 'EEG_features'
emg_feature_dir = 'EMG_features'

# Step 1: Load EEG and EMG Features
def load_data():
    eeg_files = os.listdir(eeg_feature_dir)
    emg_files = os.listdir(emg_feature_dir)
    eeg_features_list = []
    emg_features_list = []
    eeg_labels = []
    emg_labels = []
    
    # Load EEG features and labels
    for file in eeg_files:
        eeg_file_path = os.path.join(eeg_feature_dir, file)
        eeg_features = load_eeg_features(eeg_file_path)
        if eeg_features is not None:
            eeg_features_list.append(eeg_features)
            eeg_labels.append(extract_label(eeg_file_path))
    
    # Load EMG features and labels
    for file in emg_files:
        emg_file_path = os.path.join(emg_feature_dir, file)
        emg_features = load_emg_features(emg_file_path)
        if emg_features is not None:
            emg_features_list.append(emg_features)
            emg_labels.append(extract_label(emg_file_path))
    
    # Convert lists to arrays
    X_eeg = np.vstack(eeg_features_list)
    X_emg = np.vstack(emg_features_list)
    
    # Generate labels for EEG and EMG features
    y_eeg = np.repeat(eeg_labels, X_eeg.shape[0] // len(eeg_labels))
    y_emg = np.repeat(emg_labels, X_emg.shape[0] // len(emg_labels))
    
    return X_eeg, X_emg, y_eeg, y_emg

# Load data
X_eeg, X_emg, y_eeg, y_emg = load_data()

# Step 2: Preprocess Features
X_eeg_scaled, eeg_scaler = normalize_features(X_eeg)

# Check for NaN values in EEG data
if np.isnan(X_eeg_scaled).any():
    print("Found NaN values in X_eeg_scaled. Imputing NaN values...")
    imputer = SimpleImputer(strategy='mean')
    X_eeg_scaled = imputer.fit_transform(X_eeg_scaled)
    print("NaN values imputed.")

X_emg_scaled, emg_scaler = normalize_features(X_emg)

# Check for NaN values in EMG data
if np.isnan(X_emg_scaled).any():
    print("Found NaN values in X_emg_scaled. Imputing NaN values...")
    imputer = SimpleImputer(strategy='mean')
    X_emg_scaled = imputer.fit_transform(X_emg_scaled)
    print("NaN values imputed.")

#print("Shape of X_eeg_scaled:", X_eeg_scaled.shape)
#print("Shape of y_eeg:", y_eeg.shape)

# Split data into training and testing sets
X_train_eeg, X_test_eeg, y_train_eeg, y_test_eeg = train_test_split(X_eeg_scaled, y_eeg, test_size=0.2, random_state=42)

# Feature analysis
eeg_df = pd.DataFrame(X_eeg_scaled, columns=['feature_{}'.format(i) for i in range(X_eeg_scaled.shape[1])])
eeg_df['label'] = y_eeg

# Compute mean of each feature grouped by label
mean_features_by_label = eeg_df.groupby('label').mean()

# Compute feature distributions by label (assuming Gaussian distribution)
eeg_distribution_stats = eeg_df.groupby('label').agg({f'feature_{i}': ['mean', 'std'] for i in range(X_eeg_scaled.shape[1])})

# Print mean features by label
#print("\nMean features by label for EEG:")
#print(mean_features_by_label)

# Print feature distribution statistics by label
print("\nFeature distribution statistics by label for EEG:")
print(eeg_distribution_stats)


# Assuming X_emg_scaled and y_emg are already loaded
#print("Shape of X_emg_scaled:", X_emg_scaled.shape)
#print("Shape of y_emg:", y_emg.shape)

# Split data into training and testing sets
X_train_emg, X_test_emg, y_train_emg, y_test_emg = train_test_split(X_emg_scaled, y_emg, test_size=0.2, random_state=42)

# Feature analysis
emg_df = pd.DataFrame(X_emg_scaled, columns=['feature_{}'.format(i) for i in range(X_emg_scaled.shape[1])])
emg_df['label'] = y_emg

# Compute mean of each feature grouped by label
mean_features_by_label_emg = emg_df.groupby('label').mean()

# Compute feature distributions by label (assuming Gaussian distribution)
emg_distribution_stats = emg_df.groupby('label').agg({f'feature_{i}': ['mean', 'std'] for i in range(X_emg_scaled.shape[1])})

# Print mean features by label for EMG
#print("\nMean features by label for EMG:")
#print(mean_features_by_label_emg)

# Print feature distribution statistics by label for EMG
print("\nFeature distribution statistics by label for EMG:")
print(emg_distribution_stats)

# Print the number of features in EEG dataset
print("Number of features in EEG dataset:", X_eeg_scaled.shape[1])

# Print the number of features in EMG dataset
print("Number of features in EMG dataset:", X_emg_scaled.shape[1])


# Step 3: Train SVM Models
# Split EEG and EMG data with the same random state and test size
X_train_eeg, X_test_eeg, y_train_eeg, y_test_eeg = train_test_split(X_eeg_scaled, y_eeg, test_size=0.2, random_state=42)
X_train_emg, X_test_emg, y_train_emg, y_test_emg = train_test_split(X_emg_scaled, y_emg, test_size=0.2, random_state=42)

#param_grid = {'C': [0.1, 1],'gamma': [1, 0.1, 0.01, 0.001],'kernel': ['rbf']}

svm_eeg = SVC(random_state=42, C=1, kernel='rbf')
svm_emg = SVC(random_state=42, C=1, kernel='rbf')

#grid_search_eeg = GridSearchCV(svm_eeg, param_grid, cv=5, scoring='accuracy')
#grid_search_emg = GridSearchCV(svm_emg, param_grid, cv=5, scoring='accuracy')

#grid_search_eeg.fit(X_train_eeg, y_train_eeg)
#grid_search_emg.fit(X_train_emg, y_train_emg)

#print("Best parameters for EEG SVM:", grid_search_eeg.best_params_)
#print("Best parameters for EMG SVM:", grid_search_emg.best_params_)

# Re-train SVM models with best parameters
#svm_eeg_best = grid_search_eeg.best_estimator_
#svm_emg_best = grid_search_emg.best_estimator_

svm_eeg.fit(X_train_eeg, y_train_eeg)
svm_emg.fit(X_train_emg, y_train_emg)

# Predict with best models
y_pred_eeg = svm_eeg.predict(X_test_eeg)
y_pred_emg = svm_emg.predict(X_test_emg)



# Step 6: Define Fusion Rule (e.g., Weighted Average)
# Align predictions by removing extra samples
if y_pred_eeg.shape[0] > y_pred_emg.shape[0]:
    y_pred_eeg = y_pred_eeg[:y_pred_emg.shape[0]]
    y_test_eeg = y_test_eeg[:y_pred_emg.shape[0]]
elif y_pred_emg.shape[0] > y_pred_eeg.shape[0]:
    y_pred_emg = y_pred_emg[:y_pred_eeg.shape[0]]
    y_test_emg = y_test_emg[:y_pred_eeg.shape[0]]

# Assuming the number of predictions for EEG and EMG models is now the same, we can perform a simple average
y_pred_fused = (y_pred_eeg + y_pred_emg) / 2

# Convert continuous scores to integer labels
y_pred_fused = np.round(y_pred_fused).astype(int)

# Step 7: Evaluate Performance
accuracy = accuracy_score(y_test_eeg, y_pred_fused)
precision = precision_score(y_test_eeg, y_pred_fused, average='weighted')
recall = recall_score(y_test_eeg, y_pred_fused, average='weighted')
f1 = f1_score(y_test_eeg, y_pred_fused, average='weighted')

print("\nFused Model Performance:")
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-score: {f1:.2f}")

# Print Confusion Matrix
cm = confusion_matrix(y_test_eeg, y_pred_fused)
print("\nConfusion Matrix:")
print(cm)

# Performance of Separate EEG Classifier
accuracy_eeg = accuracy_score(y_test_eeg, y_pred_eeg)
precision_eeg = precision_score(y_test_eeg, y_pred_eeg, average='weighted')
recall_eeg = recall_score(y_test_eeg, y_pred_eeg, average='weighted')
f1_eeg = f1_score(y_test_eeg, y_pred_eeg, average='weighted')

print("\nPerformance of EEG Classifier:")
print(f"Accuracy: {accuracy_eeg:.2f}")
print(f"Precision: {precision_eeg:.2f}")
print(f"Recall: {recall_eeg:.2f}")
print(f"F1-score: {f1_eeg:.2f}")

# Performance of Separate EMG Classifier
accuracy_emg = accuracy_score(y_test_emg, y_pred_emg)
precision_emg = precision_score(y_test_emg, y_pred_emg, average='weighted')
recall_emg = recall_score(y_test_emg, y_pred_emg, average='weighted')
f1_emg = f1_score(y_test_emg, y_pred_emg, average='weighted')

print("\nPerformance of EMG Classifier:")
print(f"Accuracy: {accuracy_emg:.2f}")
print(f"Precision: {precision_emg:.2f}")
print(f"Recall: {recall_emg:.2f}")
print(f"F1-score: {f1_emg:.2f}")
