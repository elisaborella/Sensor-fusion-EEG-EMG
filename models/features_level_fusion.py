#currently working but with very low performances, to be improved
import scipy.io
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV

np.random.seed(42)

# Function to load EEG features and select the first 19 features
def load_eeg_features(file_path):
    try:
        mat = scipy.io.loadmat(file_path)
        features = mat['features'][:19, :]  # Select only the first 19 features
        eeg_features = np.array([[float(entry) for entry in row] for row in features])  # Convert scientific notation
        return eeg_features
    except Exception as e:
        print(f"Error loading EEG features from {file_path}: {e}")
        return None

# Function to load EMG features
def load_emg_features(file_path):
    try:
        mat = scipy.io.loadmat(file_path)
        features = mat['features']
        emg_features = np.array([[float(entry) for entry in row] for row in features])  # Convert scientific notation
        return emg_features
    except Exception as e:
        print(f"Error loading EMG features from {file_path}: {e}")
        return None

# Function to normalize features
def normalize_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler

# Function to fuse EEG and EMG features
def fuse_features(eeg_features, emg_features):
    # Adjust number of rows to match
    min_samples = min(eeg_features.shape[0], emg_features.shape[0])
    eeg_features = eeg_features[:min_samples, :]
    emg_features = emg_features[:min_samples, :]
    
    # Concatenate EEG and EMG features
    fused_features = np.concatenate((eeg_features, emg_features), axis=1)
    return fused_features

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

# Initialize lists to store concatenated features and labels
all_fused_features = []
all_labels = []

# Iterate through all files in the EEG feature directory
for file_name in os.listdir(eeg_feature_dir):
    if file_name.endswith('.mat'):
        eeg_file_path = os.path.join(eeg_feature_dir, file_name)
        emg_file_path = os.path.join(emg_feature_dir, file_name)
        
        # Check if both EEG and EMG feature files exist
        if not os.path.exists(emg_file_path):
            print(f"Missing EMG file for: {eeg_file_path}")
            continue
        
        print(f"Processing files: {eeg_file_path} and {emg_file_path}")
        
        # Load EEG features
        eeg_features = load_eeg_features(eeg_file_path)
        if eeg_features is None:
            print(f"Skipping file due to error in EEG features: {eeg_file_path}")
            continue
        print(f"EEG Features shape before NaN removal: {eeg_features.shape}")
        
        # Load EMG features
        emg_features = load_emg_features(emg_file_path)
        if emg_features is None:
            print(f"Skipping file due to error in EMG features: {emg_file_path}")
            continue
        print(f"EMG Features shape before NaN removal: {emg_features.shape}")
        
        # Find NaN rows in EEG features
        nan_indices_eeg = np.isnan(eeg_features).any(axis=1)
        print(f"Number of NaN rows in EEG features: {np.sum(nan_indices_eeg)}")
        
        # Find NaN rows in EMG features
        nan_indices_emg = np.isnan(emg_features).any(axis=1)
        print(f"Number of NaN rows in EMG features: {np.sum(nan_indices_emg)}")
        
        # Remove common NaN rows
        nan_indices_common = nan_indices_eeg | nan_indices_emg
        eeg_features_clean = eeg_features[~nan_indices_common]
        emg_features_clean = emg_features[~nan_indices_common]
        print(f"Number of rows after NaN removal: {eeg_features_clean.shape[0]}")
        
        # Normalize EEG and EMG features
        eeg_features_normalized, _ = normalize_features(eeg_features_clean)
        emg_features_normalized, _ = normalize_features(emg_features_clean)
        
        # Fuse EEG and EMG features
        fused_features = fuse_features(eeg_features_normalized, emg_features_normalized)
        print(f"Fused Features shape: {fused_features.shape}")
        
        # Extract label from file path
        try:
            label = extract_label(eeg_file_path)
            print(f"Extracted label: {label}")
        except ValueError as e:
            print(e)
            continue  # Skip files with invalid labels

        # Append fused features and labels to the lists
        all_fused_features.append(fused_features)
        all_labels.extend([label] * fused_features.shape[0])

# Convert lists to numpy arrays
if all_fused_features:
    all_fused_features = np.vstack(all_fused_features)
    all_labels = np.array(all_labels)
    
    print("All extracted labels:", all_labels)
    
    # Split dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(all_fused_features, all_labels, test_size=0.3, random_state=42)
    
    # Perform Grid Search with Cross-Validation to find the best hyperparameters
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto'],
        'kernel': ['rbf']
    }
    
    print("Starting Grid Search with Cross-Validation...")
    grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print("Grid Search finished.")
    
    best_model = grid_search.best_estimator_
    print("Best Parameters found by Grid Search:")
    print(grid_search.best_params_)
    
    y_pred = best_model.predict(X_test)
    
    # Calculate performance metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)
    
    # Print performance metrics
    print(f"Performance Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    
    # Print confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
else:
    print("No valid feature data available for training and testing.")
