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
cmc_features_directory = 'CMC_features'

def extract_label_from_filename(filename):
    # Extract class label from filename, assuming format 'S1_R1_G1_filtered_features.mat'
    label_str = filename.split('_')[2]
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

# Load CMC features
X, y, filenames = load_features(cmc_features_directory)


# Normalize the features
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)

# Reshape normalized features to 2D arrays
X_normalized = X_normalized.reshape(X_normalized.shape[0], -1)


# Impute NaN values
imputer = SimpleImputer(strategy='mean')
X_normalized = imputer.fit_transform(X_normalized)

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y, test_size=0.2, random_state=42)


# Train the SVM on data
svm = SVC(kernel='linear', C=1, random_state=42)
svm.fit(X_train, y_train)

# Predict the test set
y_pred = svm.predict(X_test)

# Evaluate the CMC model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f'CMC Model Evaluation:')
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1-Score: {f1:.2f}')
print('Classification Report CMC:')
print(classification_report(y_test, y_pred))

# Confusion matrix for CMC
labels = ['LDG', 'MRDG', 'TFSG', 'PPG', 'PG', 'Cut', 'Rest']
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - CMC')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()