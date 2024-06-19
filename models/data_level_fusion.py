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
from collections import Counter

cmc_features_directory = '../CMC_features'

# Function to extract label from filename
def extract_label_from_filename(filename):
    # Extract class label from filename, assuming format 'S1_R1_G1_filtered_features.mat'
    label_str = filename.split('_')[2]  # Assuming label is the third part after splitting by '_'
    label = int(label_str[1:])  # Remove 'G' and convert to integer
    return label

# Function to load features and labels
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


# Load features and labels from CMC results file
X, y, filenames = load_features(cmc_features_directory)

# Check if there's enough data for splitting
if len(X) < 2:
    raise ValueError('Insufficient data for splitting. You need at least 2 samples.')

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, 1))  # Reshape to 2D for scaler
X_test_scaled = scaler.transform(X_test.reshape(-1, 1))  # Reshape to 2D for scaler

# Train SVM model
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)

# Predict on test set
y_pred = svm.predict(X_test_scaled)

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
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


label_counter = Counter(y)
num_samples = len(y)
print(f'\nLabel Statistics:')
for label, count in label_counter.items():
    print(f'Label {label}: Count {count} ({count / num_samples:.2%})')