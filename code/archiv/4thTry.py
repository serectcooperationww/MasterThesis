import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report

from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss


def preprocess_data(df):
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
    df['label'] = df['label'].map({'regular': 0, 'deviant': 1})
    df_sorted = df.sort_values(by=['Case ID', 'Complete Timestamp'])
    return df_sorted

def encode_data(df):
    # Encoding
    label_encoder = LabelEncoder()
    df['Activity_encoded'] = label_encoder.fit_transform(df['Activity'])

    # Normalize 'timesincelastevent'
    scaler = MinMaxScaler()
    df['timesincelastevent'] = scaler.fit_transform(df[['timesincelastevent']])

    # Group by 'Case ID' and prepare sequences
    grouped = df.groupby('Case ID')
    features = ['Activity_encoded', 'timesincelastevent']
    sequences = [group[features].values.tolist() for _, group in grouped]

    # Pad sequences
    padded_sequences = pad_sequences(sequences, padding='post', dtype='float32')
    return padded_sequences

def train_lstm_model(padded_sequences, labels):
    features = ['Activity_encoded', 'timesincelastevent']
    n_features = len(features)
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(None, n_features)))
    lstm_model.add(Dense(50, activation='relu'))
    lstm_model.add(Dense(1, activation='sigmoid'))

    lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    lstm_model.fit(padded_sequences, labels, epochs=10, batch_size=32)

    return lstm_model


def apply_resampling_and_classification(X, y, resampler):
    clf = RandomForestClassifier(random_state=0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    reports = []

    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        clf.fit(X_resampled, y_resampled)
        y_pred = clf.predict(X_test)

        reports.append(classification_report(y_test, y_pred, output_dict=True))

    return reports

df = pd.read_csv('../data/hospital_billing_2.csv', sep=';')
df_preprocessed = preprocess_data(df)

padded_sequences = encode_data(df_preprocessed)
labels = df_preprocessed.groupby('Case ID')['label'].first().values

lstm_model = train_lstm_model(padded_sequences, labels)

# Extract features from the LSTM model
lstm_features = lstm_model.predict(padded_sequences)

# Resampling techniques
resampling_techniques = {
    "Random Over-Sampling": RandomOverSampler(random_state=0),
    "Random Under-Sampling": RandomUnderSampler(random_state=0),
    "SMOTE": SMOTE(random_state=0),
    "ADASYN": ADASYN(random_state=0),
    "NearMiss": NearMiss(version=1)
}

# Applying resampling techniques and classification
results = {}
for name, resampler in resampling_techniques.items():
    results[name] = apply_resampling_and_classification(lstm_features, labels, resampler)

# Initialize a dictionary to store the averaged results
averaged_results = {}

for name, reports in results.items():
    # Initialize dictionaries to store summed metrics for each label
    summed_metrics = {
        '0': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},  # Metrics for 'regular' (label 0)
        '1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}   # Metrics for 'deviant' (label 1)
    }

    # Sum up the metrics across all folds
    for report in reports:
        for label in ['0', '1']:
            summed_metrics[label]['precision'] += report[str(label)]['precision']
            summed_metrics[label]['recall'] += report[str(label)]['recall']
            summed_metrics[label]['f1-score'] += report[str(label)]['f1-score']
            summed_metrics[label]['support'] += report[str(label)]['support']

    # Calculate the average for each metric
    avg_metrics = {}
    num_folds = len(reports)
    for label in ['0', '1']:
        avg_metrics[label] = {k: v / num_folds for k, v in summed_metrics[label].items()}

    # Store in the results dictionary
    averaged_results[name] = avg_metrics

# Display the averaged results
for method, avg_metrics in averaged_results.items():
    print(f"Method: {method}")
    for label, metrics in avg_metrics.items():
        print(f"Label {label} Metrics:", metrics)
    print("\n")
