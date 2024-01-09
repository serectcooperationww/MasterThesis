import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import random
import time

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss

from LSTMencoder_pytorch import LSTM, SequenceDataset
from resampling_method import sample_data





def preprocess_data(df):
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
    df['label'] = df['label'].map({'regular': 0, 'deviant': 1})
    df_sorted = df.sort_values(by=['Case ID', 'Complete Timestamp'])
    df_sorted['Activity'] = pd.factorize(df_sorted['Activity'])[0]

    scaler = MinMaxScaler()
    df_sorted['timesincelastevent'] = scaler.fit_transform(df_sorted[['timesincelastevent']])
    print(df_sorted)
    return(df_sorted)

def roll_sequence(data, case_column="Case ID"):
    trace = None
    for column in data.columns:
        if column != case_column:
            data_col = data.groupby(case_column)[column].apply(np.array)
            if trace is None:
                trace = data_col
            else:
                trace = pd.merge(trace, data_col, on=case_column, how='inner')
    return trace

def one_hot_encode_activity(df):
    max_val = max(max(lst) for lst in df['Activity'])
    max_length = max(len(lst) for lst in df['Activity'])

    df['activity_time_onehot'] = df.apply(lambda row:
                              [encoded_list + [row['timesincelastevent'][i]] for i, encoded_list in enumerate(
                                  [[int(i == val) for i in range(max_val + 1)] for val in row['Activity']]
                              )] + [[0] * (max_val + 1) + [0]] * (max_length - len(row['Activity'])),
                              axis=1
                              )

    return df

def spilt_train_test(feature, label):
    X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.2, stratify=label, random_state=42)
    return X_train, X_test, y_train, y_test

def Baseline_run(df):
    df_unbalanced = preprocess_data(df)
    padded_sequences_unbalanced = Encode_data(df_unbalanced)
    labels_unbalanced = df_unbalanced.groupby('Case ID')['label'].first().values
    lstm_features_unbalanced = LSTM_model(padded_sequences_unbalanced, labels_unbalanced)

    # Training
    X_train, X_test, y_train, y_test = spilt_train_test(lstm_features_unbalanced, labels_unbalanced)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_predict = rf_model.predict(X_test)
    print(classification_report(y_predict, y_test))

def train(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def get_test_set(X,y):
    # Split the dataset into 50% train and 50% test in a stratified manner
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y, random_state=42)

    # Further split the test set into 5 equal parts
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Create a list to store the 5 test subsets as DataFrames
    test_subsets = []

    # Split and store the test subsets
    for _, test_index in skf.split(X_test, y_test):
        X_test_subset = X_test[test_index]
        y_test_subset = y_test[test_index]

        test_subset_df = pd.DataFrame(X_test_subset)
        test_subset_df['label'] = y_test_subset
        test_subsets.append(test_subset_df)

    return X_train, y_train, test_subsets

def test(test_subsets, model):
    reports = []

    for count, test_subset_df in enumerate(test_subsets):
        X_test_subset = test_subset_df.iloc[:, :-1].values
        y_test_subset = test_subset_df['label'].values
        predictions = model.predict(X_test_subset)
        report = classification_report(y_test_subset, predictions, output_dict=True)
        print(f"report {count}:", report)
        reports.append(report)

    report_dfs = [pd.DataFrame(report).transpose() for report in reports]

    average_report = pd.DataFrame()
    for metric in ['precision', 'recall', 'f1-score', 'support']:
        metric_values = [report_df[metric] for report_df in report_dfs]
        average_metric = pd.concat(metric_values, axis=1).mean(axis=1)
        average_report[metric] = average_metric

    print(average_report)



def apply_resampling_and_classification(resampler, dataloader): #X, y,

    input_size = 4  # Number of features in each sequence
    hidden_size = 50
    num_classes = 2

    model = LSTM(input_size, hidden_size, num_classes)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10  # Adjust as needed
    for epoch in range(num_epochs):
        for sequences, labels in dataloader:
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    clf = RandomForestClassifier(random_state=0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    reports = []
    time_report = []

    for train_index, test_index in skf.split(X, y):
        start_time = time.time()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if resampler is not None:
            X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train
        clf.fit(X_resampled, y_resampled)
        y_pred = clf.predict(X_test)

        reports.append(classification_report(y_test, y_pred, output_dict=True))

        end_time = time.time()
        Execution_time = end_time - start_time
        time_report.append(Execution_time)

    return reports, time_report



df = pd.read_csv('data/hospital_billing_2.csv', sep=';')
df = df.head(100)
df_unbalanced = preprocess_data(df)
df_rolled = roll_sequence(df_unbalanced)
df_onehotencoded = one_hot_encode_activity(df_rolled)

Encoded_data = SequenceDataset(df_onehotencoded)
X, y, z, k = Encoded_data[:]
dataloader = DataLoader(Encoded_data, batch_size=32, shuffle=True)


resampling_techniques = {
        "Random Over-Sampling": RandomOverSampler(random_state=0),
        "Random Under-Sampling": RandomUnderSampler(random_state=0),
        "SMOTE": SMOTE(random_state=0),
        "ADASYN": ADASYN(random_state=0),
        "NearMiss": NearMiss(version=1),
        "No Filter": None
    }

# Applying resampling techniques and classification
results = {}
time_report_all = {}
for name, resampler in resampling_techniques.items():
    results[name], time_report_all[name] = apply_resampling_and_classification(resampler, dataloader) #X, y,

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
