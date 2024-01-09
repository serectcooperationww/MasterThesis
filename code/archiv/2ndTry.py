import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report

from imblearn.under_sampling import RandomUnderSampler


def preprocess_data(df):
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
    df['label'] = df['label'].replace(['regular'], '0')
    df['label'] = df['label'].replace(['deviant'], '1')
    df['label'] = df['label'].astype(int)
    df_sorted = df.sort_values(by=['Case ID', 'Complete Timestamp'])
    print(df_sorted)
    return(df_sorted)

def sample_data(X, y):
    # df_majority = df[df['label'] == 'regular']
    # df_minority = df[df['label'] == 'deviant']
    # df_majority_downsampled = df_majority.sample(len(df_minority), random_state=42)
    # df_balanced = pd.concat([df_majority_downsampled, df_minority])

    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    return(X_resampled, y_resampled)


def Encode_data(df):
    # Encoding
    label_encoder = LabelEncoder()
    df['Activity_encoded'] = label_encoder.fit_transform(df['Activity'])

    # Normalize 'timesincelastevent'
    scaler = MinMaxScaler()
    df['timesincelastevent'] = scaler.fit_transform(df[['timesincelastevent']])

    # Group by 'case ID' and prepare sequences
    grouped = df.groupby('Case ID')
    features = ['Activity_encoded', 'timesincelastevent']
    sequences = [group[features].values.tolist() for _, group in grouped]

    # Pad sequences
    padded_sequences = pad_sequences(sequences, padding='post', dtype='float32')

    return padded_sequences


def LSTM_model(padded_sequences, labels):
    features = ['Activity_encoded', 'timesincelastevent']
    # Define LSTM model for feature extraction
    n_features = len(features)
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, activation='relu', input_shape=(None, n_features), return_sequences=True))
    lstm_model.add(LSTM(50, activation='relu', return_sequences=False))
    lstm_model.add(Dense(50, activation='relu'))
    lstm_model.add(Dense(1, activation='sigmoid'))  # Adjust based on your task

    # Compile the model
    lstm_model.compile(optimizer='adam', loss='binary_crossentropy')

    # Fit the LSTM model (this step requires your specific labels)
    # lstm_model.fit(padded_sequences, labels, epochs=10, batch_size=32)

    # Use the LSTM model to transform sequences into features
    lstm_features = lstm_model.predict(padded_sequences)
    return lstm_features

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


df = pd.read_csv('../data/hospital_billing_2.csv', sep=';')
df_unbalanced = preprocess_data(df)
padded_sequences_unbalanced = Encode_data(df_unbalanced)
labels_unbalanced = df_unbalanced.groupby('Case ID')['label'].first().values
lstm_features_unbalanced = LSTM_model(padded_sequences_unbalanced, labels_unbalanced)
X_train_notFiltered, y_train_notFiltered, test_subsets = get_test_set(lstm_features_unbalanced,labels_unbalanced)

# baseline
model = train(X_train_notFiltered, y_train_notFiltered)
test(test_subsets, model)

# sampled
X_train_RUS, y_train_RUS = sample_data(X_train_notFiltered, y_train_notFiltered)
print(X_train_RUS, y_train_RUS)
model = train(X_train_RUS, y_train_RUS)
test(test_subsets, model)