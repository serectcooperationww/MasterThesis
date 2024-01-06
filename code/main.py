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
    df['label'] = df['label'].replace(['regular'], '0')
    df['label'] = df['label'].replace(['deviant'], '1')
    df['label'] = df['label'].astype(int)
    df_sorted = df.sort_values(by=['Case ID', 'Complete Timestamp'])
    print(df_sorted)
    return(df_sorted)

def sample_data(X, y):
    rus = RandomUnderSampler(random_state=0)
    X_resampled, y_resampled = rus.fit_resample(X, y)

    resampling_techniques = {
        "Random Over-Sampling": RandomOverSampler(random_state=0),
        "Random Under-Sampling": RandomUnderSampler(random_state=0),
        "SMOTE": SMOTE(random_state=0),
        "ADASYN": ADASYN(random_state=0),
        "NearMiss": NearMiss(version=1)
    }

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









def preprocess_data(df):
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
    df['label'] = df['label'].replace(['regular'], '0')
    df['label'] = df['label'].replace(['deviant'], '1')
    df['label'] = df['label'].astype(int)
    df_sorted = df.sort_values(by=['Case ID', 'Complete Timestamp'])
    print(df_sorted)
    return(df_sorted)

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
    lstm_model.fit(padded_sequences, labels, epochs=10, batch_size=32)

    # Use the LSTM model to transform sequences into features
    lstm_features = lstm_model.predict(padded_sequences)
    return lstm_features


def apply_resampling_and_classification(X, y, resampler):
    clf = RandomForestClassifier(random_state=0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    reports = []

    for train_index, test_index in skf.split(X, y):
        print("Train label distribution:")
        print(np.bincount(y[train_index]))
        print("Test label distribution:")
        print(np.bincount(y[test_index]))
        print("---")



        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        print("Resampled label distribution:")
        print(np.bincount(y_resampled))

        clf.fit(X_resampled, y_resampled)
        y_pred = clf.predict(X_test)

        reports.append(classification_report(y_test, y_pred, output_dict=True))

    return reports



df = pd.read_csv('data/hospital_billing_2.csv', sep=';')
df_unbalanced = preprocess_data(df)

padded_sequences_unbalanced = Encode_data(df_unbalanced)
labels_unbalanced = df_unbalanced.groupby('Case ID')['label'].first().values
lstm_features_unbalanced = LSTM_model(padded_sequences_unbalanced, labels_unbalanced)

X = lstm_features_unbalanced
y = labels_unbalanced

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
    results[name] = apply_resampling_and_classification(X, y, resampler)

# Initialize a dictionary to store the averaged results
averaged_results = {}

for name, reports in results.items():
    # Initialize dictionaries to store summed metrics
    deviant_metrics = {'precision': 0, 'recall': 0, 'f1-score': 0}
    regular_metrics = {'precision': 0, 'recall': 0, 'f1-score': 0}

    # Sum up the metrics across all folds
    for report in reports:
        deviant_metrics['precision'] += report['deviant']['precision']
        deviant_metrics['recall'] += report['deviant']['recall']
        deviant_metrics['f1-score'] += report['deviant']['f1-score']

        regular_metrics['precision'] += report['regular']['precision']
        regular_metrics['recall'] += report['regular']['recall']
        regular_metrics['f1-score'] += report['regular']['f1-score']

    # Calculate the average
    num_folds = len(reports)
    avg_deviant_metrics = {k: v / num_folds for k, v in deviant_metrics.items()}
    avg_regular_metrics = {k: v / num_folds for k, v in regular_metrics.items()}

    # Store in the results dictionary
    averaged_results[name] = {'deviant': avg_deviant_metrics, 'regular': avg_regular_metrics}

# Display the averaged results
for method, avg_metrics in averaged_results.items():
    print(f"Method: {method}")
    print("Deviant Metrics:", avg_metrics['deviant'])
    print("Regular Metrics:", avg_metrics['regular'])
    print("\n")










































# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
# import pandas as pd
# import pandas as pd
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from sklearn.preprocessing import LabelEncoder, MinMaxScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
#
#
# # Data preprocessing
# df = pd.read_csv('data/hospital_billing_2.csv', sep=';')
# df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
# df_sorted = df.sort_values(by=['Case ID', 'Complete Timestamp'])
# print(df_sorted)
#
#
# # Encoding
# label_encoder = LabelEncoder()
# df_sorted['Activity_encoded'] = label_encoder.fit_transform(df_sorted['Activity'])
#
# # Normalize 'TimeSinceLastEvent'
# scaler = MinMaxScaler()
# df_sorted['timesincelastevent'] = scaler.fit_transform(df_sorted[['timesincelastevent']])
#
# # Group by 'case ID' and prepare sequences
# grouped = df_sorted.groupby('Case ID')
# features = ['Activity_encoded', 'timesincelastevent']
# sequences = [group[features].values.tolist() for _, group in grouped]
#
# # Pad sequences
# padded_sequences = pad_sequences(sequences, padding='post', dtype='float32')
#
# # Prepare labels (assuming one label per case ID)
# labels = df_sorted.groupby('Case ID')['label'].first().values
#
# # Define LSTM model for feature extraction
# n_features = len(features)
# lstm_model = Sequential()
# lstm_model.add(LSTM(50, activation='relu', input_shape=(None, n_features), return_sequences=True))
# lstm_model.add(LSTM(50, activation='relu', return_sequences=False))
# lstm_model.add(Dense(50, activation='relu'))
# lstm_model.add(Dense(1, activation='sigmoid'))  # Adjust based on your task
#
# # Compile the model
# lstm_model.compile(optimizer='adam', loss='binary_crossentropy')
#
# # Fit the LSTM model (this step requires your specific labels)
# # lstm_model.fit(padded_sequences, labels, epochs=10, batch_size=32)
#
# # Use the LSTM model to transform sequences into features
# # Assuming you have a trained LSTM model
# lstm_features = lstm_model.predict(padded_sequences)
#
#
# # Training
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(lstm_features, labels, test_size=0.2, random_state=42)
#
# # Train a Random Forest Classifier
# rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_model.fit(X_train, y_train)
#
# # Evaluate the model
# accuracy = rf_model.score(X_test, y_test)
# print(f"Random Forest Model Accuracy: {accuracy}")