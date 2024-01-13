import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
import logging
from datetime import datetime

from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import OneHotEncoder

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss

from LSTMencoder_pytorch import LSTM, SequenceDataset, train_model, evaluate_model
from resampling_and_classification import resampling_techniques
from Preprocess_dataframe import preprocess_data_hospital, preprocess_data_BPIC15, roll_sequence, one_hot_encode_activity, reshape_case, flatten_feature
from evaluation_metrics import calculate_evaluation_metrics


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')

    # preprocess dataframe
    data_path = 'data/sepsis_cases_2.csv'
    df = pd.read_csv(data_path, sep=';')

    ohe = OneHotEncoder(sparse=False)
    activities_encoded = ohe.fit_transform(df[['Activity']])
    activities_df = pd.DataFrame(activities_encoded, columns=[f'ACT_{col.split("_")[-1]}' for col in ohe.get_feature_names_out(['Activity'])])

    encoded_df = pd.concat([df[['Case ID', 'timesincelastevent']], activities_df], axis=1)
    scaler = MinMaxScaler()
    encoded_df['timesincelastevent'] = scaler.fit_transform(encoded_df[['timesincelastevent']])

    # Sorting the data by Case ID and timestamp to ensure correct sequence
    encoded_df = encoded_df.join(df['time:timestamp'])
    encoded_df.sort_values(by=['Case ID', 'time:timestamp'], inplace=True)
    encoded_df.drop(columns=['time:timestamp'], inplace=True)

    reshaped_data = encoded_df.groupby('Case ID').apply(reshape_case)
    reshaped_data = reshaped_data.reset_index(level=1, drop=True)
    reshaped_data = reshaped_data.drop("missing_caseid")

    col_position = reshaped_data.columns.get_loc("timesincelastevent_14")
    trunc_df = reshaped_data.iloc[:, :col_position]

    unique_case_ids = trunc_df.index.unique()
    case_id_to_label = df.drop_duplicates(subset='Case ID').set_index('Case ID')['label']
    labels_for_trunc_df = unique_case_ids.map(case_id_to_label)
    trunc_df['label'] = labels_for_trunc_df
    trunc_df['label'] = trunc_df['label'].map({'regular': 0, 'deviant': 1})

    trunc_df.fillna(0.0, inplace=True)

    logging.info(f"Dataframe preprocessed. ")

    # resample and train data
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    results = {}
    time_report_all = {}
    X = trunc_df.drop('label', axis=1)
    y = trunc_df['label']

    torch_device = "cpu"
    device_package = torch.cpu
    if torch.cuda.is_available():
        torch_device = torch.device("cuda")
        device_package = torch.cuda

    for name, resampler in resampling_techniques.items():
        logging.info(f"------ Using resampler: {name} ------")
        reports = []
        time_report = []

        for train_index, test_index in kf.split(X,y):

            # resample data
            start_time = time.time()

            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]

            if resampler is not None:
                X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
            else:
                X_resampled, y_resampled = X_train, y_train

            logging.info(f"Resampling done with {name}")

            # train model
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_resampled, y_resampled)

            end_time = time.time()
            execution_time = end_time - start_time
            time_report.append(execution_time)
            logging.info("Training done")

            # evaluate model
            y_pred = clf.predict(X_test)
            reports.append(classification_report(y_test, y_pred, output_dict=True))

        results[name], time_report_all[name] = reports, time_report

    calculate_evaluation_metrics(results, time_report_all)
    print(results)
















