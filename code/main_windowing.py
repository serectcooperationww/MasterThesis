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

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss

from LSTMencoder_pytorch import LSTM, SequenceDataset, train_model, evaluate_model
from resampling_and_classification import resampling_techniques
from Preprocess_dataframe import preprocess_data_hospital, preprocess_data_BPIC15, roll_sequence, one_hot_encode_activity, flatten_feature, create_windows
from evaluation_metrics import calculate_evaluation_metrics


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s:%(message)s')

    # preprocess dataframe
    data_path = 'data/sepsis_cases_2.csv'
    df = pd.read_csv(data_path, sep=';')
    df_unbalanced = preprocess_data_BPIC15(df)
    df_rolled = roll_sequence(df_unbalanced)
    df_onehotencoded = one_hot_encode_activity(df_rolled)[["label", "feature"]]
    df_onehotencoded.loc[:, 'label'] = [x[0] for x in df_onehotencoded['label']]

    logging.info(f"Dataframe preprocessed. Number of Case: {df_onehotencoded.shape[0]}")
    logging.info(f"Length of each feature: {len(df_onehotencoded.feature[0][0])}")
    logging.info(f"longest trace: {len(df_onehotencoded.feature[0])}")

    # resample and train data
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    results = {}
    time_report_all = {}
    X = df_onehotencoded.drop('label', axis=1)
    y = df_onehotencoded['label']

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

            # prepare window
            train_df_resampled = pd.concat([X_resampled, y_resampled], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)

            window_size = 3
            X_train_window, y_train_window = create_windows(train_df_resampled['feature'], train_df_resampled['label'], window_size)
            X_test_window, y_test_window = create_windows(test_df['feature'], test_df['label'], window_size)

            X_train_window_flatten = np.array([np.array(window).flatten() for window in X_train_window])
            X_test_window_flatten = np.array([np.array(window).flatten() for window in X_test_window])

            # train model
            clf = RandomForestClassifier(n_estimators=100, random_state=42)
            clf.fit(X_train_window_flatten, y_train_window)

            end_time = time.time()
            execution_time = end_time - start_time
            time_report.append(execution_time)
            logging.info("Training done")

            y_pred_window = clf.predict(X_test_window_flatten)
            reports.append(classification_report(y_test_window, y_pred_window, output_dict=True))

        results[name], time_report_all[name] = reports, time_report

    calculate_evaluation_metrics(results, time_report_all)
    print(results)
















