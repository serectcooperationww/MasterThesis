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

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, MultiLabelBinarizer, OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,  roc_auc_score, accuracy_score

from LSTMencoder_pytorch import LSTM, SequenceDataset, train_model, evaluate_model
from resampling_and_classification import resampling_techniques
from Preprocess_dataframe import preprocess_data, reshape_case, prefix_selection
from evaluation_metrics import calculate_averaged_results, write_data_to_excel, create_excel_report
from visualization import create_bar_charts


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
    logging.info(f"Preprocessing starts.")

    # preprocess dataframe
    data_path = 'data/hospital_billing_2.csv'
    df = pd.read_csv(data_path, sep=';')
    preprocessed_df = preprocess_data(df, time_column = "Complete Timestamp")

    # Prefix selection
    n = 7
    encoded_df = prefix_selection(preprocessed_df, n)

    # one hot encoding
    reshaped_data = encoded_df.groupby('Case ID').apply(reshape_case)
    reshaped_data = reshaped_data.reset_index(level=1, drop=True)

    # add label to each case
    unique_case_ids = reshaped_data.index.unique()
    case_id_to_label = df.drop_duplicates(subset='Case ID').set_index('Case ID')['label']
    labels_for_trunc_df = unique_case_ids.map(case_id_to_label)
    reshaped_data['label'] = labels_for_trunc_df

    logging.info(f"Dataframe preprocessed. ")

    # resample and train data
    kf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    results = {}
    accuracys ={}
    AUCs = {}
    time_report_all = {}
    X = reshaped_data.drop('label', axis=1)
    y = reshaped_data['label']

    torch_device = "cpu"
    device_package = torch.cpu
    if torch.cuda.is_available():
        torch_device = torch.device("cuda")
        device_package = torch.cuda

    for name, resampler in resampling_techniques.items():
        logging.info(f"------ Using resampler: {name} ------")
        reports = []
        accuracy = []
        AUC = []
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
            model = XGBClassifier(random_state = 0)
            model.fit(X_resampled, y_resampled)

            end_time = time.time()
            execution_time = end_time - start_time
            time_report.append(execution_time)
            logging.info("Training done")

            # evaluate model
            y_pred = model.predict(X_test)
            roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

            reports.append(classification_report(y_test, y_pred, output_dict=True))
            accuracy.append(accuracy_score(y_test, y_pred))
            AUC.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

        results[name], accuracys[name], AUCs[name], time_report_all[name] = reports, accuracy, AUC, time_report

    create_excel_report(results, accuracys, AUCs, time_report_all, 'my_report.xlsx')
    create_bar_charts(results, accuracys, AUCs, time_report_all)














