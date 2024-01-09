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
from resampling_and_classification import resampling_techniques





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

    df['feature'] = df.apply(lambda row:
                              [encoded_list + [row['timesincelastevent'][i]] for i, encoded_list in enumerate(
                                  [[int(i == val) for i in range(max_val + 1)] for val in row['Activity']]
                              )] + [[0] * (max_val + 1) + [0]] * (max_length - len(row['Activity'])),
                              axis=1
                              )

    return df

def flatten_feature(df):
    expanded_features = df['feature'].apply(pd.Series)
    expanded_features = expanded_features.rename(columns=lambda x: 'activity_' + str(x))
    new_df = pd.concat([df[['label']], expanded_features], axis=1)
    return new_df