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
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MultiLabelBinarizer, OneHotEncoder, MinMaxScaler
from sklearn.metrics import classification_report

from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss

from LSTMencoder_pytorch import LSTM, SequenceDataset
from resampling_and_classification import resampling_techniques


def preprocess_data(df, time_column):
    df['label'] = df['label'].map({'regular': 0, 'deviant': 1})

    ohe = OneHotEncoder(sparse=False)
    activities_encoded = ohe.fit_transform(df[['Activity']])
    activities_df = pd.DataFrame(activities_encoded, columns=[f'ACT_{col.split("_")[-1]}' for col in ohe.get_feature_names_out(['Activity'])])

    encoded_df = pd.concat([df[['Case ID', 'timesincelastevent']], activities_df], axis=1)
    scaler = MinMaxScaler()
    encoded_df['timesincelastevent'] = scaler.fit_transform(encoded_df[['timesincelastevent']])

    # Sorting the data by Case ID and timestamp to ensure correct sequence
    encoded_df = encoded_df.join(df[time_column])
    encoded_df.sort_values(by=['Case ID', time_column], inplace=True)
    encoded_df.drop(columns=[time_column], inplace=True)

    return encoded_df


def reshape_case(df):
    # Creating column names
    col_names = []
    for i in range(len(df)):
        col_names.append(f'timesincelastevent_{i+1}')
        col_names.extend([f'ACT_{i+1}_{col}' for col in df.columns[2:]])

    # Reshape the dataframe
    reshaped_values = []
    for _, row in df.iterrows():
        reshaped_values.extend(row[1:].tolist())  # Append activity columns and timesincelastevent for each event

    reshaped_df = pd.DataFrame([reshaped_values], columns=col_names)

    return reshaped_df


def prefix_selection(df, n):
    filtered_df = df.groupby("Case ID").filter(lambda x: len(x) >= n)
    return filtered_df.groupby("Case ID").apply(lambda x: x.head(n)).reset_index(drop=True)
