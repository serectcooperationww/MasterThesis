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
from AggregateTransformer import AggregateTransformer
from StaticTransformer import StaticTransformer


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


def encoding(df, encoding_method="agg", dataset="sepsis"):

    # Aggregation encoding
    if dataset == "sepsis":
        dynamic_cat_cols = ["Activity", 'org:group']
        static_cat_cols = ['Diagnose', 'DiagnosticArtAstrup', 'DiagnosticBlood', 'DiagnosticECG',
                           'DiagnosticIC', 'DiagnosticLacticAcid', 'DiagnosticLiquor',
                           'DiagnosticOther', 'DiagnosticSputum', 'DiagnosticUrinaryCulture',
                           'DiagnosticUrinarySediment', 'DiagnosticXthorax', 'DisfuncOrg',
                           'Hypotensie', 'Hypoxie', 'InfectionSuspected', 'Infusion', 'Oligurie',
                           'SIRSCritHeartRate', 'SIRSCritLeucos', 'SIRSCritTachypnea',
                           'SIRSCritTemperature',
                           'SIRSCriteria2OrMore']
        dynamic_num_cols = ['CRP', 'LacticAcid', 'Leucocytes']
        static_num_cols = ['Age']

        cat_cols = dynamic_cat_cols + static_cat_cols
        num_cols = dynamic_num_cols + static_num_cols

    if encoding_method == "agg":
        transformer = AggregateTransformer(case_id_col='Case ID', cat_cols=cat_cols, num_cols=num_cols, boolean=True,
                                           fillna=True)
    if encoding_method == "static":
        transformer = StaticTransformer(case_id_col='Case ID', cat_cols=cat_cols, num_cols=num_cols,
                                           fillna=True)

    transformer.fit(df)
    transformed_df = transformer.transform(df)

    return transformed_df

def add_label(original_df, transformed_df):
    unique_case_ids = transformed_df.index.unique()
    case_id_to_label = original_df.drop_duplicates(subset='Case ID').set_index('Case ID')['label']
    labels_for_trunc_df = unique_case_ids.map(case_id_to_label)
    transformed_df['label'] = labels_for_trunc_df
    transformed_df['label'] = transformed_df['label'].map({'regular': 0, 'deviant': 1})
    return transformed_df