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
from Preprocess_dataframe import preprocess_data, roll_sequence, one_hot_encode_activity


if __name__ == "__main__":
    # preprocess dataframe
    data_path = 'data/hospital_billing_2.csv'
    df = pd.read_csv(data_path, sep=';')
    df = df.head(100)
    df_unbalanced = preprocess_data(df)
    df_rolled = roll_sequence(df_unbalanced)
    df_onehotencoded = one_hot_encode_activity(df_rolled)[["label", "feature"]]
    df_onehotencoded["label"] = df_onehotencoded["label"].apply(lambda x: x[0])

    # resample and train data
    kf = StratifiedKFold(n_splits=5)
    results = {type(resampler).__name__: [] for resampler in resampling_techniques}
    X = df_onehotencoded['feature'].tolist()
    y = df_onehotencoded['label'].tolist()

    for resampler in resampling_techniques:
        print(f"Using resampler: {type(resampler).__name__}")

        for train_index, test_index in kf.split(X,y):
            start_time = time.time()

            # Split data
            train_data, test_data = df_onehotencoded.iloc[train_index], df_onehotencoded.iloc[test_index]

            # Resample train_data using the current resampler
            X_resampled, y_resampled = resampler.fit_resample(np.array(train_data['feature'].tolist()),
                                                              train_data['label'])

            print("1")





    # preprocess dataset
    Encoded_data = SequenceDataset(df_onehotencoded)
    X, y, z, k = Encoded_data[:]
    dataloader = DataLoader(Encoded_data, batch_size=32, shuffle=True)

    # Applying resampling techniques and classification
    results = {}
    time_report_all = {}
    for name, resampler in resampling_techniques.items():
        results[name], time_report_all[name] = apply_resampling_and_classification(resampler, dataloader)



    print("1")






















