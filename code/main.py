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

from LSTMencoder_pytorch import LSTM, SequenceDataset, train_model
from resampling_and_classification import resampling_techniques
from Preprocess_dataframe import preprocess_data, roll_sequence, one_hot_encode_activity, flatten_feature


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
    results = {}
    X = df_onehotencoded.drop('label', axis=1)
    y = df_onehotencoded['label']

    for name, resampler in resampling_techniques.items():
        print(f"Using resampler: {name}")

        for train_index, test_index in kf.split(X,y):
            start_time = time.time()

            X_train, y_train = X.iloc[train_index], y.iloc[train_index]
            X_test, y_test = X.iloc[test_index], y.iloc[test_index]

            X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
            df_resampled = pd.concat([X_resampled, y_resampled], axis=1)

            Encoded_data = SequenceDataset(df_resampled)
            padded_current_activity_tensor, current_labels_tensor = Encoded_data[:]
            dataloader = DataLoader(Encoded_data, batch_size=32, shuffle=True)

            input_size = 6  # Number of features in each sequence
            hidden_size = 50
            num_classes = 2

            model = LSTM(input_size, hidden_size, num_classes)
            criterion = nn.NLLLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            train_model(dataloader, model, criterion, optimizer, num_epochs=10)




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






















