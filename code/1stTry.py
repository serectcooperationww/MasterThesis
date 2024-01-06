import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Preprocessing Function
def preprocess_data(df):
    df['Complete Timestamp'] = pd.to_datetime(df['Complete Timestamp'])
    df['timesincelastevent'] = df['timesincelastevent'].astype(float)
    df['label'].replace('regular', '0', inplace=True)
    df['label'].replace('deviant', '1', inplace=True)
    df['label'] = df['label'].astype(float)
    df_sorted = df.sort_values(by=['Case ID', 'Complete Timestamp'])

    label_encoder = LabelEncoder()
    df_sorted['Activity_encoded'] = label_encoder.fit_transform(df_sorted['Activity'])
    scaler = MinMaxScaler()
    df_sorted['timesincelastevent'] = scaler.fit_transform(df_sorted[['timesincelastevent']])
    grouped = df_sorted.groupby('Case ID')

    features = ['Activity_encoded', 'timesincelastevent']
    sequences = [group[features].values.tolist() for _, group in grouped]
    labels = df_sorted.groupby('Case ID')['label'].first().values
    padded_sequences = pad_sequences(sequences, padding='post', dtype='float32')
    return padded_sequences, labels

# LSTM Encoding Function
def lstm_encode(X_train, y_train, X_test):
    n_features = X_train.shape[2]
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(None, n_features)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_train, y_train, epochs=10, batch_size=32)
    X_train_encoded = model.predict(X_train)
    X_test_encoded = model.predict(X_test)
    return X_train_encoded, X_test_encoded

# Random Forest Training and Evaluation Function
def train_and_evaluate_rf(X_train, y_train, X_test, y_test):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report


df = pd.read_csv('data/hospital_billing_2.csv', sep=';')
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_encoded, X_test_encoded = lstm_encode(X_train, y_train, X_test)
accuracy_imbalanced, report_imbalanced = train_and_evaluate_rf(X_train_encoded, y_train, X_test_encoded, y_test)

# Downsample the majority class for balanced dataset
df_majority = df[df['label'] == 'regular']
df_minority = df[df['label'] == 'deviant']
df_majority_downsampled = df_majority.sample(len(df_minority), random_state=42)
df_balanced = pd.concat([df_majority_downsampled, df_minority])

# Preprocess the balanced dataset
X_balanced, y_balanced = preprocess_data(df_balanced)
X_train_balanced, X_test_balanced, y_train_balanced, y_test_balanced = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Encode and train on balanced data
X_train_balanced_encoded, X_test_balanced_encoded = lstm_encode(X_train_balanced, y_train_balanced, X_test_balanced)
accuracy_balanced, report_balanced = train_and_evaluate_rf(X_train_balanced_encoded, y_train_balanced, X_test_balanced_encoded, y_test_balanced)

# Print results
print("Imbalanced Data Accuracy:", accuracy_imbalanced)
print("Imbalanced Data Report:\n", report_imbalanced)
print("Balanced Data Accuracy:", accuracy_balanced)
print("Balanced Data Report:\n", report_balanced)
