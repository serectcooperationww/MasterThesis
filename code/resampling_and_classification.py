from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from LSTMencoder_pytorch import LSTM, SequenceDataset


resampling_techniques = {
        "Random Over-Sampling": RandomOverSampler(random_state=0),
        "Random Under-Sampling": RandomUnderSampler(random_state=0),
        "SMOTE": SMOTE(random_state=0),
        "ADASYN": ADASYN(random_state=0),
        "NearMiss": NearMiss(version=1),
        "No Filter": None
    }

def apply_resampling_and_classification(resampler, dataloader): #X, y,

    input_size = 4  # Number of features in each sequence
    hidden_size = 50
    num_classes = 2

    model = LSTM(input_size, hidden_size, num_classes)
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10  # Adjust as needed
    for epoch in range(num_epochs):
        for sequences, labels in dataloader:
            # Forward pass
            outputs = model(sequences)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    clf = RandomForestClassifier(random_state=0)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    reports = []
    time_report = []

    for train_index, test_index in skf.split(X, y):
        start_time = time.time()
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if resampler is not None:
            X_resampled, y_resampled = resampler.fit_resample(X_train, y_train)
        else:
            X_resampled, y_resampled = X_train, y_train
        clf.fit(X_resampled, y_resampled)
        y_pred = clf.predict(X_test)

        reports.append(classification_report(y_test, y_pred, output_dict=True))

        end_time = time.time()
        Execution_time = end_time - start_time
        time_report.append(Execution_time)

    return reports, time_report
