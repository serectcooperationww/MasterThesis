import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.hidden_size = hidden_size
        self.log_softmax = nn.LogSoftmax(dim=1)  # Add LogSoftmax layer

    def forward(self, x):
        # print(x.shape)
        # print(x)
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)

        return self.log_softmax(out)

    def flatten(self):
        self.lstm.flatten_parameters()


def train_model(train_loader, model, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for sequences, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")




def evaluate_model(test_loader, model):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for sequences, labels in test_loader:
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.tolist())
            all_labels.extend(labels.tolist())
    return classification_report(all_labels, all_preds, output_dict=True)


class SequenceDataset(Dataset):
    def __init__(self, dataframe):
        self.all_activity = dataframe['feature'].values.tolist()
        self.all_label = dataframe['label'].values.tolist()

        self.current_activities = []
        self.current_label = []

        # Process the dataframe to generate activity pairs
        for x, sequence in enumerate(dataframe['feature']):
            for i in range(len(sequence)):
                self.current_activities.append(sequence[:i + 1])
                self.current_label.append(dataframe["label"][x])
                # Stop if the next two lists are zero-filled
                if i < len(sequence) - 2 and all(value == 0 for value in sequence[i + 1]) and all(
                        value == 0 for value in sequence[i + 2]):
                    break

        self.current_activities = np.array(self.current_activities, dtype=object).reshape(-1)
        self.current_label = np.array(self.current_label).reshape((-1))

    def __len__(self):
        return len(self.all_activity)

    def __getitem__(self, idx):
        # pad current activity
        max_length = max(len(inner_list) for inner_list in self.current_activities)
        padded_activities = []
        for inner_list in self.current_activities:
            number_of_zero = len(max(max(lst) for lst in self.current_activities))
            padding_length = max_length - len(inner_list)
            padding = [0] * number_of_zero
            padded_list = inner_list + [padding] * padding_length
            padded_activities.append(padded_list)

        padded_current_activity_tensor = pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in padded_activities], batch_first=True)[idx]
        current_labels_tensor = torch.tensor(self.current_label[idx], dtype=torch.long)

        all_activity_tensor = pad_sequence([torch.tensor(seq, dtype=torch.float32) for seq in self.all_activity], batch_first=True)[idx]
        all_labels_tensor = torch.tensor(self.all_label[idx], dtype=torch.long)

        return padded_current_activity_tensor, current_labels_tensor