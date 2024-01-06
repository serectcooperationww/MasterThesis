import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Set initial hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size)

        # Only take the output from the final timetep
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def flatten(self):
        self.lstm.flatten_parameters()



class SequenceDataset(Dataset):
    def __init__(self, dataframe):
        self.current_activities = []
        self.next_activities = []
        self.label = []

        # Process the dataframe to generate activity pairs
        for x, sequence in enumerate(dataframe['activity_time_onehot']):
            for i in range(len(sequence) - 1):
                self.current_activities.append(sequence[:i + 1])
                self.next_activities.append([sequence[i + 1]])
                self.label.append(dataframe["label"][x][0])

                # Stop if the next two lists are zero-filled
                if i < len(sequence) - 2 and all(value == 0 for value in sequence[i + 1]) and all(
                        value == 0 for value in sequence[i + 2]):
                    break
        self.next_activities = [inner_list[:-1] for inner_list in self.next_activities]

    def __len__(self):
        return len(self.current_activities)

    def __getitem__(self, idx):
        current_activity = torch.tensor(self.current_activities[idx], dtype=torch.float)
        label = torch.tensor(self.label[idx], dtype=torch.float)
        return current_activity, label