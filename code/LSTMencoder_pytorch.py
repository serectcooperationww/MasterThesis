import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

class LSTMencoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMencoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        return hidden[-1]  # Return the last hidden state

    def flatten(self):
        self.lstm.flatten_parameters()

class SequenceDataset(Dataset):
    def __init__(self, dataframe):
        self.current_activities = []
        self.next_activities = []

        # Process the dataframe to generate activity pairs
        for sequence in dataframe['activity_time_onehot']:
            for i in range(len(sequence) - 1):
                self.current_activities.append(sequence[:i + 1])
                self.next_activities.append([sequence[i + 1]])

                # Stop if the next two lists are zero-filled
                if i < len(sequence) - 2 and all(value == 0 for value in sequence[i + 1]) and all(
                        value == 0 for value in sequence[i + 2]):
                    break

    def __len__(self):
        return len(self.current_activities)

    def __getitem__(self, idx):
        current_activity = torch.tensor(self.current_activities[idx], dtype=torch.float)
        next_activity = torch.tensor(self.next_activities[idx], dtype=torch.float)
        return current_activity, next_activity


    # def Encode_data(df):
    #     # Encoding
    #     label_encoder = LabelEncoder()
    #     df['Activity_encoded'] = label_encoder.fit_transform(df['Activity'])
    #
    #     # Normalize 'timesincelastevent'
    #     scaler = MinMaxScaler()
    #     df['timesincelastevent'] = scaler.fit_transform(df[['timesincelastevent']])
    #
    #     # Group by 'case ID' and prepare sequences
    #     grouped = df.groupby('Case ID')
    #
    #     # features = ['Activity_encoded', 'timesincelastevent']
    #     # sequences = [group[features].values.tolist() for _, group in grouped]
    #     # random.shuffle(sequences)
    #     #
    #     # # Pad sequences
    #     # padded_sequences = pad_sequences(sequences, padding='post', dtype='float32')
    #
    #     return grouped
