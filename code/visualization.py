import pandas as pd
import matplotlib.pyplot as plt
import re

# Load your dataset
df = pd.read_csv('hospital_2_reshape.csv')

def extract_means_by_label(dataframe, label_column='label'):
    act_columns = [col for col in dataframe.columns if re.match(r'ACT_\d+_', col)]

    means_label_0 = {}
    means_label_1 = {}

    for col in act_columns:
        # Calculate the mean for label 0
        mean_0 = dataframe[dataframe[label_column] == 0][col].mean()
        means_label_0[col] = mean_0

        # Calculate the mean for label 1
        mean_1 = dataframe[dataframe[label_column] == 1][col].mean()
        means_label_1[col] = mean_1

    return means_label_0, means_label_1

def visualize_means(dataframe, means_label_0, means_label_1, label_column='label'):
    # Identifying unique n values
    n_values = set(int(re.search(r'ACT_(\d+)_', col).group(1)) for col in dataframe.columns if re.match(r'ACT_\d+_', col))

    for n in sorted(n_values):
        # Filtering columns for ACT_{n}_{name}
        act_n_columns = [col for col in dataframe.columns if re.match(fr'ACT_{n}_', col)]

        # Extracting means for label 0 and 1 for each column
        means_0 = [means_label_0[col] for col in act_n_columns]
        means_1 = [means_label_1[col] for col in act_n_columns]

        # Plotting
        plt.figure(figsize=(15, 6))
        x = range(len(act_n_columns))
        plt.bar([i - 0.2 for i in x], means_0, width=0.4, label='Label 0', align='center')
        plt.bar([i + 0.2 for i in x], means_1, width=0.4, label='Label 1', align='center')
        plt.xlabel(f'Event_No.{n}')
        plt.ylabel('Mean Values')
        plt.title(f'Share of activity for event_No.{n} by Label')
        plt.xticks(x, act_n_columns, rotation=90)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Using the function with the dataset
means_0, means_1 = extract_means_by_label(df)

# Displaying the results
means_0, means_1

visualize_means(df, means_0, means_1)