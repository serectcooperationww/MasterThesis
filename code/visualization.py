import pandas as pd
import matplotlib.pyplot as plt
import re

# Load your dataset
df = pd.read_csv('hospital_2_reshape.csv')
#
# def extract_means_by_label(dataframe, label_column='label'):
#     act_columns = [col for col in dataframe.columns if re.match(r'ACT_\d+_', col)]
#
#     means_label_0 = {}
#     means_label_1 = {}
#
#     for col in act_columns:
#         # Calculate the mean for label 0
#         mean_0 = dataframe[dataframe[label_column] == 0][col].mean()
#         means_label_0[col] = mean_0
#
#         # Calculate the mean for label 1
#         mean_1 = dataframe[dataframe[label_column] == 1][col].mean()
#         means_label_1[col] = mean_1
#
#     return means_label_0, means_label_1
#
# def visualize_means(dataframe, means_label_0, means_label_1, label_column='label'):
#     # Identifying unique n values
#     n_values = set(int(re.search(r'ACT_(\d+)_', col).group(1)) for col in dataframe.columns if re.match(r'ACT_\d+_', col))
#
#     for n in sorted(n_values):
#         # Filtering columns for ACT_{n}_{name}
#         act_n_columns = [col for col in dataframe.columns if re.match(fr'ACT_{n}_', col)]
#
#         # Extracting means for label 0 and 1 for each column
#         means_0 = [means_label_0[col] for col in act_n_columns]
#         means_1 = [means_label_1[col] for col in act_n_columns]
#
#         # Plotting
#         plt.figure(figsize=(15, 6))
#         x = range(len(act_n_columns))
#         plt.bar([i - 0.2 for i in x], means_0, width=0.4, label='Label 0', align='center')
#         plt.bar([i + 0.2 for i in x], means_1, width=0.4, label='Label 1', align='center')
#         plt.xlabel(f'Event_No.{n}')
#         plt.ylabel('Mean Values')
#         plt.title(f'Share of activity for event_No.{n} by Label')
#         plt.xticks(x, act_n_columns, rotation=90)
#         plt.legend()
#         plt.tight_layout()
#         plt.show()



def create_bar_charts(results, accuracys, AUCs, time_report_all):
    """
    Create separate bar charts with adjusted figure size, legends, and colors.

    :param results: Dictionary containing detailed metrics for each method and trial.
    :param accuracys: Dictionary containing accuracy values for each method.
    :param AUCs: Dictionary containing AUC values for each method.
    :param time_report_all: Dictionary containing training time for each method.
    """

    # Preparing data for aggregation
    aggregated_data = []
    for method in results.keys():
        for label in ['0', '1']:
            avg_precision = sum(trial[label]['precision'] for trial in results[method]) / len(results[method])
            avg_recall = sum(trial[label]['recall'] for trial in results[method]) / len(results[method])
            avg_f1_score = sum(trial[label]['f1-score'] for trial in results[method]) / len(results[method])
            avg_accuracy = sum(accuracys[method]) / len(accuracys[method])
            avg_auc = sum(AUCs[method]) / len(AUCs[method])
            aggregated_data.append([method, label, avg_precision, avg_recall, avg_f1_score, avg_accuracy, avg_auc])

    # Creating DataFrame
    df = pd.DataFrame(aggregated_data,
                      columns=['Method', 'Label', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'AUC'])

    # Separate data for label '0' and '1'
    df_label_0 = df[df['Label'] == '0']
    df_label_1 = df[df['Label'] == '1']

    # Mean training time
    mean_training_time = {method: sum(times) / len(times) for method, times in time_report_all.items()}

    # Academic color palette
    color_palette = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"]
    bar_width = 0.8

    # Chart for label '0'
    plt.figure(figsize=(16, 4))
    ax = df_label_0.set_index('Method')[['Precision', 'Recall', 'F1-Score', 'Accuracy', 'AUC']].plot(kind='bar',
                                                                                                     width=bar_width,
                                                                                                     color=color_palette)
    plt.title('Mean Metrics for Label "regular"')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.subplots_adjust(bottom=0.2, right=0.8)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.style.use('seaborn-dark-palette')
    plt.show()

    # Chart for label '1'
    plt.figure(figsize=(16, 4))
    ax = df_label_1.set_index('Method')[['Precision', 'Recall', 'F1-Score', 'Accuracy', 'AUC']].plot(kind='bar',
                                                                                                     width=bar_width,
                                                                                                     color=color_palette)
    plt.title('Mean Metrics for Label "deviant"')
    plt.ylabel('Score')
    plt.xticks(rotation=0)
    plt.subplots_adjust(bottom=0.2, right=0.8)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    # Chart for mean training time
    plt.figure(figsize=(12, 6))
    plt.bar(mean_training_time.keys(), mean_training_time.values(), color=color_palette[1])
    plt.title('Mean Training Time')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=0)
    plt.subplots_adjust(bottom=0.2, right=0.8)
    plt.legend(['Training Time'], loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

#
# # Using the function with the dataset
# means_0, means_1 = extract_means_by_label(df)
#
# # Displaying the results
# means_0, means_1
#
# visualize_means(df, means_0, means_1)