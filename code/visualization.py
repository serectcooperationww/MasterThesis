import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
import numpy as np
import os
from datetime import datetime
import time


# df = pd.read_csv('hospital_2_reshape.csv')
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
#
# # Using the function with the dataset
# means_0, means_1 = extract_means_by_label(df)
#
# # Displaying the results
# means_0, means_1
#
# visualize_means(df, means_0, means_1)

def create_bar_charts(results, accuracys, AUCs, time_report_all):
    """
    Create separate bar charts for each metric: Precision, Recall, F1-Score, Accuracy, AUC
    for both labels '0' (regular) and '1' (deviant), and a separate chart for mean training time.
    Different background colors are used for original, oversampling, and undersampling methods.
    Bars are ordered as per the original order in resampling_techniques.

    :param results: Dictionary containing detailed metrics for each method and trial.
    :param accuracys: Dictionary containing accuracy values for each method.
    :param AUCs: Dictionary containing AUC values for each method.
    :param time_report_all: Dictionary containing training time for each method.
    :param method_categories: Dictionary categorizing each method as 'original', 'oversampling', or 'undersampling'.
    """

    output_folder = f"D:/SS2023/MasterThesis/code/visulization_plot/{timestr}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define background colors for each category
    bg_colors = {
        'original': 'lightgray',
        'oversampling': 'lightblue',
        'undersampling': 'lightgreen'
    }

    # Define legend patches
    legend_patches = [
        mpatches.Patch(color=bg_colors['original'], label='Original'),
        mpatches.Patch(color=bg_colors['oversampling'], label='Oversampling'),
        mpatches.Patch(color=bg_colors['undersampling'], label='Undersampling')
    ]

    # Preparing data for aggregation
    aggregated_data = []
    for i, method in enumerate(results.keys()):
        category_index = 0 if i == 0 else 1 if i <= 5 else 2
        for label in ['0', '1']:
            avg_precision = sum(trial[label]['precision'] for trial in results[method]) / len(results[method])
            avg_recall = sum(trial[label]['recall'] for trial in results[method]) / len(results[method])
            avg_f1_score = sum(trial[label]['f1-score'] for trial in results[method]) / len(results[method])
            avg_accuracy = sum(accuracys[method]) / len(accuracys[method])
            avg_auc = sum(AUCs[method]) / len(AUCs[method])
            aggregated_data.append(
                [method, label, avg_precision, avg_recall, avg_f1_score, avg_accuracy, avg_auc, category_index])

    # Creating DataFrame
    df = pd.DataFrame(aggregated_data, columns=['Method', 'Label', 'Precision', 'Recall', 'F1-Score', 'Accuracy', 'AUC',
                                                'Category Index'])
    df['Method'] = pd.Categorical(df['Method'], categories=results.keys(), ordered=True)

    # Color palette
    color_palette = ["#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"]

    # Separate charts for each metric
    metrics = ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'AUC']

    for i, metric in enumerate(metrics):
        color = color_palette[i]

        # Plotting charts for label '0' and '1'
        for label_df, label in [(df[df['Label'] == '0'], '0'), (df[df['Label'] == '1'], '1')]:
            label_text = "regular" if label == 0 else "deviant"
            plt.figure(figsize=(12, 6))
            ax = label_df.groupby('Method').mean()[metric].plot(kind='bar', color=color)
            plt.title(f'Mean {metric} for Label "{label_text}" by resampling method')
            plt.ylabel(metric)
            plt.xticks(rotation=45)

            for p in ax.patches:
                ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                            textcoords='offset points')

            # Setting background color based on category index
            for method in label_df['Method'].unique():
                category_index = label_df[label_df['Method'] == method]['Category Index'].iloc[0]
                ax.get_children()[list(results.keys()).index(method)].set_facecolor(
                    bg_colors[list(bg_colors.keys())[category_index]])

            # Adding custom legends outside the plot
            plt.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1, 0.5))
            plt.subplots_adjust(right=0.8)  # Adjust subplot to fit legend
            plt.savefig(os.path.join(output_folder, f"{metric}_Label_{label}.png"))
            plt.close()

    # Mean training time chart
    mean_training_time = {method: sum(times) / len(times) for method, times in time_report_all.items()}
    plt.figure(figsize=(12, 6))
    plt.bar(mean_training_time.keys(), mean_training_time.values(), color="#8bd3c7")
    plt.title('Mean Training Time')
    plt.ylabel('Time (seconds)')
    plt.xticks(rotation=45)
    plt.legend(['Training Time'], loc='upper left', bbox_to_anchor=(1, 0.5))
    plt.subplots_adjust(right=0.8)

    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    plt.savefig(os.path.join(output_folder, "Mean_Training_Time.png"))
    plt.close()

def plot_distribution(dfs, columns_to_plot, resampler_name="Original DataFrame"):
    output_folder = f"D:/SS2023/MasterThesis/code/visulization_plot/{timestr}"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # If a single DataFrame is passed, wrap it in a list
    if isinstance(dfs, pd.DataFrame):
        dfs = [dfs]

    # Check if all DataFrames have the necessary columns
    for df in dfs:
        for col in columns_to_plot:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in DataFrame.")
        if "label" not in df.columns:
            raise ValueError("Label column not found in DataFrame.")

    # Calculating average DataFrame if multiple DataFrames are provided
    if len(dfs) > 1:
        avg_df = pd.concat(dfs).groupby(level=0).mean()
    else:
        avg_df = dfs[0]

    # Initialize the plot
    plt.figure(figsize=(15, 6))
    bar_width = 0.35
    x = np.arange(len(columns_to_plot))

    # Plotting
    for i, label_value in enumerate([0, 1]):
        proportions = [
            avg_df[col][avg_df["label"] == label_value].value_counts(normalize=True).get(1, 0)
            for col in columns_to_plot
        ]
        plt.bar(x + i * bar_width, proportions, width=bar_width, label=f'Label {label_value}')

    # Formatting the plot
    plt.xlabel('Column Names')
    plt.ylabel('Share of activities happening')
    plt.title(f'Share of activities by label - {resampler_name}')
    plt.xticks(x + bar_width / 2, columns_to_plot, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{resampler_name}_activity_distribution.png"))
    plt.close()



