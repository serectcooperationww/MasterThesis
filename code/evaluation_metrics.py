import pandas as pd


def calculate_evaluation_metrics(results, accuracys, AUCs, time_report_all):
    averaged_results = {}

    summed_accuracy = {key: sum(accuracys[key]) for key in accuracys}
    counts_accuracy = {key: len(accuracys[key]) for key in accuracys}

    summed_AUC = {key: sum(AUCs[key]) for key in AUCs}
    counts_AUC = {key: len(AUCs[key]) for key in AUCs}

    summed_timings = {key: sum(time_report_all[key]) for key in time_report_all}
    counts_timings = {key: len(time_report_all[key]) for key in time_report_all}

    for name, reports in results.items():
        # Initialize dictionaries to store summed metrics for each label
        summed_metrics = {
            '0': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},  # Metrics for 'regular' (label 0)
            '1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},  # Metrics for 'deviant' (label 1)
        }

        # Sum up the metrics across all folds
        for report in reports:
            for label in ['0', '1']:
                if label in report:
                    summed_metrics[label]['precision'] += report[label]['precision']
                    summed_metrics[label]['recall'] += report[label]['recall']
                    summed_metrics[label]['f1-score'] += report[label]['f1-score']
                    summed_metrics[label]['support'] += report[label]['support']

        # Calculate the average for each metric
        num_folds = len(reports)
        avg_metrics = {label: {k: v / num_folds for k, v in summed_metrics[label].items()} for label in ['0', '1']}

        # Store in the results dictionary
        averaged_results[name] = avg_metrics

    # Calculate averages for accuracy, AUC, and timings
    averages_accuracy = {key: summed_val / counts_accuracy[key] for key, summed_val in summed_accuracy.items()}
    averages_AUC = {key: summed_val / counts_AUC[key] for key, summed_val in summed_AUC.items()}
    averages_timings = {key: summed_val / counts_timings[key] for key, summed_val in summed_timings.items()}

    # Display the averaged results for metrics and timings
    for method in averaged_results.keys():
        print(f"Method: {method}")

        for label, metrics in averaged_results[method].items():
            print(f"Label {label} Metrics:", metrics)

        # Display averaged accuracy, AUC, and training time for each method
        print(f"Average accuracy for {method}: {averages_accuracy.get(method, 'N/A')}")
        print(f"Average AUC for {method}: {averages_AUC.get(method, 'N/A')}")
        print(f"Average training time for {method}: {averages_timings.get(method, 'N/A')}")
        print("\n")


def calculate_averaged_results(results, accuracys, AUCs, time_report_all):
    averaged_results = {}
    averages_accuracy = {key: sum(accuracys[key]) / len(accuracys[key]) for key in accuracys}
    averages_AUC = {key: sum(AUCs[key]) / len(AUCs[key]) for key in AUCs}
    averages_timings = {key: sum(time_report_all[key]) / len(time_report_all[key]) for key in time_report_all}
    for name, reports in results.items():
        summed_metrics = {
            '0': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
            '1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
            'accuracy': 0,
            'macro avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},
            'weighted avg': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}
        }
        num_reports = len(reports)
        for report in reports:
            for label in ['0', '1']:
                for metric in summed_metrics[label]:
                    summed_metrics[label][metric] += report[label].get(metric, 0)
            summed_metrics['accuracy'] += report.get('accuracy', 0)
            for avg_type in ['macro avg', 'weighted avg']:
                if avg_type in report:
                    for metric in summed_metrics[avg_type]:
                        summed_metrics[avg_type][metric] += report[avg_type].get(metric, 0)

        avg_metrics = {label: {k: v / num_reports for k, v in summed_metrics[label].items()} for label in ['0', '1']}
        avg_metrics['accuracy'] = summed_metrics['accuracy'] / num_reports
        for avg_type in ['macro avg', 'weighted avg']:
            avg_metrics[avg_type] = {k: v / num_reports for k, v in summed_metrics[avg_type].items()}

        averaged_results[name] = avg_metrics

    return averaged_results, averages_accuracy, averages_AUC, averages_timings


def write_data_to_excel(averaged_results, averages_accuracy, averages_AUC, averages_timings, results, accuracys, AUCs, time_report_all, file_name):
    writer = pd.ExcelWriter(file_name, engine='xlsxwriter')

    # Initialize row counters for each sheet
    row_counters = {
        'Averaged Results': 0,
        'Original Results': 0,
        'Accuracy': 0,
        'AUC': 0,
        'Time Report': 0}

    # Saving averaged results along with averaged metrics
    for method, metrics in averaged_results.items():
        df = pd.DataFrame(metrics)
        if 'Metric' in df.columns:
            df = df.drop('Metric', axis=1)
        df = df.rename_axis('Metric').reset_index()
        df['Method'] = method
        df = df.pivot(index='Method', columns='Metric', values=df.columns[1:])
        df['Average Accuracy'] = averages_accuracy.get(method, float('nan'))
        df['Average AUC'] = averages_AUC.get(method, float('nan'))
        df['Average Training Time'] = averages_timings.get(method, float('nan'))
        df.to_excel(writer, sheet_name='Averaged Results', startrow=row_counters['Averaged Results'], index=True,
                    header=row_counters['Averaged Results'] == 0)
        row_counters['Averaged Results'] += len(df) + 2  # Adding 2 for spacing

    # Saving original 'results' data
    for method, method_results in results.items():
        df_results = pd.DataFrame()
        for report in method_results:
            if isinstance(report, dict):
                for key, value in report.items():
                    if isinstance(value, dict):
                        df_temp = pd.DataFrame(value, index=[0])
                        df_temp['Label'] = key
                        df_temp['Method'] = method
                        df_results = pd.concat([df_results, df_temp], ignore_index=True, sort=False)
                    else:
                        df_results[key] = [value]

        if not df_results.empty:
            df_results = df_results.melt(id_vars=['Method', 'Label'], var_name='Metric', value_name='Value')
            df_results.to_excel(writer, sheet_name='Original Results', startrow=row_counters['Original Results'],
                                index=False, header=row_counters['Original Results'] == 0)
            row_counters['Original Results'] += len(df_results) + 2

    # Saving other original data (accuracys, AUCs, time_report_all)
    original_data = {
        'Accuracy': accuracys,
        'AUC': AUCs,
        'Time Report': time_report_all
    }

    for key, value in original_data.items():
        df = pd.DataFrame.from_dict(value, orient='index').T.melt(var_name='Method', value_name=key)
        df.to_excel(writer, sheet_name=key, startrow=row_counters[key], index=False, header=row_counters[key] == 0)
        row_counters[key] += len(df) + 2

    # Save and close the writer
    writer.save()


import pandas as pd

def create_excel_report(results, accuracys, AUCs, time_report_all, filename='output_metrics.xlsx'):
    """
    Create an Excel report with two sheets based on the provided data.

    :param results: Dictionary containing detailed metrics for each method and trial.
    :param accuracys: Dictionary containing accuracy values for each method.
    :param AUCs: Dictionary containing AUC values for each method.
    :param time_report_all: Dictionary containing training time for each method.
    :param filename: Name of the Excel file to be created.
    """

    # Preparing data for Sheet 1
    data_sheet1 = []
    for method, trials in results.items():
        for idx, trial in enumerate(trials):
            for label, metrics in trial.items():
                if label in ['0', '1']:
                    accuracy = accuracys[method][idx]
                    auc = AUCs[method][idx]
                    time_report = time_report_all[method][idx]
                    data_sheet1.append([method, idx + 1, label] + list(metrics.values()) + [accuracy, auc, time_report])

    # Creating DataFrame for Sheet 1
    df_sheet1 = pd.DataFrame(data_sheet1, columns=['Method', 'Trial', 'Label', 'Precision', 'Recall', 'F1-Score', 'Support', 'Accuracy', 'AUC', 'Training Time'])

    # Preparing data for Sheet 2
    data_sheet2 = []
    for method in results.keys():
        for label in ['0', '1']:
            avg_precision = sum(trial[label]['precision'] for trial in results[method]) / len(results[method])
            avg_recall = sum(trial[label]['recall'] for trial in results[method]) / len(results[method])
            avg_f1_score = sum(trial[label]['f1-score'] for trial in results[method]) / len(results[method])
            avg_support = sum(trial[label]['support'] for trial in results[method]) / len(results[method])
            avg_accuracy = sum(accuracys[method]) / len(accuracys[method])
            avg_auc = sum(AUCs[method]) / len(AUCs[method])
            avg_time = sum(time_report_all[method]) / len(time_report_all[method])

            data_sheet2.append([method, label, avg_precision, avg_recall, avg_f1_score, avg_support, avg_accuracy, avg_auc, avg_time])

    # Creating DataFrame for Sheet 2
    df_sheet2 = pd.DataFrame(data_sheet2, columns=['Method', 'Label', 'Precision', 'Recall', 'F1-Score', 'Support', 'Average Accuracy', 'Average AUC', 'Average Training Time'])

    # Writing to Excel
    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        df_sheet1.to_excel(writer, sheet_name='Original Data', index=False)
        df_sheet2.to_excel(writer, sheet_name='Aggregated Metrics', index=False)

    print(f"Excel file '{filename}' has been created.")


