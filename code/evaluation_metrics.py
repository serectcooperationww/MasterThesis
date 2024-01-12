

def calculate_evaluation_metrics(results, time_report_all):
    averaged_results = {}
    summed_timings = {key: 0 for key in time_report_all}
    counts_timings = {key: len(time_report_all[key]) for key in time_report_all}

    for name, reports in results.items():
        # Initialize dictionaries to store summed metrics for each label
        summed_metrics = {
            '0': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},  # Metrics for 'regular' (label 0)
            '1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}   # Metrics for 'deviant' (label 1)
        }

        # Sum up the metrics across all folds
        for report_idx in range(len(reports)):
            for label in ['0', '1']:
                if label in reports[report_idx]:
                    summed_metrics[label]['precision'] += reports[report_idx][str(label)]['precision']
                    summed_metrics[label]['recall'] += reports[report_idx][str(label)]['recall']
                    summed_metrics[label]['f1-score'] += reports[report_idx][str(label)]['f1-score']
                    summed_metrics[label]['support'] += reports[report_idx][str(label)]['support']

        for key in time_report_all:
            summed_timings[key] += time_report_all[key][report_idx]

        # Calculate the average for each metric
        avg_metrics = {}
        num_folds = len(reports)
        for label in ['0', '1']:
            avg_metrics[label] = {k: v / num_folds for k, v in summed_metrics[label].items()}

        # Store in the results dictionary
        averaged_results[name] = avg_metrics
        averages_timings = {key: summed_val / counts for key, summed_val in summed_timings.items() for counts in
                            [counts_timings[key]]}


    # Display the averaged results for metrics and timings
    for method in averaged_results.keys():
        print(f"Method: {method}")
        # Display metrics
        for label, metrics in averaged_results[method].items():
            print(f"Label {label} Metrics:", metrics)

        print(f"Average training time for {method}: {averages_timings[method]}")
        print("\n")