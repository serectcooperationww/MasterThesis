

def calculate_evaluation_metrics(results):
    for name, reports in results.items():
        # Initialize dictionaries to store summed metrics for each label
        summed_metrics = {
            '0': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0},  # Metrics for 'regular' (label 0)
            '1': {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}   # Metrics for 'deviant' (label 1)
        }

        # Sum up the metrics across all folds
        for report in reports:
            for label in ['0', '1']:
                summed_metrics[label]['precision'] += report[str(label)]['precision']
                summed_metrics[label]['recall'] += report[str(label)]['recall']
                summed_metrics[label]['f1-score'] += report[str(label)]['f1-score']
                summed_metrics[label]['support'] += report[str(label)]['support']

        # Calculate the average for each metric
        avg_metrics = {}
        num_folds = len(reports)
        for label in ['0', '1']:
            avg_metrics[label] = {k: v / num_folds for k, v in summed_metrics[label].items()}

        # Store in the results dictionary
        averaged_results[name] = avg_metrics

    # Display the averaged results
    for method, avg_metrics in averaged_results.items():
        print(f"Method: {method}")
        for label, metrics in avg_metrics.items():
            print(f"Label {label} Metrics:", metrics)
        print("\n")