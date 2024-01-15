import pandas as pd


def transform_data(data):
    structured_data = {}
    for method, values in data.items():
        structured_data[method] = {}
        label_0_metrics = values["Label 0 Metrics"]
        label_1_metrics = values["Label 1 Metrics"]
        average_training_time = values["Average training time for " + method]

        structured_data[method]["Label 0 Metrics"] = label_0_metrics
        structured_data[method]["Label 1 Metrics"] = label_1_metrics
        structured_data[method]["Average Training Time"] = average_training_time

    return structured_data

# Data provided
data = {
    "Random Over-Sampling (RO)": {
        "Label 0 Metrics": {'precision': 0.9730430256236666, 'recall': 0.9726463699075689, 'f1-score': 0.9728296641425261, 'support': 1557.4},
        "Label 1 Metrics": {'precision': 0.6818157306615215, 'recall': 0.6823194349510139, 'f1-score': 0.6806290073193375, 'support': 132.2},
        "Average Training Time": 1.1627975940704345
    },
    "All k-nearest neighbours Under-Sampling (AL)": {
        "Label 0 Metrics": {'precision': 0.9756865395790637, 'recall': 0.9632720835878879, 'f1-score': 0.9694314376339073, 'support': 1557.4},
        "Label 1 Metrics": {'precision': 0.6246761772777913, 'recall': 0.7171565276828435, 'f1-score': 0.6671592254096722, 'support': 132.2},
        "Average Training Time": 8.102190732955933
    },
    "Condensed nearest neighbours Under-Sampling (CN)": {
        "Label 0 Metrics": {'precision': 0.9675303791968318, 'recall': 0.9373319218437087, 'f1-score': 0.9521800905632901, 'support': 1557.4},
        "Label 1 Metrics": {'precision': 0.4611857180934864, 'recall': 0.6293688767372978, 'f1-score': 0.5318640679482274, 'support': 132.2},
        "Average Training Time": 464.35903706550596
    },
    "Edited nearest neighbours Under-Sampling (EN)": {
        "Label 0 Metrics": {'precision': 0.9736709203397191, 'recall': 0.968408108480233, 'f1-score': 0.9710181050749975, 'support': 1557.4},
        "Label 1 Metrics": {'precision': 0.6531837195559254, 'recall': 0.6914217361585783, 'f1-score': 0.6705768528516965, 'support': 132.2},
        "Average Training Time": 3.8963891506195067
    },
    "Repeated edited nearest neighbours Under-Sampling": {
        "Label 0 Metrics": {'precision': 0.9774761017901111, 'recall': 0.9581355640145996, 'f1-score': 0.9677008944218551, 'support': 1557.4},
        "Label 1 Metrics": {'precision': 0.6008999136617252, 'recall': 0.7398268398268399, 'f1-score': 0.6626571294946174, 'support': 132.2},
        "Average Training Time": 19.539843702316283
    },
    "Synthetic minority oversampling-Edited nearest neighbour Hybrid-Sampling (SE)": {
        "Label 0 Metrics": {'precision': 0.9797739199705993, 'recall': 0.9327092933235388, 'f1-score': 0.9556534921777867, 'support': 1557.4},
        "Label 1 Metrics": {'precision': 0.4942113551562396, 'recall': 0.7731032125768967, 'f1-score': 0.6026895084236453, 'support': 132.2},
        "Average Training Time": 12.161863708496094
    },
    "Tomek links Under-Sampling (TM)": {
        "Label 0 Metrics": {'precision': 0.9664726535452806, 'recall': 0.984333042296045, 'f1-score': 0.9753142869868208, 'support': 1557.4},
        "Label 1 Metrics": {'precision': 0.7659637326019342, 'recall': 0.5975620870357712, 'f1-score': 0.6702882716758534, 'support': 132.2},
        "Average Training Time": 3.6652042865753174
    },
    "Synthetic minority oversampling-Tomek’s link Over-Sampling (ST)": {
        "Label 0 Metrics": {'precision': 0.9729157731552048, 'recall': 0.9589066891581602, 'f1-score': 0.9658429395298264, 'support': 1557.4},
        "Label 1 Metrics": {'precision': 0.5884722810882319, 'recall': 0.6853953064479381, 'f1-score': 0.6320476422837775, 'support': 132.2},
        "Average Training Time": 9.078138875961304
    },
    "Instance hardness threshold Under-Sampling (IH)": {
        "Label 0 Metrics": {'precision': 0.9896371174285268, 'recall': 0.6872981598693382, 'f1-score': 0.8109527919761966, 'support': 1557.4},
        "Label 1 Metrics": {'precision': 0.19988059001684316, 'recall': 0.9153337890179996, 'f1-score': 0.3278912732139637, 'support': 132.2},
        "Average Training Time": 9.349972438812255
    },
"Neighbourhood cleaning rule Hybrid-Sampling (NC)": {
        "Label 0 Metrics": {'precision': 0.9755284815298657, 'recall': 0.9672530284779575, 'f1-score': 0.9713660636983891, 'support': 1557.4},
        "Label 1 Metrics": {'precision': 0.6504854821756231, 'recall': 0.7141148325358851, 'f1-score': 0.680266429762042, 'support': 132.2},
        "Average Training Time": 4.797392749786377
    },
    "One-sided selection (OS)": {
        "Label 0 Metrics": {'precision': 0.9672185851776309, 'recall': 0.9849748083729697, 'f1-score': 0.9760092045413579, 'support': 1557.4},
        "Label 1 Metrics": {'precision': 0.7776512444093088, 'recall': 0.6066757803599909, 'f1-score': 0.680517373985456, 'support': 132.2},
        "Average Training Time": 4.137778949737549
    },
    "ADASYN": {
        "Label 0 Metrics": {'precision': 0.9735807867253167, 'recall': 0.9508156052050328, 'f1-score': 0.9620534170274846, 'support': 1557.4},
        "Label 1 Metrics": {'precision': 0.5466666262475369, 'recall': 0.6959671907040328, 'f1-score': 0.6117792790261255, 'support': 132.2},
        "Average Training Time": 1.9414052009582519
    },
    "NearMiss": {
        "Label 0 Metrics": {'precision': 0.9729781007385393, 'recall': 0.21523419432551488, 'f1-score': 0.35181868290204626, 'support': 1557.4},
        "Label 1 Metrics": {'precision': 0.09134850260459668, 'recall': 0.9289359763043974, 'f1-score': 0.16632460001273094, 'support': 132.2},
        "Average Training Time": 0.9893615245819092
    },
    "No Filter": {
        "Label 0 Metrics": {'precision': 0.9638695005587149, 'recall': 0.9865159044045567, 'f1-score': 0.9750573648317935, 'support': 1557.4},
        "Label 1 Metrics": {'precision': 0.7822413453532043, 'recall': 0.5643084985190249, 'f1-score': 0.654935805964638, 'support': 132.2},
        "Average Training Time": 0.8329752922058106
    }
}

# Creating a list to hold the formatted data
formatted_data = []

# Iterating over each method and its details
for method, details in data.items():
    for label, metrics in details.items():
        if label != "Average Training Time":
            formatted_data.append({
                "Method": method,
                "Label": label,
                "Precision": metrics['precision'],
                "Recall": metrics['recall'],
                "F1-Score": metrics['f1-score'],
                "Support": metrics['support'],
                "Average Training Time": details["Average Training Time"]
            })

# Creating a DataFrame
df = pd.DataFrame(formatted_data)

# Saving the DataFrame to an Excel file
excel_file_path = "sampling_methods_analysis.xlsx"
df.to_excel(excel_file_path, index=False)