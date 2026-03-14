import pandas as pd
import argparse
import ast
import os
import matplotlib.pyplot as plt
import numpy as np

LABELS = ["LIN", "SI", "CL", "D", "HI", "PL", "TI", "PC"]

def plot_confusion_matrix(tp, fp, fn, tn, title, save_path):
    cm = np.array([[tn, fp],
                   [fn, tp]])
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ['Negative', 'Positive']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label',
           title=title)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

def compute_binary_metrics(tp, fp, fn, tn):
    pos_precision = tp / (tp + fp) if (tp + fp) else 0
    pos_recall = tp / (tp + fn) if (tp + fn) else 0
    pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall) if (pos_precision + pos_recall) else 0

    neg_precision = tn / (tn + fn) if (tn + fn) else 0
    neg_recall = tn / (tn + fp) if (tn + fp) else 0
    neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall) if (neg_precision + neg_recall) else 0

    macro_precision = (pos_precision + neg_precision) / 2
    macro_recall = (pos_recall + neg_recall) / 2
    macro_f1 = (pos_f1 + neg_f1) / 2

    return {
        "pos_precision": pos_precision,
        "pos_recall": pos_recall,
        "pos_f1": pos_f1,
        "neg_precision": neg_precision,
        "neg_recall": neg_recall,
        "neg_f1": neg_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1
    }

def evaluate_binary(df, config_name, csv_filename=None, matrix=False):
    tp = fp = tn = fn = 0
    for _, row in df.iterrows():
        gold, pred = int(row['Gold Labels']), int(row['Predicted Labels'])
        if gold == 1 and pred == 1: tp += 1
        elif gold == 0 and pred == 1: fp += 1
        elif gold == 0 and pred == 0: tn += 1
        elif gold == 1 and pred == 0: fn += 1

    metrics = compute_binary_metrics(tp, fp, fn, tn)

    print(f"\n[{config_name}]")
    for k, v in metrics.items():
        label = k.replace("_", " ").title() + ":"
        print(f"{label:<25} {v:.4f}")

    
    # Generate confusion matrix only if requested
    if matrix and csv_filename:
        appendix_dir = "appendix"
        os.makedirs(appendix_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(csv_filename))[0]
        plot_confusion_matrix(tp, fp, fn, tn,
                              title=f"{config_name}",
                              save_path=os.path.join(appendix_dir, f"{base_name}_confusion_matrix.png"))

    return metrics

def average_binary_results(all_metrics, num_files):
    metric_names = list(next(iter(all_metrics.values())).keys())
    avg_metrics = {metric: 0 for metric in metric_names}

    for m in all_metrics.values():
        for k in avg_metrics:
            avg_metrics[k] += m[k]

    for k in avg_metrics:
        avg_metrics[k] = round(round(avg_metrics[k] / num_files, 4) * 100, 2)

    return avg_metrics

def compute_binary_std(all_metrics):
    """Compute standard deviation for binary classification metrics."""
    metric_names = list(next(iter(all_metrics.values())).keys())
    metrics_values = {metric: [] for metric in metric_names}
    
    for file_metrics in all_metrics.values():
        for metric in metrics_values:
            metrics_values[metric].append(file_metrics[metric] * 100)  # Convert to percentage
    
    std_metrics = {}
    for metric in metrics_values:
        std_metrics[f"{metric}_std"] = round(np.std(metrics_values[metric], ddof=1), 2) if len(metrics_values[metric]) > 1 else 0.0
    
    return std_metrics

def save_binary_to_csv(experiment, avg_metrics, out_path, std_metrics=None):
    if std_metrics:
        headers = [
            "Experiment",
            "Pos Precision", "Pos Recall", "Pos F1",
            "Neg Precision", "Neg Recall", "Neg F1",
            "Macro Precision", "Macro Recall", "Macro F1",
            "Pos Precision Std", "Pos Recall Std", "Pos F1 Std",
            "Neg Precision Std", "Neg Recall Std", "Neg F1 Std",
            "Macro Precision Std", "Macro Recall Std", "Macro F1 Std"
        ]
        row = [experiment, 
               avg_metrics["pos_precision"], avg_metrics["pos_recall"], avg_metrics["pos_f1"],
               avg_metrics["neg_precision"], avg_metrics["neg_recall"], avg_metrics["neg_f1"],
               avg_metrics["macro_precision"], avg_metrics["macro_recall"], avg_metrics["macro_f1"],
               std_metrics["pos_precision_std"], std_metrics["pos_recall_std"], std_metrics["pos_f1_std"],
               std_metrics["neg_precision_std"], std_metrics["neg_recall_std"], std_metrics["neg_f1_std"],
               std_metrics["macro_precision_std"], std_metrics["macro_recall_std"], std_metrics["macro_f1_std"]]
    else:
        headers = [
            "Experiment",
            "Pos Precision", "Pos Recall", "Pos F1",
            "Neg Precision", "Neg Recall", "Neg F1",
            "Macro Precision", "Macro Recall", "Macro F1"
        ]
        row = [experiment,
               avg_metrics["pos_precision"], avg_metrics["pos_recall"], avg_metrics["pos_f1"],
               avg_metrics["neg_precision"], avg_metrics["neg_recall"], avg_metrics["neg_f1"],
               avg_metrics["macro_precision"], avg_metrics["macro_recall"], avg_metrics["macro_f1"]]

    write_header = (not os.path.exists(out_path)) or os.path.getsize(out_path) == 0
    df = pd.DataFrame([row], columns=headers)
    df.to_csv(out_path, mode='a', header=write_header, index=False)

def normalize_labels(value):
    if isinstance(value, list):
        return ''.join(str(int(x)) for x in value)
    if isinstance(value, str):
        if value.startswith("[") and value.endswith("]"):
            return ''.join(str(int(x)) for x in ast.literal_eval(value))
        return value
    raise ValueError(f"Unsupported label format: {value}")

def plot_multilabel_confusion_matrix(tp, fp, fn, tn, title, save_path):
    """Plot confusion matrix for multilabel classification with aggregated counts."""
    cm = np.array([[tn, fp],
                   [fn, tp]])
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    classes = ['Negative', 'Positive']
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes,
           yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label',
           title=title)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')

    # Add additional information
    total_predictions = tp + fp + fn + tn
    accuracy = (tp + tn) / total_predictions if total_predictions > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    info_text = f"Accuracy: {accuracy:.3f}\nPrecision: {precision:.3f}\nRecall: {recall:.3f}\nF1-Score: {f1:.3f}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

def evaluate_multilabel(df, include_precision_recall=False, generate_confusion=False, csv_filename=None):
    num_labels = len(LABELS)
    metrics = {label: {'tp': 0, 'fp': 0, 'fn': 0, 'tn': 0} for label in LABELS}
    total_samples = perfect_matches = total_hamming = 0
    noarg_total = noarg_correct = 0

    # Aggregated confusion matrix counts
    total_tp = total_fp = total_fn = total_tn = 0

    for idx, row in df.iterrows():
        gold = normalize_labels(row['Gold Labels'])
        pred = normalize_labels(row['Predicted Labels'])

        if len(gold) != num_labels or len(pred) != num_labels:
            raise ValueError(f"Row {idx} has invalid label lengths after normalization.")

        total_samples += 1
        if gold == pred: perfect_matches += 1
        total_hamming += sum(1 for g, p in zip(gold, pred) if g != p)

        if gold == "00000000":
            noarg_total += 1
            if pred == "00000000":
                noarg_correct += 1

        for i, label in enumerate(LABELS):
            g, p = gold[i], pred[i]
            if g == '1' and p == '1':
                metrics[label]['tp'] += 1
                total_tp += 1
            elif g == '0' and p == '1':
                metrics[label]['fp'] += 1
                total_fp += 1
            elif g == '1' and p == '0':
                metrics[label]['fn'] += 1
                total_fn += 1
            elif g == '0' and p == '0':
                metrics[label]['tn'] += 1
                total_tn += 1

    # Generate aggregated confusion matrix if requested
    if generate_confusion and csv_filename:
        appendix_dir = "appendix"
        os.makedirs(appendix_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(csv_filename))[0]
        plot_multilabel_confusion_matrix(
            total_tp, total_fp, total_fn, total_tn,
            title=f"Multilabel Confusion Matrix - {base_name}",
            save_path=os.path.join(appendix_dir, f"{base_name}_multilabel_confusion_matrix.png")
        )
        
        # Print aggregated metrics for this file
        total_predictions = total_tp + total_fp + total_fn + total_tn
        if total_predictions > 0:
            accuracy = (total_tp + total_tn) / total_predictions
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"\n[Aggregated Confusion Matrix - {base_name}]")
            print(f"Total TP: {total_tp}, FP: {total_fp}, FN: {total_fn}, TN: {total_tn}")
            print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

    results = {}
    for label in LABELS:
        m = metrics[label]
        pos_precision = m['tp'] / (m['tp'] + m['fp']) if (m['tp'] + m['fp']) else 0
        pos_recall = m['tp'] / (m['tp'] + m['fn']) if (m['tp'] + m['fn']) else 0
        pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall) if (pos_precision + pos_recall) else 0

        neg_precision = m['tn'] / (m['tn'] + m['fn']) if (m['tn'] + m['fn']) else 0
        neg_recall = m['tn'] / (m['tn'] + m['fp']) if (m['tn'] + m['fp']) else 0
        neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall) if (neg_precision + neg_recall) else 0

        macro_precision = (pos_precision + neg_precision) / 2
        macro_recall = (pos_recall + neg_recall) / 2
        macro_f1 = (pos_f1 + neg_f1) / 2

        if include_precision_recall:
            results[label] = [pos_f1, neg_f1, macro_f1, pos_precision, pos_recall, neg_precision, neg_recall, macro_precision, macro_recall]
        else:
            results[label] = [pos_f1, neg_f1, macro_f1]

    return results

def average_multilabel_results(all_metrics, num_files, include_precision_recall=False):
    if include_precision_recall:
        avg_dict = {label: [0, 0, 0, 0, 0, 0, 0, 0, 0] for label in LABELS}
        num_metrics = 9
    else:
        avg_dict = {label: [0, 0, 0] for label in LABELS}
        num_metrics = 3

    for metrics in all_metrics.values():
        for label in LABELS:
            for i in range(num_metrics):
                avg_dict[label][i] += metrics[label][i]

    for label in LABELS:
        avg_dict[label] = [round(round(x / num_files, 4) * 100, 2) for x in avg_dict[label]]

    if include_precision_recall:
        avg_dict["Macro Avg"] = [
            round(sum(avg_dict[label][0] for label in LABELS) / len(LABELS), 2),  # pos_f1
            round(sum(avg_dict[label][1] for label in LABELS) / len(LABELS), 2),  # neg_f1
            round(sum(avg_dict[label][2] for label in LABELS) / len(LABELS), 2),  # macro_f1
            round(sum(avg_dict[label][3] for label in LABELS) / len(LABELS), 2),  # pos_precision
            round(sum(avg_dict[label][4] for label in LABELS) / len(LABELS), 2),  # pos_recall
            round(sum(avg_dict[label][5] for label in LABELS) / len(LABELS), 2),  # neg_precision
            round(sum(avg_dict[label][6] for label in LABELS) / len(LABELS), 2),  # neg_recall
            round(sum(avg_dict[label][7] for label in LABELS) / len(LABELS), 2),  # macro_precision
            round(sum(avg_dict[label][8] for label in LABELS) / len(LABELS), 2)   # macro_recall
        ]
    else:
        avg_dict["Macro Avg"] = [
            round(sum(avg_dict[label][0] for label in LABELS) / len(LABELS), 2),
            round(sum(avg_dict[label][1] for label in LABELS) / len(LABELS), 2),
            round(sum(avg_dict[label][2] for label in LABELS) / len(LABELS), 2)
        ]

    return avg_dict

def compute_multilabel_std(all_metrics, include_precision_recall=False):
    """Compute standard deviation for multilabel classification metrics."""
    if include_precision_recall:
        metrics_storage = {label: {
            "pos_f1": [], "neg_f1": [], "macro_f1": [],
            "pos_precision": [], "pos_recall": [], "neg_precision": [], "neg_recall": [],
            "macro_precision": [], "macro_recall": []
        } for label in LABELS}
        metric_keys = ["pos_f1", "neg_f1", "macro_f1", "pos_precision", "pos_recall", 
                      "neg_precision", "neg_recall", "macro_precision", "macro_recall"]
    else:
        metrics_storage = {label: {"pos_f1": [], "neg_f1": [], "macro_f1": []} for label in LABELS}
        metric_keys = ["pos_f1", "neg_f1", "macro_f1"]
    
    # Collect all values across files
    for file_metrics in all_metrics.values():
        for label in LABELS:
            for i, key in enumerate(metric_keys):
                metrics_storage[label][key].append(file_metrics[label][i])
    
    # Calculate standard deviations
    std_results = {}
    for label in LABELS:
        label_stds = []
        for key in metric_keys:
            std_val = round(np.std(metrics_storage[label][key], ddof=1), 2) if len(metrics_storage[label][key]) > 1 else 0.0
            label_stds.append(std_val)
        std_results[label] = label_stds
    
    # Calculate macro average standard deviations
    if include_precision_recall:
        macro_stds = []
        for i in range(9):
            macro_std = round(np.mean([std_results[label][i] for label in LABELS]), 2)
            macro_stds.append(macro_std)
        std_results["Macro Avg"] = macro_stds
    else:
        macro_pos_std = round(np.mean([std_results[label][0] for label in LABELS]), 2)
        macro_neg_std = round(np.mean([std_results[label][1] for label in LABELS]), 2)
        macro_avg_std = round(np.mean([std_results[label][2] for label in LABELS]), 2)
        std_results["Macro Avg"] = [macro_pos_std, macro_neg_std, macro_avg_std]
    
    return std_results
def get_experiment_name(csv_files):
    common = os.path.commonprefix([os.path.basename(f) for f in csv_files])
    return common.rstrip("_- ")

def save_multilabel_to_csv(experiment, avg_dict, out_path, std_dict=None, include_precision_recall=False):
    labels_for_csv = LABELS + ["Macro Avg"]

    if include_precision_recall:
        base_headers = ["Experiment", "Type"] + labels_for_csv
        if std_dict:
            headers = base_headers + [f"{label}_F1_Std" for label in labels_for_csv] + \
                     [f"{label}_Precision_Std" for label in labels_for_csv] + \
                     [f"{label}_Recall_Std" for label in labels_for_csv]
            
            # Rows for F1, Precision, and Recall with standard deviations
            pos_f1_row = [experiment, "Positive_F1"] + [avg_dict[label][0] for label in labels_for_csv] + \
                        [std_dict[label][0] for label in labels_for_csv] + [0] * len(labels_for_csv) + [0] * len(labels_for_csv)
            neg_f1_row = [experiment, "Negative_F1"] + [avg_dict[label][1] for label in labels_for_csv] + \
                        [std_dict[label][1] for label in labels_for_csv] + [0] * len(labels_for_csv) + [0] * len(labels_for_csv)
            avg_f1_row = [experiment, "Average_F1"] + [avg_dict[label][2] for label in labels_for_csv] + \
                        [std_dict[label][2] for label in labels_for_csv] + [0] * len(labels_for_csv) + [0] * len(labels_for_csv)
            
            pos_prec_row = [experiment, "Positive_Precision"] + [avg_dict[label][3] for label in labels_for_csv] + \
                          [0] * len(labels_for_csv) + [std_dict[label][3] for label in labels_for_csv] + [0] * len(labels_for_csv)
            neg_prec_row = [experiment, "Negative_Precision"] + [avg_dict[label][5] for label in labels_for_csv] + \
                          [0] * len(labels_for_csv) + [std_dict[label][5] for label in labels_for_csv] + [0] * len(labels_for_csv)
            avg_prec_row = [experiment, "Average_Precision"] + [avg_dict[label][7] for label in labels_for_csv] + \
                          [0] * len(labels_for_csv) + [std_dict[label][7] for label in labels_for_csv] + [0] * len(labels_for_csv)
            
            pos_rec_row = [experiment, "Positive_Recall"] + [avg_dict[label][4] for label in labels_for_csv] + \
                         [0] * len(labels_for_csv) + [0] * len(labels_for_csv) + [std_dict[label][4] for label in labels_for_csv]
            neg_rec_row = [experiment, "Negative_Recall"] + [avg_dict[label][6] for label in labels_for_csv] + \
                         [0] * len(labels_for_csv) + [0] * len(labels_for_csv) + [std_dict[label][6] for label in labels_for_csv]
            avg_rec_row = [experiment, "Average_Recall"] + [avg_dict[label][8] for label in labels_for_csv] + \
                         [0] * len(labels_for_csv) + [0] * len(labels_for_csv) + [std_dict[label][8] for label in labels_for_csv]
            
            rows = [pos_f1_row, neg_f1_row, avg_f1_row, pos_prec_row, neg_prec_row, avg_prec_row, pos_rec_row, neg_rec_row, avg_rec_row]
        else:
            headers = base_headers
            pos_f1_row = [experiment, "Positive_F1"] + [avg_dict[label][0] for label in labels_for_csv]
            neg_f1_row = [experiment, "Negative_F1"] + [avg_dict[label][1] for label in labels_for_csv]
            avg_f1_row = [experiment, "Average_F1"] + [avg_dict[label][2] for label in labels_for_csv]
            pos_prec_row = [experiment, "Positive_Precision"] + [avg_dict[label][3] for label in labels_for_csv]
            neg_prec_row = [experiment, "Negative_Precision"] + [avg_dict[label][5] for label in labels_for_csv]
            avg_prec_row = [experiment, "Average_Precision"] + [avg_dict[label][7] for label in labels_for_csv]
            pos_rec_row = [experiment, "Positive_Recall"] + [avg_dict[label][4] for label in labels_for_csv]
            neg_rec_row = [experiment, "Negative_Recall"] + [avg_dict[label][6] for label in labels_for_csv]
            avg_rec_row = [experiment, "Average_Recall"] + [avg_dict[label][8] for label in labels_for_csv]
            
            rows = [pos_f1_row, neg_f1_row, avg_f1_row, pos_prec_row, neg_prec_row, avg_prec_row, pos_rec_row, neg_rec_row, avg_rec_row]
    else:
        if std_dict:
            headers = ["Experiment", "Type"] + labels_for_csv + [f"{label}_Std" for label in labels_for_csv]
            pos_row = [experiment, "Positive"] + [avg_dict[label][0] for label in labels_for_csv] + [std_dict[label][0] for label in labels_for_csv]
            neg_row = [experiment, "Negative"] + [avg_dict[label][1] for label in labels_for_csv] + [std_dict[label][1] for label in labels_for_csv]
            avg_row = [experiment, "Average"] + [avg_dict[label][2] for label in labels_for_csv] + [std_dict[label][2] for label in labels_for_csv]
        else:
            headers = ["Experiment", "Type"] + labels_for_csv
            pos_row = [experiment, "Positive"] + [avg_dict[label][0] for label in labels_for_csv]
            neg_row = [experiment, "Negative"] + [avg_dict[label][1] for label in labels_for_csv]
            avg_row = [experiment, "Average"] + [avg_dict[label][2] for label in labels_for_csv]
        
        rows = [pos_row, neg_row, avg_row]

    write_header = (not os.path.exists(out_path)) or os.path.getsize(out_path) == 0
    df = pd.DataFrame(rows, columns=headers)
    df.to_csv(out_path, mode='a', header=write_header, index=False)

def main():
    parser = argparse.ArgumentParser(description="Evaluate one or more CSV files with classification results.")
    parser.add_argument("csv_files", nargs='+', help="List of CSV files.")
    parser.add_argument("--config", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--out_csv", type=str, help="Path to save Config 2 or 3 results as CSV.")
    parser.add_argument("--matrix", action="store_true",
                        help="Generate confusion matrix plots (only for config 1 and 2)")
    parser.add_argument("--std_dev", action="store_true",
                        help="Calculate and include standard deviation in results")
    parser.add_argument("--pre_rec", action="store_true",
                        help="Include precision and recall metrics (only for config 3)")
    parser.add_argument("--confusion_3", action="store_true",
                        help="Generate aggregated confusion matrix for multilabel classification (only for config 3)")
    args = parser.parse_args()

    # Validate --matrix usage
    if args.matrix and args.config not in [1, 2]:
        parser.error("--matrix is only available for config 1 or config 2.")
    
    # Validate --pre_rec usage
    if args.pre_rec and args.config != 3:
        parser.error("--pre_rec is only available for config 3.")
    
    # Validate --confusion_3 usage
    if args.confusion_3 and args.config != 3:
        parser.error("--confusion_3 is only available for config 3.")

    if args.config in [1, 2]:
        binary_results = {}
        for i, csv_file in enumerate(args.csv_files, 1):
            df = pd.read_csv(csv_file)
            binary_results[csv_file] = evaluate_binary(
                df,
                config_name=f"Config {args.config} - Binary (File {i})",
                csv_filename=csv_file,
                matrix=args.matrix
            )

        avg_results = average_binary_results(binary_results, len(args.csv_files))
        std_results = compute_binary_std(binary_results) if args.std_dev else None
        experiment_name = get_experiment_name(args.csv_files)
        experiment_name = experiment_name.replace("_seed", "")
        experiment_name = experiment_name.replace("-seed", "")
        
        if args.out_csv:
            save_binary_to_csv(experiment_name, avg_results, args.out_csv, std_results)
        else:
            print("\n=== Average Binary Metrics ===")
            for k, v in avg_results.items():
                if args.std_dev and std_results:
                    std_key = f"{k}_std"
                    std_val = std_results.get(std_key, 0.0)
                    print(f"{k.replace('_', ' ').title()}: {v:.2f} ± {std_val:.2f}")
                else:
                    print(f"{k.replace('_', ' ').title()}: {v:.2f}")

    elif args.config == 3:
        include_full_metrics = args.pre_rec or bool(args.out_csv)
        multilabel_results = {}
        for i, csv_file in enumerate(args.csv_files, 1):
            df = pd.read_csv(csv_file, dtype={'Gold Labels': str, 'Predicted Labels': str})
            multilabel_results[csv_file] = evaluate_multilabel(
                df, 
                include_precision_recall=include_full_metrics,
                generate_confusion=args.confusion_3,
                csv_filename=csv_file
            )

        avg_results = average_multilabel_results(multilabel_results, len(args.csv_files), include_precision_recall=include_full_metrics)
        std_results = compute_multilabel_std(multilabel_results, include_precision_recall=include_full_metrics) if args.std_dev else None
        experiment_name = get_experiment_name(args.csv_files)
        experiment_name = experiment_name.replace("_seed", "")
        experiment_name = experiment_name.replace("-seed", "")
        
        if args.out_csv:
            save_multilabel_to_csv(experiment_name, avg_results, args.out_csv, std_results, include_precision_recall=True)
        else:
            print("\n=== Average Multilabel Metrics ===")
            for label in LABELS + ["Macro Avg"]:
                if include_full_metrics:
                    pos_f1, neg_f1, macro_f1, pos_prec, pos_rec, neg_prec, neg_rec, macro_prec, macro_rec = avg_results[label]
                    if args.std_dev and std_results:
                        pos_f1_std, neg_f1_std, macro_f1_std, pos_prec_std, pos_rec_std, neg_prec_std, neg_rec_std, macro_prec_std, macro_rec_std = std_results[label]
                        print(f"{label:<3} | Pos F1: {pos_f1:.2f}±{pos_f1_std:.2f} | Neg F1: {neg_f1:.2f}±{neg_f1_std:.2f} | Macro F1: {macro_f1:.2f}±{macro_f1_std:.2f}")
                        print(f"     | Pos Prec: {pos_prec:.2f}±{pos_prec_std:.2f} | Neg Prec: {neg_prec:.2f}±{neg_prec_std:.2f} | Macro Prec: {macro_prec:.2f}±{macro_prec_std:.2f}")
                        print(f"     | Pos Rec: {pos_rec:.2f}±{pos_rec_std:.2f} | Neg Rec: {neg_rec:.2f}±{neg_rec_std:.2f} | Macro Rec: {macro_rec:.2f}±{macro_rec_std:.2f}")
                    else:
                        print(f"{label:<3} | Pos F1: {pos_f1:.2f} | Neg F1: {neg_f1:.2f} | Macro F1: {macro_f1:.2f}")
                        print(f"     | Pos Prec: {pos_prec:.2f} | Neg Prec: {neg_prec:.2f} | Macro Prec: {macro_prec:.2f}")
                        print(f"     | Pos Rec: {pos_rec:.2f} | Neg Rec: {neg_rec:.2f} | Macro Rec: {macro_rec:.2f}")
                else:
                    pos_f1, neg_f1, macro_f1 = avg_results[label]
                    if args.std_dev and std_results:
                        pos_std, neg_std, macro_std = std_results[label]
                        print(f"{label:<3} | Pos F1: {pos_f1:.2f}±{pos_std:.2f} | Neg F1: {neg_f1:.2f}±{neg_std:.2f} | Macro F1: {macro_f1:.2f}±{macro_std:.2f}")
                    else:
                        print(f"{label:<3} | Pos F1: {pos_f1:.4f} | Neg F1: {neg_f1:.4f} | Macro F1: {macro_f1:.4f}")

if __name__ == "__main__":
    main()
