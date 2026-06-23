import numpy as np
import h5py
import os
import sys
import copy
import torch
import matplotlib.ticker as mtick
import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_auc_score
from decimal import Decimal, ROUND_HALF_UP, ROUND_HALF_EVEN
sys.path.append('..')
import pandas as pd



def calculate_prec_recall_f1(labels_lst, predicted_lst):
    # Beta: The strength of recall versus precision in the F-score. beta == 1.0 means recall and precision are equally important, that is F1-score
    mac_precision, mac_recall, mac_f1, _ = precision_recall_fscore_support(y_true=labels_lst, y_pred=predicted_lst, beta=1.0, average='macro')
    
    return mac_precision, mac_recall, mac_f1

def calculate_metrics(labels_lst, predicted_lst):
    # Beta: The strength of recall versus precision in the F-score. beta == 1.0 means recall and precision are equally important, that is F1-score
    mac_precision, mac_recall, mac_f1, _ = precision_recall_fscore_support(y_true=labels_lst, y_pred=predicted_lst, beta=1.0, average='macro')
    accuracy = accuracy_score(y_true=labels_lst, y_pred=predicted_lst)

    return mac_precision, mac_recall, mac_f1, accuracy

def calculate_metrics_per_class(labels_lst, predicted_lst, expected_tracts, prob_lst=None, csv_path="per_class_metrics.csv"):

    labels_lst = np.array(labels_lst)
    predicted_lst = np.array(predicted_lst)
    num_classes = len(expected_tracts)

    mac_precision, mac_recall, mac_f1, _ = precision_recall_fscore_support(
        y_true=labels_lst,
        y_pred=predicted_lst,
        beta=1.0,
        average='macro'
    )
    overall_accuracy = accuracy_score(labels_lst, predicted_lst)

    if prob_lst is not None:
        try:
            overall_auc = roc_auc_score(labels_lst, prob_lst, multi_class='ovr', average='macro')
        except Exception as e:
            print(f"AUC calculation failed: {e}")
            overall_auc = None
    else:
        overall_auc = None

    per_class_metrics = []

    for i in range(num_classes):
        y_true_binary = (labels_lst == i).astype(int)
        y_pred_binary = (predicted_lst == i).astype(int)

        if y_true_binary.sum() == 0:
            precision = recall = f1 = accuracy = np.nan
            auc = np.nan
        else:
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true=y_true_binary,
                y_pred=y_pred_binary,
                beta=1.0,
                average='binary'
            )
            accuracy = (y_true_binary == y_pred_binary).mean()

            if prob_lst is not None:
                prob_i = prob_lst[:, i]  
                try:
                    auc = roc_auc_score(y_true_binary, prob_i)
                except Exception:
                    auc = np.nan
            else:
                auc = np.nan

        per_class_metrics.append({
            "fiber": expected_tracts[i],
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "count": int(np.sum(labels_lst == i))
        })

    df = pd.DataFrame(per_class_metrics)
    df.to_csv(csv_path, index=False)
    print(f"Saved per-class metrics to: {csv_path}")

    return df, mac_precision, mac_recall, mac_f1, overall_accuracy, overall_auc

def save_confusion_matrix(labels_lst, predicted_lst, save_path, class_names=None):

    cm = confusion_matrix(labels_lst, predicted_lst)
    
    num_classes = cm.shape[0]
    
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    elif len(class_names) != num_classes:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',  
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        square=True,
        cbar_kws={'shrink': 0.8}
    )
    
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
        
def classify_report(labels_lst, predicted_lst, label_names, logger, out_path, metric_name):
    """Generate classification performance report"""
    cls_report = classification_report(y_true=labels_lst, y_pred=predicted_lst, digits=5, target_names=label_names)
    logger.info('=' * 55)
    logger.info('Best {} classification report:\n{}'.format(metric_name, cls_report))
    logger.info('=' * 55)
    logger.info('\n')

    if 'test' in metric_name:
        test_res = h5py.File(out_path, "w")
        test_res['val_predictions'] = predicted_lst
        test_res['val_labels'] = labels_lst
        test_res['label_names'] = label_names
        test_res['classification_report'] = cls_report
    else:
        val_res = h5py.File(os.path.join(out_path, 'entire_data_validation_results_best_{}.h5'.format(metric_name)), "w")
        val_res['val_predictions'] = predicted_lst
        val_res['val_labels'] = labels_lst
        val_res['label_names'] = label_names
        val_res['classification_report'] = cls_report
