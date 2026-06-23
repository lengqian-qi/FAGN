import argparse
import os
import sys
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
from typing import *
from matplotlib import pyplot as plt
from pathlib import Path
from torch.cuda.amp import autocast as autocast
import h5py
from utils.visualize_fibers import  plot_selected_fibers_error, plot_selected_fibers_multi_views_error
from utils.metrics_plots import save_confusion_matrix, calculate_metrics, calculate_metrics_per_class
from FAGN import FrenetPointNetCls
import json

json_path = "./global_label_mapping.json"
with open(json_path, 'r', encoding='utf-8') as f:
    label_mapping = json.load(f)
expected_tracts = list(label_mapping.keys())

expected_tracts_select = ["AF", "FX", 
                         "SLF_III", "ILF", "SLF_I", "FPT"]

def stratified_sampling(points, labels, ratio=0.05, min_samples=1000):

    unique_labels = np.unique(labels)

    sampled_indices = []

    for lb in unique_labels:
        idx = np.where(labels == lb)[0]

        num_total = len(idx)
        num_sample = max(int(num_total * ratio), min_samples)

        num_sample = min(num_sample, num_total)

        sampled_idx = np.random.choice(idx, num_sample, replace=False)

        sampled_indices.append(sampled_idx)

        print(f"Label {lb}: {num_total} -> {num_sample}")

    sampled_indices = np.concatenate(sampled_indices)

    np.random.shuffle(sampled_indices)

    return points[sampled_indices], labels[sampled_indices]

def clean_features_and_labels(features, labels):
    """
    Remove features containing NaN or Inf values
    """

    invalid_mask = np.isnan(features) | np.isinf(features)

    if not invalid_mask.any():
        return features, labels

    # Filtering by fibre
    valid_mask = ~invalid_mask.any(axis=(1, 2))

    cleaned_features = features[valid_mask]
    cleaned_labels = labels[valid_mask]

    removed_count = np.sum(~valid_mask)
    print(f"Remove {removed_count} fibres containing NaN/Inf")

    return cleaned_features, cleaned_labels

def predict_in_batches(model, features, labels, batch_size=32, device='cuda:0'):
    
    """
    Perform predictions in batches to avoid memory overflow.
    """
    model.eval()
    N = features.shape[0]
    all_preds = []
    
    if N <= batch_size:
        with torch.no_grad():
            features_batch = features.to(device)
            preds = model(features_batch)
            return preds
    else:
        num_batches = (N + batch_size - 1) // batch_size
        batch_preds = []
        
        with torch.no_grad():
            for i in tqdm(range(num_batches), desc=f"(batch size={batch_size})", leave=False):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, N)
                
                features_batch = features[start_idx:end_idx].to(device)
                
                with autocast():
                    preds_batch = model(features_batch)
                
                batch_preds.append(preds_batch.cpu())
                
                del features_batch, preds_batch
                if i % 5 == 0:  # 
                    torch.cuda.empty_cache()
        
        all_preds = torch.cat(batch_preds, dim=0)
        torch.cuda.empty_cache()
        
        return all_preds.to(device)
    
def iter_one_sample(model: torch.nn.Module,
                   device,
                   root):

    model.eval()
    sample_list = sorted(os.listdir(root))
    mac_precision_list, mac_recall_list, mac_f1_list, accuracy_list = [], [], [], []

    for sample_idx, sample_name in enumerate(tqdm(sample_list, desc="process")):
        sample_path = os.path.join(root, sample_name)
        if not os.path.isdir(sample_path):
            continue

        features_path = os.path.join(sample_path, "features.h5")
        labels_path = os.path.join(sample_path, "labels.h5")
        
        if not (os.path.exists(features_path) and os.path.exists(labels_path)):
            continue
        
        with h5py.File(features_path, 'r') as f:
            features = f['features'][:]  # [N_i, 15, 3]
            
        with h5py.File(labels_path, 'r') as f:
            labels = f['labels'][:]      # [N_i,]
            
        if features.shape[0] != labels.shape[0]:
            continue
        
        features, labels = clean_features_and_labels(features, labels)


        features = np.array(features)  # [N, 15, 3]
        labels = np.array(labels)      # [N,]
        
        points = torch.from_numpy(features.astype(np.float32))  # [N, 15, 3]
        labels = torch.from_numpy(labels.astype(np.int64))   # [N,]

        if labels.dim() == 2:
            labels = labels[:, 0]  # [B,1] -> [B]
            
        points = points.transpose(2, 1)  # points [B, 3, N]
        points, labels = points.to(device), labels.to(device)

        with torch.no_grad():
            pred = predict_in_batches(model, points, labels)  
            
        labels = labels.cpu().detach().numpy()
    
        _, pred_idx = torch.max(pred, dim=1)
        pred_idx = pred_idx.cpu().detach().numpy()
        
        confusion_matrix_path = os.path.join(sample_path,  "confusion_matrix_FAGN.png")
        mac_precision, mac_recall, mac_f1, accuracy = calculate_metrics(labels.tolist(), pred_idx.tolist())
       
        csv_path = os.path.join(sample_path, "per_class_metrics_FAGN.csv")
        calculate_metrics_per_class(labels.tolist(), pred_idx.tolist(), expected_tracts, prob_lst=None, csv_path=csv_path)
        save_confusion_matrix(labels, pred_idx, confusion_matrix_path, class_names = expected_tracts)
        print(f" mac_precision:{mac_precision}, mac_recall:{mac_recall}, mac_f1:{mac_f1}, accuracy:{accuracy}")

        mac_precision_list.append(mac_precision)
        mac_recall_list.append(mac_recall)
        mac_f1_list.append(mac_f1)
        accuracy_list.append(accuracy)
        
        h5_preds_path = os.path.join(sample_path,  "preds_FAGN.h5")
        
        with h5py.File(h5_preds_path, 'w') as f:
            f.create_dataset('labels', data=pred_idx, compression='gzip')
            print(f"  标签已保存到: {h5_preds_path}")
        
        output_path_label =os.path.join(sample_path, "fiber_visualization_gt")
        output_path_pred =os.path.join(sample_path, "fiber_visualization_pred_FAGN")
        output_path_label_3view =os.path.join(sample_path, "fiber_visualization_gt_3view")
        output_path_pred_3view =os.path.join(sample_path, "fiber_visualization_pred_FAGN_3view")

        plot_selected_fibers_error(features, labels, labels, expected_tracts, expected_tracts_select, output_dir=output_path_label)
        plot_selected_fibers_error(features, labels, pred_idx, expected_tracts, expected_tracts_select, output_dir=output_path_pred)
        plot_selected_fibers_multi_views_error(features, labels, labels, expected_tracts, expected_tracts_select, output_dir=output_path_label_3view)
        plot_selected_fibers_multi_views_error(features, labels, pred_idx, expected_tracts, expected_tracts_select, output_dir=output_path_pred_3view)
        
    nums = len(mac_precision_list)
    print(f"finish! mac_precision:{sum(mac_precision_list)/nums},mac_recall:{sum(mac_recall_list)/nums},mac_f1:{sum(mac_f1_list)/nums},accuracy:{sum(accuracy_list)/nums}")
    
if __name__ == '__main__':
    device = "cuda:0"
    checkpoint_path = ""
    num_classes = 72
    root = ""
    
    checkpoints = torch.load(checkpoint_path,
                            map_location=device,
                            weights_only=True)
    model = FrenetPointNetCls(k=num_classes, use_frenet=True) 

    model.to(device)
    if checkpoints['model_state_dict'] is not None:
        model.load_state_dict(checkpoints['model_state_dict'])
    
    iter_one_sample(model,
                   device,
                   root)