from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import h5py
import sys
import os
sys.path.append('../')
# import utils.tract_feat as tract_feat
from tqdm import tqdm
import random

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


def split_list_ratio(original_list, split_ratio = 0.8):
    if not original_list:
        return [], []
    
    total_len = len(original_list)
    split_point = round(total_len * split_ratio) 
    
    # Randomise the original list
    shuffled_list = original_list.copy()
    random.shuffle(shuffled_list)
    
    list1 = shuffled_list[:split_point]
    list2 = shuffled_list[split_point:]
    
    return list1, list2

class h5_dataset(data.Dataset):
    def __init__(self, root, sample_limit_num=0):
        self.root = root  # Folder path
        self.Points_list = []  # Store all Points
        self.labels_list = []    # Store all abels
        self.global_indices = [] # Global Index Mapping
        self.sample_info = []    # Sample information
        
        sample_list = sorted(os.listdir(root))
        if sample_limit_num:
            sample_list = sample_list[:sample_limit_num]
            
        for sample_idx, sample_name in enumerate(tqdm(sample_list, desc="processing")):
            sample_path = os.path.join(self.root, sample_name)
            if not os.path.isdir(sample_path):
                continue

            features_path = os.path.join(sample_path, "features.h5")
            labels_path = os.path.join(sample_path, "labels.h5")
            
            if not (os.path.exists(features_path) and os.path.exists(labels_path)):
                print(f"Warning: Sample {sample_name} is missing the H5 file; skipping.")
                continue
            
            with h5py.File(features_path, 'r') as f:
                features = f['features'][:]  # [N_i, 15, 12]
                
            with h5py.File(labels_path, 'r') as f:
                labels = f['labels'][:]      # [N_i,]
                
            if features.shape[0] != labels.shape[0]:
                print(f"Warning: Sample {sample_name} has mismatched feature and label counts")
                continue
            
            features, labels = clean_features_and_labels(features, labels)
                            
            num_fibers = features.shape[0]
            
            # Save sample information
            self.sample_info.append({
                'sample_name': sample_name,
                'start_idx': len(self.labels_list),
                'end_idx': len(self.labels_list) + num_fibers - 1,
                'num_fibers': num_fibers,
                'point_shape': features.shape,
            })
            
            for i in range(num_fibers):
                self.Points_list.append(features[i]) 
                self.labels_list.append(labels[i]) 
                self.global_indices.append((sample_idx, i))
        
        self.Points = np.array(self.Points_list)  # [N, 15, 12]
        self.labels = np.array(self.labels_list)      # [N,]

        print(f"{len(self.sample_info)} samples, {len(self.labels)} fibers")
    
    def __getitem__(self, index):
        point_set = self.Points[index]
        label = self.labels[index]       
        
        if not isinstance(point_set, torch.Tensor):
            point_set = torch.from_numpy(point_set.astype(np.float32))
        
        if not isinstance(label, torch.Tensor):
            label = torch.tensor(label, dtype=torch.long)
        
        return point_set, label
    
    def __len__(self):
        return len(self.labels)

