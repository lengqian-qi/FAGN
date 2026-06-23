import os
import json
import numpy as np
import nibabel as nib
import h5py
from scipy.interpolate import interp1d
from collections import OrderedDict
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def compute_frenet_frame(points):
    """
    Input:
        points: [P,3]
    Output:
        T: [P,3] tangent
        N: [P,3] normal
        B: [P,3] binormal
        curvature: [P]
        torsion: [P]
    """
    eps = 1e-8
    
    d1 = np.gradient(points, axis=0)
    d2 = np.gradient(d1, axis=0)
    d3 = np.gradient(d2, axis=0)

    # ===== Tangent =====
    T = d1 / (np.linalg.norm(d1, axis=1, keepdims=True) + eps)

    # ===== Curvature =====
    cross = np.cross(d1, d2)
    cross_norm = np.linalg.norm(cross, axis=1, keepdims=True)

    d1_norm = np.linalg.norm(d1, axis=1, keepdims=True)
    curvature = (cross_norm[:,0]) / (d1_norm[:,0]**3 + eps)

    # ===== Normal =====
    N = np.zeros_like(T)
    valid = cross_norm[:,0] > eps
    N[valid] = cross[valid] / (cross_norm[valid] + eps)

    # ===== Binormal =====
    B = np.cross(T, N)

    # ===== Torsion =====
    torsion = np.einsum('ij,ij->i', cross, d3) / ((cross_norm[:,0]**2) + eps)

    return T, N, B, curvature, torsion

def extract_features_from_trk(trk_file, num_points=15):
    """
    Extract RAS features directly from TRK files
    Parameters:
        trk_file: Path to TRK file
        num_points: Number of sampling points per fiber
    Returns:
        feat: Feature array [N, num_points, 3]
    """

    trk = nib.streamlines.load(trk_file)
    streamlines = trk.streamlines  
    num_fibers = len(streamlines)
    
    feat = np.zeros((num_fibers, num_points, 3))  # [N, num_points, 3]
    
    #Resampling each fiber
    for i, fiber in enumerate(streamlines):
        if i % 10000 == 0:
            print(f"Processing fibers {i}/{num_fibers}")
        
        n_points = len(fiber)
        if n_points <= 1:
            continue  
            
        # Create an Interpolator (Linear Interpolation)
        t_original = np.linspace(0, 1, n_points)
        t_target = np.linspace(0, 1, num_points)
        
        for dim in range(3):
            interpolator = interp1d(t_original, fiber[:, dim], 
                                   kind='linear', 
                                   bounds_error=False,
                                   fill_value="extrapolate")
            feat[i, :, dim] = interpolator(t_target)
    
    return feat, streamlines

def extract_features_frenet(trk_file, num_points=15):
    """
    output:
        feat: [N,15,12]
    """

    trk = nib.streamlines.load(trk_file)
    streamlines = trk.streamlines  
    num_fibers = len(streamlines)
    
    feat = np.zeros((num_fibers, num_points, 12))
    
    for i, fiber in enumerate(streamlines):
        if i % 10000 == 0:
            print(f"Processing fibers {i}/{num_fibers}")
        
        if len(fiber) <= 3:
            continue
        
        # ===== resample =====
        t_ori = np.linspace(0,1,len(fiber))
        t_tar = np.linspace(0,1,num_points)

        resampled = np.zeros((num_points,3))
        for d in range(3):
            f = interp1d(t_ori, fiber[:,d], kind='linear', fill_value="extrapolate")
            resampled[:,d] = f(t_tar)

        # ===== Frenet =====
        T, N, B, curvature, torsion = compute_frenet_frame(resampled)

        # ===== concat =====
        feat[i,:,0:3] = resampled
        feat[i,:,3:6] = T
        feat[i,:,6:9] = N
        feat[i,:,9:12] = B

    return feat

def process_sample(root_path, label_mapping, out_sample_dir, num_points=15):
    """
    Processing individual sample folders
    Parameters:
        root_path: Path to the sample folder
        label_mapping: Label mapping dictionary
        num_points: Number of sampling points per fibre
    Returns:
        all_features: Merged feature array [N, num_points, 12]
        all_labels: Merged label array [N,]
    """
    print(f"\processing: {os.path.basename(root_path)}")
    print("="*60)
    os.makedirs(out_sample_dir, exist_ok=True)
    expected_tracts = list(label_mapping.keys())

    all_features_list = []
    all_labels_list = []
    
    for trk_name in expected_tracts:
        trk_path = os.path.join(root_path, trk_name + ".trk")
                
        # extraction
        features = extract_features_frenet(trk_path, num_points)
        num_fibers = features.shape[0]
        
        # Retrieve the label from the global mapping
        label = label_mapping[trk_name]
        labels = np.full(num_fibers, label, dtype=np.int32)
        
        all_features_list.append(features)
        all_labels_list.append(labels)
        
        print(f" {trk_name}: {num_fibers}fibers, label={label}")
    

    # Merge all features and tags
    all_features = np.vstack(all_features_list)  # [N, num_points, 3]
    all_labels = np.concatenate(all_labels_list)  # [N,]
    
    print(f"features.shape: {all_features.shape}")
    print(f"labels.shape: {all_labels.shape}")
    
    h5_features_path = os.path.join(out_sample_dir,  "features.h5")
    h5_labels_path = os.path.join(out_sample_dir, "labels.h5")
    
    with h5py.File(h5_features_path, 'w') as f:
        f.create_dataset('features', data=all_features, compression='gzip')
        print(f"  Features have been saved to: {h5_features_path}")
    
    with h5py.File(h5_labels_path, 'w') as f:
        f.create_dataset('labels', data=all_labels, compression='gzip')
        print(f"  The labels has been saved to: {h5_labels_path}")
    
    return all_features, all_labels

def check_all_trk_files_exist(root_path, expected_tracts):
    
    for trk_name in expected_tracts:
        trk_path = os.path.join(root_path, trk_name + ".trk")
        if not os.path.exists(trk_path):
            return False
    return True

def Global_label_Mapping(root_dir, output_json_dir):
    label_mapping = OrderedDict()
    
    for i, trk_name in enumerate(sorted(os.listdir(root_dir)), 0):
        if not trk_name.endswith('.trk'):
            continue
        tract_name = os.path.splitext(trk_name)[0]
        label_mapping[tract_name] = i
        
    print(f"Global label Mapping:")
    for tract_name, label in label_mapping.items():
        print(f"  {tract_name} -> label {label}")
        
    os.makedirs(output_json_dir, exist_ok=True)
    mapping_path = os.path.join(output_json_dir, "global_label_mapping.json")
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(label_mapping, f, indent=2, ensure_ascii=False)
            
def process_all_samples(root_dir, output_dir, json_path):
    """
    Process all sample folders
    Parameters:
        root_dir: Root directory containing multiple sample folders
        output_dir: Output directory
        json_path: Path for JSON file
    """
    
    with open(json_path, 'r', encoding='utf-8') as f:
        label_mapping = json.load(f)
    expected_tracts = list(label_mapping.keys())

    all_sample_list = sorted(os.listdir(root_dir))
    for modal in ["train", "eval", "test"]:
        if modal == "train":
            sample_list = all_sample_list[:int(0.6*len(all_sample_list))]
        elif modal == "eval":
            sample_list = all_sample_list[int(0.6*len(all_sample_list)):int(0.8*len(all_sample_list))]
        elif modal == "test":
            sample_list = all_sample_list[int(0.8*len(all_sample_list)):]

        for item in sample_list:
            item_path = os.path.join(root_dir, item, "tracts")
            
            if not os.path.isdir(item_path):
                continue
            
            all_trk_exist = check_all_trk_files_exist(item_path, expected_tracts)
            
            if not all_trk_exist:
                continue
            
            out_sample_dir = os.path.join(output_dir, modal, item)
            
            features, labels = process_sample(
                item_path, 
                label_mapping, 
                out_sample_dir,
                num_points=15
            )
            
    print(f"\n{'='*80}")
    print("All samples have been processed.!")
    print(f"{'='*80}")

if __name__ == "__main__":
    root_dir = ""  
    json_path = "./global_label_mapping.json"
    output_dir = ""
    
    process_all_samples(root_dir, output_dir, json_path)
