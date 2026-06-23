# FAGN
Official implementation of "FAGN: Frenet-Serret Aware Graph Network for Whole-Brain Tractography Parcellation"

![model](FAGN.png)

>Tractography based on diffusion magnetic resonance imaging (dMRI) can generate a large number of whole-brain fiber streamlines. Performing tractography parcellation on these streamlines helps construct consistent representations of white matter structures, which is critical for understanding brain development and the progression of neuropsychiatric disorders. In recent years, point cloud-based learning methods have gradually become the mainstream paradigm for tractography parcellation. However, existing methods rely solely on the Euclidean coordinates of points, treating streamlines as simple point sequences. These approaches struggles to fully capture intrinsic geometric properties of white matter fibers, such as local curvature, directional changes, and spatial torsion, and often overlooks the local structural relationships along the continuous fiber trajectory. To address these challenges, this study introduces a fiber representation method based on the Frenet-Serret Frame from differential geometry, enabling explicit modeling of the intrinsic geometric structure of fiber streamlines. Moreover, we propose the Frenet-Serret Aware Graph Network (FAGN), which not only extracts global features of fibers but also models the point-to-point geometric relationship within local neighborhoods. Experiments on public HCP and in-house CLAS datasets demonstrate that the proposed method outperforms existing approaches for parcellation of 72 fiber classes in whole-brain, validating the effectiveness and generalizability of both the Frenet-Serret Frame representation and the FAGN architecture.

## Requirements
```bash
conda env create -f environment.yml
```

## Dataset
- Human Connectome Project (HCP) dataset
- China Longitudinal Aging Study (CLAS) dataset

The HCP dataset can be downloaded in [HCP](https://humanconnectome.org/study/hcp-young-adult/data-use-terms).

## Frenet-Serret Representation
We use the `FrameNet_representation_from_trk.py` script to process the raw .trk files. The pipeline consists of two main steps:  
(1) Resampling: each fier tract is resampled to a fixed number of points;  
(2) Frenet–Serret representation: for each point, we compute and append three geometric vectors—the tangent (T), normal (N), and binormal (B).  

As a result, we obtain a feature matrix of shape (N, P, 12) (where N is the number of samples and P is the fixed number of points) along with corresponding label vectors of shape (N,).  
The current implementation is built upon the directory structure of the HCP dataset. Although the code can be extended to other datasets, users must ensure that the folder organization follows the same format as HCP, i.e.:

```dataset
dataset/
├── sample1/
│   ├── UF_right.trk
│   ├── UF_left.trk
│   └── ...
├── sample2/
│   ├── UF_right.trk
│   ├── UF_left.trk
│   └── ...
└── ...
```

After processing, point-cloud features for each sample are saved in .h5 format and organized into training, validation, and test sets as follows:

```dataset
dataset/
├── train/
│   ├── sample1/
│   │   ├── feature.h5
│   │   └── label.h5
│   ├── sample2/
│   │   ├── feature.h5
│   │   └── label.h5
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

## Training 
Run the following command to start training:
```bash
python train.py --experiment_name FAGN --device 0 --num_classes 72
```
## Testing
```bash
python predict.py --experiment_name FAGN --device 0 --num_classes 72
```
