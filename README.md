# IRMTraj: Maneuver-Aware Invariant Risk Minimization for Generalized Trajectory Prediction in Unseen Domain

This is the source code of our IRMTraj.

## Overview

<img src="./overview.png" style="zoom:30%;" />


## Highlights

1. We propose a **maneuver-aware generalized trajectory prediction** model that integrates map topological information and endpoint prediction. By finely partitioning domains into environments based on maneuver-type, we conduct transfer learning of generalized features within these environments and enhance feature learning through Self-supervised Maneuver-type Prediction tasks.

2. We introduce an **environment-aware IRM endpoint prediction** module. By using maneuver-type as a fine-grained environment and integrating it into the IRM framework, we enhance the generalization of endpoint prediction, which in turn improves the generalization of the subsequent complete trajectory prediction.

3. We propose a **spatial topological feature** extraction network for contextual map information modeling. This network enhances this by constructing a topological information extraction grid and identifying topologically related neighboring nodes for each road.

4. We designed cross-scenario and cross-map experiments targeting unseen domains in trajectory datasets, demonstrating that our method achieves **state-of-the-art** performance with a significant improvement up to 2.3 times.


## Implement



### 1. Requirements
Recommend version:
```
conda install python=3.10 pytorch=2.0.0 torchvision torchaudio pytorch-cuda=12.1 numpy scipy scikit-learn matplotlib pandas tqdm pyyaml opencv pip -c pytorch -c nvidia -c defaults

```

Other packages:
```
pip install Argoverse
pip install seaborn
```
### 2. Data Preparation
 
```
data/processed/ZS3s/train/
data/processed/ZS3s/val/
```
### 3. Training
To train IRMTraj on the new **cross-map and cross-scenario setting**, run the following command:

```
python train.py --train_data1 EP --train_data2 FT --test_data SR OF LN GL MA EP

```

For IRM ablation experiments (disable IRM loss), run

```
python train.py --train_data1 EP --train_data2 FT --test_data SR OF LN GL MA EP --l 0
```

### 5. Trajectory Visualization
After training, run the following to visualize predicted trajectories:  

```
python picture.py --trained_models
```

### 6. Logs and Checkpoints
Training logs are saved in the `log/` directory and model checkpoints are saved in the `trained_models/` directory.




Thank you for your interest in our project. If you have any questions, please feel free to contact us at yangfeng@seu.edu.cn. :-)
