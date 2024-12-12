# HetereoGNN Anomaly Classification System

## Our code is split into multiple branches:
* **master**: CIDDS Dataset
* **UNSW-NB15**: UNSW_NB15 Dataset
* **few_shot**: few shot method where equal number of classified anomalies from each class are given to train the model
* **semisupervised**: small proportional subset of classified anomalies are given to train the model, use loss weighting to increase classification accuracy
* **smote**: Synthetic Minority Oversampling Technique on minority anomaly classes to increase classification accuracy

## master
run HetereoGNN on CIDDS-001 week 1 dataset
1. download the CIDDS-001 dataset [here](https://www.hs-coburg.de/forschen/cidds-coburg-intrusion-detection-data-sets/)
2. change `#SBATCH --mail-user=cczeng@umich.edu` in `cmd_slurm_code_589.sh` to corresponding uniqname
3. run `cmd_slurm_code_589.sh` on GreatLakes

## UNSW-NB15
run HetereoGNN on UNSW-NB15 dataset
1. download the UNSW-NB15 dataset [here](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
2. change `#SBATCH --mail-user=cczeng@umich.edu` in `cmd_slurm_code_589.sh` to corresponding uniqname
3. run `cmd_slurm_code_589.sh` on GreatLakes

## few_shot
run HetereoGNN on a equal subset of the CIDDS-001 dataset\
our code currently trains on a subset of 5 examples for each anomaly class
1. download the CIDDS-001 dataset [here](https://www.hs-coburg.de/forschen/cidds-coburg-intrusion-detection-data-sets/)
2. change `#SBATCH --mail-user=mitchang@umich.edu` in `few_shot_589.sh` to corresponding uniqname
3. run `few_shot_589.sh` on GreatLakes

## semisupervised
run HetereoGNN on a proportional subset of the CIDDS-001 dataset, applies weighted loss function to increase testing classification accuracy on minority classes
1. download the CIDDS-001 dataset [here](https://www.hs-coburg.de/forschen/cidds-coburg-intrusion-detection-data-sets/)
2. change `#SBATCH --mail-user=cczeng@umich.edu` in each `sh` to corresponding uniqname

### 4128 samples of CIDDS-001 dataset
* with weighted loss: run `CIDDS_weights_4128.sh` on GreatLakes
* without weighted loss: run `CIDDS_no_weights_4128.sh` on GreatLakes

### 2064 samples of CIDDS-001 dataset
* with weighted loss: run `CIDDS_weights_2064.sh` on GreatLakes
* without weighted loss: run `CIDDS_no_weights_2064.sh` on GreatLakes

## smote
run HetereoGNN on a equal subset on the CIDDS-001 dataset, applies SMOTE to increase testing classification accuracy on minority classes\
our code currently trains on a subset of 10 examples with Synthetic Minority Oversampling Technique (SMOTE) for each anomaly class 
1. download the CIDDS-001 dataset [here](https://www.hs-coburg.de/forschen/cidds-coburg-intrusion-detection-data-sets/)
2. change `#SBATCH --mail-user=mitchang@umich.edu` in `smote.sh` to corresponding uniqname
3. run `smote.sh` on GreatLakes

## package dependencies
```
# Data manipulation and visualization
pip3 install numpy pandas matplotlib

# Machine learning
pip3 install scikit-learn

# PyTorch (with CUDA support)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# PyTorch Geometric
pip3 install torch-geometric

# Additional PyTorch Geometric dependencies
pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
```
