# Temporal Image Sandwiches Enable Link between Functional Data Analysis and Deep Learning for Single-Plant Cotton Senescence

### This repository is meant to serve as a companion to a manuscript of the same name. It will be updated throughout the preprint and publication process.

### Link to the preprint: https://www.biorxiv.org/content/10.1101/2024.06.30.601428v1

### Scripts and associated files:
- R_Script_V8.R: R script used to run FPCA and ANOVA
  - df1.csv: used for FPCA of visual senescence scores, RCC, and TNDGR vegetation indices (VIs)
  - df2.csv: used for ANOVA of E1 visual senescence scores
  - df3.csv: used for ANOVA of E1 RCC and TNDGR VI values
  - M2_RCC_ReLU_Means_E1.csv, M4_Visual_Means_E1.csv, and M6_TNDGR_Means_E1.csv: used for ANOVA of FPC1 values that were output by CNN regression. To obtain these FPC1 scores, CNN regression was performed using a 50/50 train/test split within E1. Optuna-derived optimal hyperparameters were used to build the model. FPC1 scores were averaged across 25 replications.

- M1_M2_M3.py: Python script used to run the first set of three convolutional neural networks (CNNs). Here, Optuna was used for hyperparameter optimization using E1 data. After hyperparameters were saved, M1, M2, and M3 were evaluated by training each model with E1 data and testing on E2 data. These CNNs all have in common that the activation function for the first of two dense layers was always set to ReLU.
  - df4.csv: contains regression targets (FPC1 scores) for each concatenated time-series image (TSI, or "image sandwich").

- M4_M5_M6.py: Python script used to run the second set of three convolutional neural networks (CNNs). Here, Optuna was used for hyperparameter optimization using E1 data. After hyperparameters were saved, M4, M5, and M6 were evaluated by training each model with E1 data and testing on E2 data. These CNNs all have in common that the activation function for the first of two dense layers was searchable by Optuna (not set to ReLU as with M1-3).
  - df4.csv: contains regression targets (FPC1 scores) for each concatenated time-series image (TSI, or "image sandwich").
 
- Saliency_Maps.py: Python script meant to be pasted after line 464 of M1_M2_M3.py and M4_M5_M6.py. By selecting between the two indices provided in the script, activation/saliency maps can be produced for an example of a plant demonstrating stay-green or rapid senescence.

- Images.zip: contains all JPEGs necessary to run the Python scripts.
