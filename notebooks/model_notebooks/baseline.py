# # Notebook for the first baseline model 
# Starting with PCA embeddings of RNA, using Linear Regression to predict how a clone responses to vaccination in a late time point, using data from P1 (initial time point) to predcit S1 and T1  
# Random pairing
# 

 
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

 
import pandas as pd
import numpy as np
import scirpy as ir
import anndata as ad
import scanpy as sc
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import smogn
import matplotlib.pyplot as plt
import moscot
from sklearn.metrics import mean_squared_error, r2_score

# ## Get data & explore data

 
adata_PCA = sc.read_h5ad('../../../data/02_dex_annotated_cd8_PCA.h5ad')
adata_deepTCR = sc.read_h5ad('../../../data/deepTCR/02_dex_annotated_cd8_deepTCR_VAE_dim64.h5ad')
adata_mvTCR = sc.read_h5ad('../../../data/mvTCR/02_dex_annotated_cd8_mvTCR_leidenCD8_and_Responder_pp_c_donor_batch_experiment_sample_e50_con_None.h5ad')
adata_mvTCR.obsm['X_mvTCR'] = adata_mvTCR.X
adata_deepTCR_PCA = sc.read_h5ad('../../../data/combined/02_dex_annotated_cd8_deepTCR_PCA.h5ad')
adata_mvTCR_deepTCR = sc.read_h5ad('../../../data/combined/02_dex_annotated_cd8_mvTCR_deepTCR.h5ad')
adata_mvTCR_PCA = sc.read_h5ad('../../../data/combined/02_dex_annotated_cd8_mvTCR_PCA.h5ad')
adata_mvTCR_deepTCR_PCA = sc.read_h5ad('../../../data/combined/02_dex_annotated_cd8_mvTCR_deepTCR_PCA.h5ad')


adata_list = [adata_PCA, adata_deepTCR, adata_mvTCR, adata_deepTCR_PCA, adata_mvTCR_deepTCR, adata_mvTCR_PCA, adata_mvTCR_deepTCR_PCA]
embeddings = ['X_pca', 'deepTCR_VAE_dim64', 'X_mvTCR', 'emb_deepTCR_PCA', 'emb_mvTCR_deepTCR', 'emb_mvTCR_PCA', 'emb_mvTCR_deepTCR_PCA']

oversampling = False

def recall_at_k_clonotype(y_true, y_pred, k):
    """
    Compute recall at K based on clonotype-level responsiveness.

    Args:
        y_true (array-like): True responses for each cell.
        y_pred (array-like): Predicted responses for each cell.
        k (int): Number of top clonotypes to consider.

    Returns:
        float: Recall at K based on clonotypes.

        
    """
    
    # Aggregate responses by clonotype
    true_responses_by_clonotype = aggregate_predictions_by_clonotype(y_true)
    pred_responses_by_clonotype = aggregate_predictions_by_clonotype(y_pred)

    # Sort clonotypes by their average true and predicted responses -> # TODO why? 
    top_k_true_clonotypes = sorted(
        true_responses_by_clonotype, key=true_responses_by_clonotype.get, reverse=True
    )[:k]
    top_k_pred_clonotypes = sorted(
        pred_responses_by_clonotype, key=pred_responses_by_clonotype.get, reverse=True
    )[:k]

    # Compute the intersection of top-k true and predicted clonotypes
    intersection = set(top_k_true_clonotypes) & set(top_k_pred_clonotypes)

    # Compute Recall@K
    recall_k = len(intersection) / k
    return recall_k


def mse_clonotype(y_pred, y_true):
    """
    Aggregate responses by clonotype and calculate MSE per clonotype.

    Args:
        y_pred (array-like): Predicted responses for each cell.
        y_true (array-like): True responses for each cell.

    Returns:
        dict: A dictionary mapping each clonotype to its predicted responses and MSE.
        float: The mean MSE across all clonotypes.
    """

    # Get the clonotype labels for the test set (assumed to be pre-defined)
    test_clonotype_labels = unique_clonotypes_aggregated[test_mask]  # Make sure test_mask is defined
     
    # Create an empty dictionary to store predictions and MSE for each clonotype
    predictions_by_clonotype = {}
    mse_by_clonotype = {}

    # Iterate over each unique clonotype in the test set
    for clonotype in np.unique(test_clonotype_labels):
        # Get indices of the test samples corresponding to this clonotype
        clonotype_mask = test_clonotype_labels == clonotype

        # Get the predicted values and true values for this clonotype
        clonotype_predictions = y_pred[clonotype_mask]
        clonotype_true_values = y_true[clonotype_mask]

        # Calculate the MSE for this clonotype
        mse = mean_squared_error(clonotype_true_values, clonotype_predictions)

        # Store predictions and MSE in the dictionary
        predictions_by_clonotype[clonotype] = clonotype_predictions
        mse_by_clonotype[clonotype] = mse

    # Calculate the mean MSE across all clonotypes
    mean_mse = np.mean(list(mse_by_clonotype.values()))

    return mse_by_clonotype, mean_mse

def recall_at_k(y_true, y_pred, k):
    # Get the indices of the top-k predicted scores
    top_k_pred_indices = np.argsort(y_pred)[-k:]  # Indices of top-k predicted scores
    # Get the indices of the top-k true scores
    top_k_true_indices = np.argsort(y_true)[-k:]  # Indices of top-k true scores

    # Compute intersection of predicted and true top-k indices
    intersection = np.intersect1d(top_k_pred_indices, top_k_true_indices)

    # Compute Recall@k
    recall_k = len(intersection) / k
    return recall_k

# # Clone Aggregation and Pairing:
# For each clone early and late time point data is paired.
# Not all clones in P1 might exist in S1 or T1. Here, we ensure that only clones with valid response scores are included in the model training.
for k in range(0,7):
    adata = adata_list[k]

    adata = adata[~adata.obs.clone_id.isna()]
    adata = adata[~(adata.obs.clone_id=='nan')]
    adata = adata[~(adata.obs.time.isin(['X3','extra']))]

    P1_mask = adata.obs["time"] == "P1"
    S1_mask = adata.obs["time"] == "S1"
    T1_mask = adata.obs["time"] == "T1"

    # Features: Embeddings from P1
    X = adata[P1_mask].obsm[embeddings[k]]

    # get target variable and clones from S1 and T1
    S1_clones = adata[S1_mask].obs["clone_id"].values  # Clonotypes at S1
    S1_targets = adata[S1_mask].obs["IFN Response_score"].values # target variable

    T1_clones = adata[T1_mask].obs["clone_id"].values  # Clonotypes at T1
    T1_targets = adata[T1_mask].obs["IFN Response_score"].values # target variable

    P1_clones = adata[P1_mask].obs["clone_id"].values
    P1_targets = adata[P1_mask].obs["IFN Response_score"].values    

    unique_clonotypes_aggregated = np.unique(P1_clones) # Unique clonotypes in the aggregated data

    random_indices_s1 = np.random.permutation(len(S1_targets))
    random_indices_t1 = np.random.permutation(len(T1_targets))

    # Shuffle and align S1 and T1 responses with P1 clones randomly
    s1_random = S1_targets[random_indices_s1[:len(P1_clones)]]
    t1_random = T1_targets[random_indices_t1[:len(P1_clones)]]

    # Aggregate features for training
    X_aggregated = []
    y_s1_aggregated = []
    y_t1_aggregated = []

    for clone in np.unique(P1_clones):
        mask = P1_clones == clone  # Select rows for the current clone
        X_aggregated.append(X[mask].mean(axis=0))  # Mean of features
        y_s1_aggregated.append(s1_random[mask].mean())  # Randomized S1 response
        y_t1_aggregated.append(t1_random[mask].mean())  # Randomized T1 response

    # Convert to arrays
    X_aggregated = np.array(X_aggregated)
    y_s1_aggregated_baseline = np.array(y_s1_aggregated)
    y_t1_aggregated_baseline = np.array(y_t1_aggregated)

    X_s1_train, X_s1_test, y_s1_train, y_s1_test = train_test_split(
        X_aggregated, y_s1_aggregated_baseline, test_size=0.2, random_state=42
    )

    X_t1_train, X_t1_test, y_t1_train, y_t1_test = train_test_split(
        X_aggregated, y_t1_aggregated_baseline, test_size=0.2, random_state=42
    )

    # # Oversampling

    if oversampling:
        # Define feature names for creating DataFrames
        feature_names = [f'feature_{i}' for i in range(X_s1_train.shape[1])]

        k_smogn = 5
        samp_method_smogn = 'extreme'
        rel_thres_smogn = 0.7

        rg_mtrx = [
            [0.0, 0, 0],    # At score 0.0, relevance is 0 (not minority)
            [0.5, 0, 0],   # Up to 0.59, relevance stays 0
            [0.50001, 1, 0],    # At 0.6, relevance jumps to 1 (minority)
            [3, 1, 0]     # Up to 1.0, relevance remains 1
        ]

        ### --- For S1 Predictions ---

        # Create DataFrame for S1 data
        df_s1 = pd.DataFrame(X_s1_train, columns=feature_names)
        df_s1['target'] = y_s1_train
        df_s1 = df_s1.dropna()

        # Apply SMOGN oversampling for regression on S1 data.
        # Adjust the parameters (e.g., rel_thres) based on your data distribution.
        df_s1_resampled = smogn.smoter(
            data=df_s1,
            y='target',
            k=k_smogn,
            samp_method=samp_method_smogn,  # oversample the extreme target values
            rel_thres=rel_thres_smogn,          # relevance threshold (tune this parameter)
            rel_method='manual',      # let SMOGN automatically compute relevance scores
            drop_na_row=True,
            rel_ctrl_pts_rg = rg_mtrx
        )

        # Separate features and target for S1
        X_s1_resampled = df_s1_resampled.drop(columns=['target']).values
        y_s1_resampled = df_s1_resampled['target'].values

        ### --- For T1 Predictions ---

        # Create DataFrame for T1 data
        df_t1 = pd.DataFrame(X_t1_train, columns=feature_names)
        df_t1['target'] = y_t1_train
        df_t1 = df_t1.dropna()

        # Apply SMOGN oversampling for regression on T1 data.
        df_t1_resampled = smogn.smoter(
            data=df_t1,
            y='target',
            k=k_smogn,
            samp_method=samp_method_smogn,  # oversample the extreme target values
            rel_thres=rel_thres_smogn,          # relevance threshold (tune this parameter)
            rel_method='manual',      # let SMOGN automatically compute relevance scores
            drop_na_row=True,
            rel_ctrl_pts_rg = rg_mtrx
        )

        # Separate features and target for T1
        X_t1_resampled = df_t1_resampled.drop(columns=['target']).values
        y_t1_resampled = df_t1_resampled['target'].values


    # Train a regression model on the resampled S1 data
    model_s1 = LinearRegression()
    model_s1.fit(X_s1_train, y_s1_train)
    # Predict on the S1 test set
    y_s1_pred = model_s1.predict(X_s1_test) 

    # Train a regression model on the resampled T1 data
    model_t1 = LinearRegression()
    model_t1.fit(X_t1_train, y_t1_train)
    # Predict on the T1 test set
    y_t1_pred = model_t1.predict(X_t1_test)

    #Evaluation

    k = 5  # Recall@k setting

    # --- Standard Evaluation ---
    # Evaluate S1 predictions
    mse_s1 = mean_squared_error(y_s1_test, y_s1_pred)
    mse_s1_clonotype, mse_s1_mean = mse_clonotype(y_s1_test, y_s1_pred)
    recall_ks1 = recall_at_k_clonotype(y_s1_test, y_s1_pred, k)

    mean_clonotype_activation_s1_predicted = y_s1_pred.mean()
    mean_clonotype_activation_s1 = y_s1_test.mean()

    print(f"Recall@{k} for S1: {recall_ks1:.2f}")
    print(f"MSE- Mean for S1: {mse_s1}")
    print(f"MSE- Clonotype mean for S1: {mse_s1_mean}")
    print(f"Mean Clonotype Activation (S1): {mean_clonotype_activation_s1:.4f}")
    print(f"Mean Clonotype Activation (S1) Predicted: {mean_clonotype_activation_s1_predicted:.4f}")

    # --- Quartile MSE Analysis for S1 ---
    # Sort clones by true response level
    sorted_indices_s1 = np.argsort(y_s1_test)

    # Get quartile indices
    q1_s1 = sorted_indices_s1[: len(sorted_indices_s1) // 4]  # Bottom 25% (low responders)
    q4_s1 = sorted_indices_s1[-len(sorted_indices_s1) // 4:]  # Top 25% (high responders)

    # Compute MSE for the two quartiles
    mse_s1_low = mean_squared_error(y_s1_test[q1_s1], y_s1_pred[q1_s1])
    mse_s1_high = mean_squared_error(y_s1_test[q4_s1], y_s1_pred[q4_s1])

    print(f"MSE for bottom 25% (low-responding clones) in S1: {mse_s1_low:.4f}")
    print(f"MSE for top 25% (high-responding clones) in S1: {mse_s1_high:.4f}")


    # --- Standard Evaluation for T1 ---
    mse_t1 = mean_squared_error(y_t1_test, y_t1_pred)
    mse_t1_clonotype, mse_t1_mean = mse_clonotype(y_t1_test, y_t1_pred)
    recall_kt1 = recall_at_k_clonotype(y_t1_test, y_t1_pred, k)

    mean_clonotype_activation_t1 = y_t1_test.mean()
    mean_clonotype_activation_t1_predicted = y_t1_pred.mean()

    print(f"Recall@{k} for T1: {recall_kt1:.2f}")
    print(f"MSE- Mean for T1: {mse_t1}")
    print(f"MSE- Clonotype mean for T1: {mse_t1_mean}")
    print(f"Mean Clonotype Activation (T1): {mean_clonotype_activation_t1:.4f}")
    print(f"Mean Clonotype Activation (T1) Predicted: {mean_clonotype_activation_t1_predicted:.4f}")

    # --- Quartile MSE Analysis for T1 ---
    sorted_indices_t1 = np.argsort(y_t1_test)

    q1_t1 = sorted_indices_t1[: len(sorted_indices_t1) // 4]  # Bottom 25%
    q4_t1 = sorted_indices_t1[-len(sorted_indices_t1) // 4:]  # Top 25%

    mse_t1_low = mean_squared_error(y_t1_test[q1_t1], y_t1_pred[q1_t1])
    mse_t1_high = mean_squared_error(y_t1_test[q4_t1], y_t1_pred[q4_t1])

    print(f"MSE for bottom 25% (low-responding clones) in T1: {mse_t1_low:.4f}")
    print(f"MSE for top 25% (high-responding clones) in T1: {mse_t1_high:.4f}")



