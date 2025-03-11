
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
import smogn

from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor 
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import moscot
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
import ot  

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



def aggregate_predictions_by_clonotype(y_pred):
    """
    Aggregate responses by clonotype.

    Args:
        y_pred (array-like): Responses (predicted) for each cell.

    Returns:
        dict: A dictionary mapping each clonotype to its predicted responses.
    """
    test_clonotype_labels = unique_clonotypes_aggregated[test_mask]  # Get the clonotype labels for the test set


    # Create an empty dictionary to store predictions for each clonotype
    predictions_by_clonotype = {}

    # Iterate over each unique clonotype in the test set
    for clonotype in np.unique(test_clonotype_labels):
        # Get indices of the test samples corresponding to this clonotype
        clonotype_mask = test_clonotype_labels == clonotype

        # Get the predicted values for this clonotype
        clonotype_predictions = y_pred[clonotype_mask]

        # Store these predictions in the dictionary
        predictions_by_clonotype[clonotype] = clonotype_predictions

    return predictions_by_clonotype



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




def run_eval_S1(y_s1_test, y_s1_pred, k):
    # Evaluate S1 predictions
    mse_s1 = mean_squared_error(y_s1_test, y_s1_pred)
    mse_s1_clonotype, mse_s1_mean = mse_clonotype(y_s1_test, y_s1_pred)

    recall_ks1 = recall_at_k_clonotype(
        y_s1_test,
        y_s1_pred,
        k
    )
    mean_clonotype_activation_s1_predicted = y_s1_pred.mean()  # Predicted mean clonotype activation for the test set
    mean_clonotype_activation_s1 = y_s1_test.mean()# True mean clonotype activation for the test set

    print(f"Recall@{k} for S1: {recall_ks1:.2f}")
    print(f"MSE- Mean for S1: {mse_s1}")
    # print(f"MSE- Clonotype mean for S1: {mse_s1_mean}")
    print(f"Mean Clonotype Activation (S1): {mean_clonotype_activation_s1:.4f}")
    print(f"Mean Clonotype Activation (S1) Predicted: {mean_clonotype_activation_s1_predicted:.4f}")

    print(f"{recall_ks1:.2f}")
    print(f"{mse_s1:.4f}")
    print(f"{mean_clonotype_activation_s1:.4f}")
    print(f"{mean_clonotype_activation_s1_predicted:.4f}")

def run_eval_T1(y_t1_test, y_t1_pred, k):
    # Evaluate T1 predictions
    mse_t1 = mean_squared_error(y_t1_test, y_t1_pred)
    mse_t1_clonotype, mse_t1_mean = mse_clonotype(y_t1_test, y_t1_pred)
    recall_kt1 = recall_at_k_clonotype(
        y_t1_test,
        y_t1_pred,
        k
    )
    mean_clonotype_activation_t1 = y_t1_test.mean()
    mean_clonotype_activation_t1_predicted = y_t1_pred.mean()

    print(f"Recall@{k} for T1: {recall_kt1:.2f}")
    print(f"MSE- Mean for T1: {mse_t1}")
    # print(f"MSE- Clonotype mean for T1: {mse_t1_mean}")
    print(f"Mean Clonotype Activation (T1): {mean_clonotype_activation_t1:.4f}")
    print(f"Mean Clonotype Activation (T1) Predicted: {mean_clonotype_activation_t1_predicted:.4f}")

    print(f"{recall_kt1:.2f}")
    print(f"{mse_t1:.4f}")
    print(f"{mean_clonotype_activation_t1:.4f}")
    print(f"{mean_clonotype_activation_t1_predicted:.4f}")


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

oversampling = True

for k in range(0,7):
    adata = adata_list[k]
    print(embeddings[k])

    # filter out cells that we can't use
    adata = adata[~adata.obs.clone_id.isna()]
    adata = adata[~(adata.obs.clone_id=='nan')]
    adata = adata[~(adata.obs.time.isin(['X3','extra']))]

    # Filter for time points
    P1_mask = adata.obs["time"] == "P1"
    S1_mask = adata.obs["time"] == "S1"
    T1_mask = adata.obs["time"] == "T1"

    # Features: Embeddings from P1
    X_p1 = adata[P1_mask].obsm[embeddings[k]]
    X_s1 = adata[S1_mask].obsm[embeddings[k]]
    X_t1 = adata[T1_mask].obsm[embeddings[k]]

    # get target variable and clones from S1 and T1
    S1_clones = adata[S1_mask].obs["clone_id"].values  # Clonotypes at S1
    S1_targets = adata[S1_mask].obs["IFN Response_score"].values # target variable

    T1_clones = adata[T1_mask].obs["clone_id"].values  # Clonotypes at T1
    T1_targets = adata[T1_mask].obs["IFN Response_score"].values # target variable

    P1_clones = adata[P1_mask].obs["clone_id"].values
    P1_targets = adata[P1_mask].obs["IFN Response_score"].values  


    # Map S1 and T1 responses to corresponding P1 clones to ensure that only clones with corresponding targets are included
    """s1 = np.array([S1_targets[np.where(S1_clones == cid)[0][0]] if cid in S1_clones else np.nan for cid in P1_clones])
    t1 = np.array([T1_targets[np.where(T1_clones == cid)[0][0]] if cid in T1_clones else np.nan for cid in P1_clones])
    p1 = np.array([P1_targets[np.where(P1_clones == cid)[0][0]] if cid in P1_clones else np.nan for cid in P1_clones])"""

    s1 = np.array([np.mean(S1_targets[np.where(S1_clones == cid)])
                if cid in S1_clones else np.nan for cid in P1_clones])

    t1 = np.array([np.mean(T1_targets[np.where(T1_clones == cid)])
                if cid in T1_clones else np.nan for cid in P1_clones])

    p1 = np.array([np.mean(P1_targets[np.where(P1_clones == cid)])
                if cid in P1_clones else np.nan for cid in P1_clones])

    # Remove clones without corresponding targets
    valid_indices = ~np.isnan(s1) & ~np.isnan(t1)& ~np.isnan(p1)
    X_p1, s1, t1, p1 = X_p1[valid_indices], s1[valid_indices], t1[valid_indices], p1[valid_indices]
    P1_clones = P1_clones[valid_indices]
    unique_clonotypes_aggregated = np.unique(P1_clones) # Unique clonotypes in the aggregated data


    # X_aggregated (features), P1_clones (clonotype labels), 
    # y_s1_aggregated (true target values for S1), y_p1_aggregated (true target values for P1),
    # y_s1_pred (predicted values for each cell), and y_p1_pred (predictions for each cell).
    # Recreate aggregated arrays aligned with unique clonotypes
    X_aggregated = []  # features
    y_s1_aggregated = [] # target values for s1
    y_t1_aggregated = []
    y_p1_aggregated = []
    X_s1_aggregated = []  # features S1 
    X_p1_aggregated = []  # features P1
    X_t1_aggregated = []  # features T1

    # Step 1: Split the clonotypes into train, test, and eval sets
    train_clonotypes, temp_clonotypes = train_test_split(unique_clonotypes_aggregated, test_size=0.3, random_state=42)

    # Then split temp_clonotypes into test and eval (50% each)
    test_clonotypes, eval_clonotypes = train_test_split(temp_clonotypes, test_size=0.5, random_state=42)

    # Create masks for train, test, and eval sets based on clonotypes
    train_mask = np.isin(unique_clonotypes_aggregated, train_clonotypes)
    test_mask = np.isin(unique_clonotypes_aggregated, test_clonotypes)
    eval_mask = np.isin(unique_clonotypes_aggregated, eval_clonotypes)

    # Step 2: 
    # Loop to aggregate the features for each clonotype
    for clone in unique_clonotypes_aggregated:
        mask = P1_clones == clone  # Select rows matching the current clone for P1
        X_p1_aggregated.append(X_p1[mask].mean(axis=0))  # Mean of features for P1
        y_s1_aggregated.append(s1[mask].mean())      # Mean of s1 for the clone
        y_t1_aggregated.append(t1[mask].mean())      # Mean of t1 for the clone
        y_p1_aggregated.append(p1[mask].mean())      # Mean of p1 for the clone

        mask = S1_clones == clone  # Select rows matching the current clone for S1
        X_s1_aggregated.append(X_s1[mask].mean(axis=0))  # Mean of features for S1

        mask = T1_clones == clone  # Select rows matching the current clone for T1
        X_t1_aggregated.append(X_t1[mask].mean(axis=0))  # Mean of features for T1


    # Convert lists to arrays
    X_aggregated = np.array(X_aggregated)
    y_s1_aggregated = np.array(y_s1_aggregated)
    y_t1_aggregated = np.array(y_t1_aggregated)
    y_p1_aggregated = np.array(y_p1_aggregated)
    X_s1_aggregated = np.array(X_s1_aggregated)
    X_t1_aggregated = np.array(X_t1_aggregated)
    X_p1_aggregated = np.array(X_p1_aggregated)

    # Split aggregated data into train and test sets
    #X_train, X_test = X_aggregated[train_mask], X_aggregated[test_mask]
    X_p1_train, X_p1_test = X_p1_aggregated[train_mask], X_p1_aggregated[test_mask]
    X_s1_train, X_s1_test = X_s1_aggregated[train_mask], X_s1_aggregated[test_mask]
    X_t1_train, X_t1_test = X_t1_aggregated[train_mask], X_t1_aggregated[test_mask]
    y_s1_train, y_s1_test = y_s1_aggregated[train_mask], y_s1_aggregated[test_mask]
    y_t1_train, y_t1_test = y_t1_aggregated[train_mask], y_t1_aggregated[test_mask]
    y_p1_train, y_p1_test = y_p1_aggregated[train_mask], y_p1_aggregated[test_mask]

    X_s1_test = np.hstack([X_p1_test, X_s1_test])
    X_t1_test = np.hstack([X_p1_test, X_t1_test])


    if oversampling:
        # Define feature names for creating DataFrames
        k_smogn = 9
        samp_method_smogn = 'extreme'
        rel_thres_smogn = 0.5

        rg_mtrx = [
            [0.0, 0, 0],    # At score 0.0, relevance is 0 (not minority)
            [0.5, 0, 0],   # Up to 0.59, relevance stays 0
            [0.50001, 1, 0],    # At 0.6, relevance jumps to 1 (minority)
            [3, 1, 0]     # Up to 1.0, relevance remains 1
        ]
        feature_names = [f'feature_{i}' for i in range(X_p1_train.shape[1])]


        ### --- For P1 Predictions ---

        # Create DataFrame for P1 data
        df_p1 = pd.DataFrame(X_p1_train, columns=feature_names)
        df_p1['target'] = y_p1_train
        df_p1 = df_p1.dropna()

        # Apply SMOGN oversampling for regression on P1 data.
        # Adjust the parameters (e.g., rel_thres) based on your data distribution.
        df_p1_resampled = smogn.smoter(
            data=df_p1,
            y='target',
            k=k_smogn,
            samp_method=samp_method_smogn,  # oversample the extreme target values
            rel_thres=rel_thres_smogn,          # relevance threshold (tune this parameter)
            rel_method='manual',      # let SMOGN automatically compute relevance scores
            drop_na_row=True,
            rel_ctrl_pts_rg = rg_mtrx
        )

        # Separate features and target for S1
        X_p1_resampled = df_p1_resampled.drop(columns=['target']).values
        y_p1_resampled = df_p1_resampled['target'].values


        # Define feature names for creating DataFrames
        feature_names = [f'feature_{i}' for i in range(X_s1_train.shape[1])]


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


        # Define feature names for creating DataFrames
        feature_names = [f'feature_{i}' for i in range(X_t1_train.shape[1])]


        ### --- For S1 Predictions ---

        # Create DataFrame for S1 data
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


        X_p1_train = X_p1_resampled
        X_s1_train = X_s1_resampled
        X_t1_train = X_t1_resampled

        y_p1_train = y_p1_resampled
        y_s1_train = y_s1_resampled
        y_t1_train = y_t1_resampled


    # Ot pairing for P1 and S1
    # Extract target IFN response scores
    y_p1 = y_p1_train
    y_s1 = y_s1_train

    # Extract PCA embeddings and target values
    X_p1 = X_p1_train
    X_s1 = X_s1_train

    # Compute cost matrix and solve OT problem
    cost_matrix_s1 = ot.dist(X_p1, X_s1, metric='euclidean')
    a_s1 = np.ones(len(X_p1)) / len(X_p1)
    b_s1 = np.ones(len(X_s1)) / len(X_s1)
    transport_plan_s1 = ot.emd(a_s1, b_s1, cost_matrix_s1)

    # Extract paired indices
    paired_indices_s1 = np.argwhere(transport_plan_s1 > 0)
    paired_p1_indices_s1 = paired_indices_s1[:, 0]
    paired_s1_indices = paired_indices_s1[:, 1]

    # Prepare training data
    paired_features_p1_s1 = X_p1[paired_p1_indices_s1]
    paired_features_s1 = X_s1[paired_s1_indices]
    paired_targets_s1 = y_s1[paired_s1_indices]

    X_s1_train = np.hstack([paired_features_p1_s1, paired_features_s1])
    y_s1_train = paired_targets_s1 


    weights_s1 = np.where(y_s1_train > 0.4, 1, 1) 

    # Train the model
    model_lr = LinearRegression()
    model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    model_lr.fit(X_s1_train, y_s1_train, sample_weight=weights_s1)
    model_gb.fit(X_s1_train, y_s1_train, sample_weight=weights_s1)
    model_rf.fit(X_s1_train, y_s1_train, sample_weight=weights_s1)

    # Test the model
    y_s1_pred_lr = model_lr.predict(X_s1_test)
    y_s1_pred_gb = model_gb.predict(X_s1_test)
    y_s1_pred_rf = model_rf.predict(X_s1_test)

    # Ot pairing for P1 and S1
    # Extract target IFN response scores
    y_p1 = y_p1_train
    y_t1 = y_t1_train

    # Extract PCA embeddings and target values
    X_p1 = X_p1_train
    X_t1 = X_t1_train

    # Compute cost matrix and solve OT problem
    cost_matrix_t1 = ot.dist(X_p1, X_t1, metric='euclidean')
    a_t1 = np.ones(len(X_p1)) / len(X_p1)
    b_t1 = np.ones(len(X_t1)) / len(X_t1)
    transport_plan_t1 = ot.emd(a_t1, b_t1, cost_matrix_t1)

    # Extract paired indices
    paired_indices_t1 = np.argwhere(transport_plan_t1 > 0)
    paired_p1_indices_t1 = paired_indices_t1[:, 0]
    paired_t1_indices = paired_indices_t1[:, 1]

    # Prepare training data
    paired_features_p1_t1 = X_p1[paired_p1_indices_t1]
    paired_features_t1 = X_s1[paired_t1_indices]
    paired_targets_t1 = y_s1[paired_t1_indices]

    X_t1_train = np.hstack([paired_features_p1_t1, paired_features_t1])
    y_t1_train = paired_targets_t1 



    weights_t1 = np.where(y_t1_train > 0.4, 1, 1) 


    # Train the model
    model_lr = LinearRegression()
    model_gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)

    # Fit the model
    model_lr.fit(X_t1_train, y_t1_train, sample_weight=weights_t1)
    model_gb.fit(X_t1_train, y_t1_train, sample_weight=weights_t1)
    model_rf.fit(X_t1_train, y_t1_train, sample_weight=weights_t1)

    # Test the model
    y_t1_pred_lr = model_lr.predict(X_t1_test)
    y_t1_pred_gb = model_gb.predict(X_t1_test)
    y_t1_pred_rf = model_rf.predict(X_t1_test)

    k = 5  # Recall@k setting

    # --- Standard Evaluation ---
    # Evaluate S1 predictions
    mse_s1 = mean_squared_error(y_s1_test, y_s1_pred_lr)
    mse_s1_clonotype, mse_s1_mean = mse_clonotype(y_s1_test, y_s1_pred_lr)
    recall_ks1 = recall_at_k_clonotype(y_s1_test, y_s1_pred_lr, k)

    mean_clonotype_activation_s1_predicted = y_s1_pred_lr.mean()
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
    mse_s1_low = mean_squared_error(y_s1_test[q1_s1], y_s1_pred_lr[q1_s1])
    mse_s1_high = mean_squared_error(y_s1_test[q4_s1], y_s1_pred_lr[q4_s1])

    print(f"MSE for bottom 25% (low-responding clones) in S1: {mse_s1_low:.4f}")
    print(f"MSE for top 25% (high-responding clones) in S1: {mse_s1_high:.4f}")

    print(q1_s1)
    print(q4_s1)

    # --- Standard Evaluation for T1 ---
    mse_t1 = mean_squared_error(y_t1_test, y_t1_pred_lr)
    mse_t1_clonotype, mse_t1_mean = mse_clonotype(y_t1_test, y_t1_pred_lr)
    recall_kt1 = recall_at_k_clonotype(y_t1_test, y_t1_pred_lr, k)

    mean_clonotype_activation_t1 = y_t1_test.mean()
    mean_clonotype_activation_t1_predicted = y_t1_pred_lr.mean()

    print(f"Recall@{k} for T1: {recall_kt1:.2f}")
    print(f"MSE- Mean for T1: {mse_t1}")
    print(f"MSE- Clonotype mean for T1: {mse_t1_mean}")
    print(f"Mean Clonotype Activation (T1): {mean_clonotype_activation_t1:.4f}")
    print(f"Mean Clonotype Activation (T1) Predicted: {mean_clonotype_activation_t1_predicted:.4f}")

    # --- Quartile MSE Analysis for T1 ---
    sorted_indices_t1 = np.argsort(y_t1_test)

    q1_t1 = sorted_indices_t1[: len(sorted_indices_t1) // 4]  # Bottom 25%
    q4_t1 = sorted_indices_t1[-len(sorted_indices_t1) // 4:]  # Top 25%

    mse_t1_low = mean_squared_error(y_t1_test[q1_t1], y_t1_pred_lr[q1_t1])
    mse_t1_high = mean_squared_error(y_t1_test[q4_t1], y_t1_pred_lr[q4_t1])

    print(f"MSE for bottom 25% (low-responding clones) in T1: {mse_t1_low:.4f}")
    print(f"MSE for top 25% (high-responding clones) in T1: {mse_t1_high:.4f}")


    k = 5  # Recall@k setting

    # --- Standard Evaluation ---
    # Evaluate S1 predictions
    mse_s1 = mean_squared_error(y_s1_test, y_s1_pred_gb)
    mse_s1_clonotype, mse_s1_mean = mse_clonotype(y_s1_test, y_s1_pred_gb)
    recall_ks1 = recall_at_k_clonotype(y_s1_test, y_s1_pred_gb, k)

    mean_clonotype_activation_s1_predicted = y_s1_pred_gb.mean()
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
    mse_s1_low = mean_squared_error(y_s1_test[q1_s1], y_s1_pred_gb[q1_s1])
    mse_s1_high = mean_squared_error(y_s1_test[q4_s1], y_s1_pred_gb[q4_s1])

    print(f"MSE for bottom 25% (low-responding clones) in S1: {mse_s1_low:.4f}")
    print(f"MSE for top 25% (high-responding clones) in S1: {mse_s1_high:.4f}")

    print(q1_s1)
    print(q4_s1)

    # --- Standard Evaluation for T1 ---
    mse_t1 = mean_squared_error(y_t1_test, y_t1_pred_gb)
    mse_t1_clonotype, mse_t1_mean = mse_clonotype(y_t1_test, y_t1_pred_gb)
    recall_kt1 = recall_at_k_clonotype(y_t1_test, y_t1_pred_gb, k)

    mean_clonotype_activation_t1 = y_t1_test.mean()
    mean_clonotype_activation_t1_predicted = y_t1_pred_gb.mean()

    print(f"Recall@{k} for T1: {recall_kt1:.2f}")
    print(f"MSE- Mean for T1: {mse_t1}")
    print(f"MSE- Clonotype mean for T1: {mse_t1_mean}")
    print(f"Mean Clonotype Activation (T1): {mean_clonotype_activation_t1:.4f}")
    print(f"Mean Clonotype Activation (T1) Predicted: {mean_clonotype_activation_t1_predicted:.4f}")

    # --- Quartile MSE Analysis for T1 ---
    sorted_indices_t1 = np.argsort(y_t1_test)

    q1_t1 = sorted_indices_t1[: len(sorted_indices_t1) // 4]  # Bottom 25%
    q4_t1 = sorted_indices_t1[-len(sorted_indices_t1) // 4:]  # Top 25%

    mse_t1_low = mean_squared_error(y_t1_test[q1_t1], y_t1_pred_gb[q1_t1])
    mse_t1_high = mean_squared_error(y_t1_test[q4_t1], y_t1_pred_gb[q4_t1])

    print(f"MSE for bottom 25% (low-responding clones) in T1: {mse_t1_low:.4f}")
    print(f"MSE for top 25% (high-responding clones) in T1: {mse_t1_high:.4f}")


    k = 5  # Recall@k setting

    # --- Standard Evaluation ---
    # Evaluate S1 predictions
    mse_s1 = mean_squared_error(y_s1_test, y_s1_pred_rf)
    mse_s1_clonotype, mse_s1_mean = mse_clonotype(y_s1_test, y_s1_pred_rf)
    recall_ks1 = recall_at_k_clonotype(y_s1_test, y_s1_pred_rf, k)

    mean_clonotype_activation_s1_predicted = y_s1_pred_rf.mean()
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
    mse_s1_low = mean_squared_error(y_s1_test[q1_s1], y_s1_pred_rf[q1_s1])
    mse_s1_high = mean_squared_error(y_s1_test[q4_s1], y_s1_pred_rf[q4_s1])

    print(f"MSE for bottom 25% (low-responding clones) in S1: {mse_s1_low:.4f}")
    print(f"MSE for top 25% (high-responding clones) in S1: {mse_s1_high:.4f}")

    print(q1_s1)
    print(q4_s1)

    # --- Standard Evaluation for T1 ---
    mse_t1 = mean_squared_error(y_t1_test, y_t1_pred_rf)
    mse_t1_clonotype, mse_t1_mean = mse_clonotype(y_t1_test, y_t1_pred_rf)
    recall_kt1 = recall_at_k_clonotype(y_t1_test, y_t1_pred_rf, k)

    mean_clonotype_activation_t1 = y_t1_test.mean()
    mean_clonotype_activation_t1_predicted = y_t1_pred_rf.mean()

    print(f"Recall@{k} for T1: {recall_kt1:.2f}")
    print(f"MSE- Mean for T1: {mse_t1}")
    print(f"MSE- Clonotype mean for T1: {mse_t1_mean}")
    print(f"Mean Clonotype Activation (T1): {mean_clonotype_activation_t1:.4f}")
    print(f"Mean Clonotype Activation (T1) Predicted: {mean_clonotype_activation_t1_predicted:.4f}")

    # --- Quartile MSE Analysis for T1 ---
    sorted_indices_t1 = np.argsort(y_t1_test)

    q1_t1 = sorted_indices_t1[: len(sorted_indices_t1) // 4]  # Bottom 25%
    q4_t1 = sorted_indices_t1[-len(sorted_indices_t1) // 4:]  # Top 25%

    mse_t1_low = mean_squared_error(y_t1_test[q1_t1], y_t1_pred_rf[q1_t1])
    mse_t1_high = mean_squared_error(y_t1_test[q4_t1], y_t1_pred_rf[q4_t1])

    print(f"MSE for bottom 25% (low-responding clones) in T1: {mse_t1_low:.4f}")
    print(f"MSE for top 25% (high-responding clones) in T1: {mse_t1_high:.4f}")


