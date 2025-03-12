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
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import moscot
from sklearn.metrics import mean_squared_error, r2_score
import smogn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

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
    test_clonotype_labels = unique_clonotypes[test_mask]  # Get the clonotype labels for the test set


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
    test_clonotype_labels = unique_clonotypes[test_mask]  # Make sure test_mask is defined
     
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
linreg = False
gbr = False
rf = True


for k in range(0,7):
    adata = adata_list[k]
    # filter out cells that we can't use
    adata = adata[~adata.obs.clone_id.isna()]
    adata = adata[~(adata.obs.clone_id=='nan')]
    adata = adata[~(adata.obs.time.isin(['X3','extra']))]


    # Filter for time points
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


    # Map S1 and T1 responses to corresponding P1 clones to ensure that only clones with corresponding targets are included
    s1 = np.array([np.mean(S1_targets[np.where(S1_clones == cid)])
                if cid in S1_clones else np.nan for cid in P1_clones])

    t1 = np.array([np.mean(T1_targets[np.where(T1_clones == cid)])
                if cid in T1_clones else np.nan for cid in P1_clones])


    # Remove clones without corresponding targets
    valid_indices_s1 = ~np.isnan(s1)
    valid_indices_t1 = ~np.isnan(t1)
    valid_indices = valid_indices_s1 & valid_indices_t1

    X, s1, t1 = X[valid_indices], s1[valid_indices], t1[valid_indices]
    P1_clones = P1_clones[valid_indices]


    # get cells per clonotype
    clonotype_cell_counts = pd.DataFrame(P1_clones, columns=["Clonotype"]).value_counts().reset_index()
    clonotype_cell_counts.columns = ["Clonotype", "Cell Count"]

    # Display the counts per clonotype
    clonotype_cell_counts


    # Extract donor information and unique clonotypes
    donors = adata.obs["donor"].values  # Donor IDs for each cell
    unique_clonotypes = np.unique(P1_clones)  # Unique clonotypes in P1

    # Map each clonotype to a donor
    clonotype_to_donor = {clone: donors[np.where(P1_clones == clone)[0][0]] for clone in unique_clonotypes}

    # Extract unique donors
    unique_donors = np.unique(list(clonotype_to_donor.values()))

    # Split donors into train and test
    train_donors, test_donors = train_test_split(unique_donors, test_size=0.2, random_state=42)

    # Create masks for train and test clonotypes based on donors
    train_clonotypes = [clone for clone, donor in clonotype_to_donor.items() if donor in train_donors]
    test_clonotypes = [clone for clone, donor in clonotype_to_donor.items() if donor in test_donors]

    # Aggregate features and targets by clone (for training)
    X_aggregated = []
    y_s1_aggregated = []
    y_t1_aggregated = []
    clonotype_list = []

    for clone in unique_clonotypes:
        mask = P1_clones == clone  # Select rows matching the current clone
        if np.any(mask):  # Ensure at least one match exists
            X_aggregated.append(X[mask].mean(axis=0))  # Mean of features
            y_s1_aggregated.append(np.mean(s1[mask]))  # Mean of S1 for the clone
            y_t1_aggregated.append(np.mean(t1[mask]))  # Mean of T1 for the clone
            clonotype_list.append(clone)  # Store clonotype labels

    # Convert lists to numpy arrays
    X_aggregated = np.array(X_aggregated)
    y_s1_aggregated = np.array(y_s1_aggregated)
    y_t1_aggregated = np.array(y_t1_aggregated)
    clonotype_list = np.array(clonotype_list)

    # Create masks for train and test sets based on donors
    train_mask = np.isin(clonotype_list, train_clonotypes)
    test_mask = np.isin(clonotype_list, test_clonotypes)

    # Train/Test Split (Aggregated Data)
    X_train, X_test = X_aggregated[train_mask], X_aggregated[test_mask]
    y_s1_train, y_s1_test = y_s1_aggregated[train_mask], y_s1_aggregated[test_mask]
    y_t1_train, y_t1_test = y_t1_aggregated[train_mask], y_t1_aggregated[test_mask]

    X_train_s1 = X_train
    X_train_t1 = X_train


    if oversampling:
        # Define feature names for creating DataFrames
        feature_names = [f'feature_{i}' for i in range(X_train.shape[1])]


        k_smogn = 5
        samp_method_smogn = 'balance'
        rel_thres_smogn = 0.6

        rg_mtrx = [
            [0.0, 0, 0],    # At score 0.0, relevance is 0 (not minority)
            [0.45, 0, 0],   # Up to 0.59, relevance stays 0
            [0.4501, 1, 0],    # At 0.6, relevance jumps to 1 (minority)
            [3, 1, 0]     # Up to 1.0, relevance remains 1
        ]


        ### --- For S1 Predictions ---

        # Create DataFrame for S1 data
        df_s1 = pd.DataFrame(X_train, columns=feature_names)
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
        df_t1 = pd.DataFrame(X_train, columns=feature_names)
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



        X_train_s1 = X_s1_resampled
        y_s1_train = y_s1_resampled


        X_train_t1 = X_t1_resampled
        y_t1_train = y_t1_resampled

    # ## Train Models
    # For the first baseline model, a Linear Regression is used, since our goal is to predict a numerical and continuous output value  (=IFN score) based on the input features. One can furthermore tune the parameters to improve this model.   
    # Either use Linear Regression or GBR! 


    weights_s1 = np.where(y_s1_train > 0.5, 1, 1) 
    weights_t1 = np.where(y_t1_train > 0.5, 1, 1) 

    if linreg:
        # Train a model for S1 using Linear Regression
        model_s1 = LinearRegression()
        model_s1.fit(X_train_s1, y_s1_train, sample_weight=weights_s1)
        y_s1_pred = model_s1.predict(X_test)

        # Train a model for T1 using Linear Regression
        model_t1 = LinearRegression()
        model_t1.fit(X_train_t1, y_t1_train, sample_weight=weights_t1)
        y_t1_pred = model_t1.predict(X_test)

    if gbr:
        gbr_s1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gbr_s1.fit(X_train_s1, y_s1_train, sample_weight=weights_s1)
        y_s1_pred = gbr_s1.predict(X_test)


        gbr_t1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gbr_t1.fit(X_train_t1, y_t1_train, sample_weight=weights_t1)
        y_t1_pred = gbr_t1.predict(X_test)

    if rf:
        # Initialize and train the Random Forest Regressor for S1
        rf_s1 = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_s1.fit(X_train_s1, y_s1_train, sample_weight=weights_s1)
        y_s1_pred = rf_s1.predict(X_test)

        # Initialize and train the Random Forest Regressor for T1
        rf_t1 = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_t1.fit(X_train_t1, y_t1_train, sample_weight=weights_t1)
        y_t1_pred = rf_t1.predict(X_test)


    k = 5 # Recall@k setting

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
    # print(f"Mean Clonotype Activation (T1) Predicted: {mean_clonotype_activation_t1_predicted:.4f}")



