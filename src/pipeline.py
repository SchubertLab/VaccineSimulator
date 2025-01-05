import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import rcParams
import seaborn as sns
import os
from tqdm import tqdm
import scanpy as sc
from sklearn.model_selection import train_test_split

def pipeline(data, embedding):
    if embedding == "pca":
        transformed_data = create_pca_embeddings()

    cell_paring(transformed_data)
    train_test_split(transformed_data)
    model_s1, model_t1 = train(transformed_data)

    

    return model_s1, model_t1, data

def create_pca_embeddings(data):
    sc.tl.pca(data.adata, n_comps=10) # Perform PCA -> use 10 components since they explain 90% of the variance (plot)
    sc.pl.pca(data.adata) # Plot only the top two principal components as they are most informative
    sc.pl.pca_variance_ratio(data.adata, n_pcs=10) 


def cell_paring(data):

    # filter out cells that we can't use
    adata = adata[~adata.obs.clone_id.isna()]
    adata = adata[~(adata.obs.clone_id=='nan')]
    adata = adata[~(adata.obs.time.isin(['X3','extra']))]

    # Filter for time points
    P1_mask = adata.obs["time"] == "P1"
    S1_mask = adata.obs["time"] == "S1"
    T1_mask = adata.obs["time"] == "T1"

    # Features: Embeddings from P1
    X = adata[P1_mask].obsm["X_pca"]

    # get target variable and clones from S1 and T1
    data.S1_clones = adata[S1_mask].obs["clone_id"].values  # Clonotypes at S1
    data.S1_targets = adata[S1_mask].obs["IFN Response_score"].values # target variable

    data.T1_clones = adata[T1_mask].obs["clone_id"].values  # Clonotypes at T1
    data.T1_targets = adata[T1_mask].obs["IFN Response_score"].values # target variable

    # Map S1 and T1 responses to corresponding P1 clones to ensure that only clones with corresponding targets are included
    data.P1_clones = adata[P1_mask].obs["clone_id"].values
    s1 = np.array([data.S1_targets[np.where(data.S1_clones == cid)[0][0]] if cid in data.S1_clones else np.nan for cid in data.P1_clones])
    t1 = np.array([data.T1_targets[np.where(data.T1_clones == cid)[0][0]] if cid in data.T1_clones else np.nan for cid in data.P1_clones])

    # Remove clones without corresponding targets
    valid_indices = ~np.isnan(s1) & ~np.isnan(t1)
    X, s1, t1 = X[valid_indices], s1[valid_indices], t1[valid_indices]
    data.P1_clones = data.P1_clones[valid_indices]


def train_test_split(data):

    # Extract unique clonotypes from P1
    unique_clonotypes_aggregated = np.unique(data.P1_clones)

    # Train-test split by unique clonotypes
    train_clonotypes, test_clonotypes = train_test_split(unique_clonotypes_aggregated, test_size=0.2, random_state=42)

    # Create masks for train and test sets based on the asigned clonotypes
    train_mask = np.isin(unique_clonotypes_aggregated, train_clonotypes)
    test_mask = np.isin(unique_clonotypes_aggregated, test_clonotypes)

    # Recreate aggregated arrays aligned with unique clonotypes
    X_aggregated = []  # features
    y_s1_aggregated = [] # target values for s1
    y_t1_aggregated = []


    # Aggregate features and targets by clone
    for clone in unique_clonotypes_aggregated:
        mask = data.P1_clones == clone  # Select rows matching the current clone
        X_aggregated.append(data.X[mask].mean(axis=0))  # Mean of features
        y_s1_aggregated.append(data.s1[mask].mean())      # Mean of s1 for the clone
        y_t1_aggregated.append(data.t1[mask].mean())      # Mean of t1 for the clone

    # Convert lists to arrays
    X_aggregated = np.array(X_aggregated)
    y_s1_aggregated = np.array(y_s1_aggregated)
    y_t1_aggregated = np.array(y_t1_aggregated)


    # Split aggregated data into train and test sets
    data.X_train, data.X_test = X_aggregated[train_mask], X_aggregated[test_mask]
    data.y_s1_train, data.y_s1_test = y_s1_aggregated[train_mask], y_s1_aggregated[test_mask]
    data.y_t1_train, data.y_t1_test = y_t1_aggregated[train_mask], y_t1_aggregated[test_mask]



def train(data):
    # Train a model for S1 using Linear Regression
    model_s1 = LinearRegression()
    model_s1.fit(data.X_train, data.y_s1_train)
    data.y_s1_pred = model_s1.predict(data.X_test)

    # Train a model for T1 using Linear Regression
    model_t1 = LinearRegression()
    model_t1.fit(data.X_train, data.y_t1_train)
    data.y_t1_pred = model_t1.predict(data.X_test)

    return model_s1, model_t1