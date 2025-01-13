import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os
from tqdm import tqdm
import scanpy as sc
from sklearn.model_selection import train_test_split
import ot  # Python Optimal Transport library
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt


def pipeline(data, embedding, outcome_variable, clone_pairing_method, split, model):

    # TODO include outcome variable in the pipeline
    if embedding == "pca":
        transformed_data = create_pca_embeddings(data)

    transformed_data = clone_aggregation(transformed_data)
    if clone_pairing_method == "time_point":
        cell_paring(transformed_data)
    elif clone_pairing_method == "optimal_transport":
        optimal_transport(transformed_data)
    elif clone_pairing_method == "random":
        random_pairing(transformed_data)

    if split == "timepoint":
        train_test_split(transformed_data)
    elif split == "random": # TODO adapt here
        train_test_split(transformed_data)
    elif split == "donor":
        train_test_split(transformed_data)
    
    model_s1, model_t1 = train(transformed_data, model)

    

    return model_s1, model_t1, data

def create_pca_embeddings(data):
    sc.tl.pca(data, n_comps=10) # Perform PCA -> use 10 components since they explain 90% of the variance (plot)
    sc.pl.pca(data) # Plot only the top two principal components as they are most informative
    sc.pl.pca_variance_ratio(data, n_pcs=15) 
    return data

def clone_aggregation(adata):
    # filter out cells that we can't use
    adata = adata[~adata.obs.clone_id.isna()]
    adata = adata[~(adata.obs.clone_id=='nan')]
    adata = adata[~(adata.obs.time.isin(['X3','extra']))]

    # Filter for time points
    P1_mask = adata.obs["time"] == "P1"
    S1_mask = adata.obs["time"] == "S1"
    T1_mask = adata.obs["time"] == "T1"

    # Features: Embeddings from P1
    self.X = adata[P1_mask].obsm["X_pca"]

    # get target variable and clones from S1 and T1
    self.S1_clones = adata[S1_mask].obs["clone_id"].values  # Clonotypes at S1
    self.S1_targets = adata[S1_mask].obs["IFN Response_score"].values # target variable

    self.T1_clones = adata[T1_mask].obs["clone_id"].values  # Clonotypes at T1
    self.T1_targets = adata[T1_mask].obs["IFN Response_score"].values # target variable
    self.P1_clones = adata[P1_mask].obs["clone_id"].values
    return adata

def random_pairing(adata):
    # alternative: random pairing of cells - baseline model
    # Random pairing of cells
    random_indices_s1 = np.random.permutation(len(self.S1_targets))
    random_indices_t1 = np.random.permutation(len(self.T1_targets))

    # Shuffle and align S1 and T1 responses with P1 clones randomly
    s1_random = self.S1_targets[random_indices_s1[:len(self.P1_clones)]]
    t1_random = self.T1_targets[random_indices_t1[:len(self.P1_clones)]]

    # Aggregate features for training
    X_aggregated = []
    y_s1_aggregated = []
    y_t1_aggregated = []

    for clone in np.unique(self.P1_clones):
        mask = self.P1_clones == clone  # Select rows for the current clone
        X_aggregated.append(self.X[mask].mean(axis=0))  # Mean of features
        y_s1_aggregated.append(s1_random[mask].mean())  # Randomized S1 response
        y_t1_aggregated.append(t1_random[mask].mean())  # Randomized T1 response

    # Convert to arrays
    X_aggregated = np.array(X_aggregated)
    y_s1_aggregated = np.array(y_s1_aggregated)
    y_t1_aggregated = np.array(y_t1_aggregated)

    return adata

def cell_paring(adata):

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
    unique_clonotypes_aggregated = np.unique(P1_clones)

    # Train-test split by optimal transport aggregation
    train_clonotypes, test_clonotypes  = train_test_split(unique_clonotypes_aggregated, test_size=0.2, random_state=42)

    # Create masks for train and test sets based on the asigned clonotypes
    train_mask = np.isin(unique_clonotypes_aggregated, train_clonotypes)
    test_mask = np.isin(unique_clonotypes_aggregated, test_clonotypes)

    # Recreate aggregated arrays aligned with unique clonotypes
    X_aggregated = []  # features
    y_s1_aggregated = [] # target values for s1
    y_t1_aggregated = []


    # Aggregate features and targets by clone
    for clone in unique_clonotypes_aggregated:
        mask = P1_clones == clone  # Select rows matching the current clone
        X_aggregated.append(X[mask].mean(axis=0))  # Mean of features
        y_s1_aggregated.append(s1[mask].mean())      # Mean of s1 for the clone
        y_t1_aggregated.append(t1[mask].mean())      # Mean of t1 for the clone

    # Convert lists to arrays
    X_aggregated = np.array(X_aggregated)
    y_s1_aggregated = np.array(y_s1_aggregated)
    y_t1_aggregated = np.array(y_t1_aggregated)


    # Split aggregated data into train and test sets
    X_s1_train, X_s1_test = X_aggregated[train_mask], X_aggregated[test_mask]
    y_s1_train, y_s1_test = y_s1_aggregated[train_mask], y_s1_aggregated[test_mask]
    y_t1_train, y_t1_test = y_t1_aggregated[train_mask], y_t1_aggregated[test_mask]



def train(data, model):

    if model == "linear_regression":
        # Train a model for S1 using Linear Regression
        model_s1 = LinearRegression()
        model_s1.fit(data.X_s1_train, data.y_s1_train)
        data.y_s1_pred = model_s1.predict(data.X_s1_test)

        # Train a model for T1 using Linear Regression
        model_t1 = LinearRegression()
        model_t1.fit(data.X_t1_train, data.y_t1_train)
        data.y_t1_pred = model_t1.predict(data.X_t1_test)

    elif model == "gradient_boosting":
        # Initialize and train the Gradient Boosting Regressor
        gbr_s1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gbr_s1.fit(data.X_s1_train, data.y_s1_train)
        data.y_s1_pred = gbr_s1.predict(data.X_s1_test)


        gbr_t1 = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
        gbr_t1.fit(data.X_s1_train, data.y_t1_train)
        data.y_t1_pred = gbr_t1.predict(data.X_t1_test)

    return model_s1, model_t1


def optimal_transport(data):
    # P1 and S1 masks
    # Extract target IFN response scores
    y_p1 = adata[P1_mask].obs["IFN Response_score"].values
    y_s1 = adata[S1_mask].obs["IFN Response_score"].values

    # Extract PCA embeddings and target values
    X_p1 = adata[P1_mask].obsm["X_pca"]
    X_s1 = adata[S1_mask].obsm["X_pca"]

    # Compute the cost matrix (Euclidean distances)
    cost_matrix = ot.dist(X_p1, X_s1, metric='euclidean')

    # Normalize the cost matrix to probabilities
    a = np.ones(len(X_p1)) / len(X_p1)  # Uniform distribution over P1 cells
    b = np.ones(len(X_s1)) / len(X_s1)  # Uniform distribution over S1 cells

    # apply optimal transport library to match
    transport_output = ot.emd(a, b, cost_matrix)

    # Extract paired indices
    paired_indices = np.argwhere(transport_output > 0)
    paired_p1_indices = paired_indices[:, 0]
    paired_s1_indices = paired_indices[:, 1]

    # Extract paired features and targets
    paired_features_p1 = X_p1[paired_p1_indices]
    paired_features_s1 = X_s1[paired_s1_indices]
    paired_targets_p1 = y_p1[paired_p1_indices]
    paired_targets_s1 = y_s1[paired_s1_indices]

    # Combine paired features (e.g., concatenate or compute differences)
    X_regression = np.hstack([paired_features_p1, paired_features_s1])  # Concatenate features
    y_s1_regression = paired_targets_s1  # Target is S1 response    


    # Split data into training and testing sets
    X_s1_train, X_s1_test, y_s1_train, y_s1_test = train_test_split(X_regression, y_s1_regression, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()
    model.fit(X_s1_train, y_s1_train)

    # Predict on the test set
    y_s1_pred = model.predict(X_s1_test)

    # T1 and P1 masks
    # Extract target IFN response scores for T1
    y_t1 = adata[T1_mask].obs["IFN Response_score"].values

    # Extract PCA embeddings for T1
    X_t1 = adata[T1_mask].obsm["X_pca"]

    # Compute cost matrix between P1 and T1
    cost_matrix_t1 = ot.dist(X_p1, X_t1, metric='euclidean')

    # Uniform distributions for P1 and T1
    a_t1 = np.ones(len(X_p1)) / len(X_p1)
    b_t1 = np.ones(len(X_t1)) / len(X_t1)

    # Solve the optimal transport problem for P1 and T1
    transport_plan_t1 = ot.emd(a_t1, b_t1, cost_matrix_t1)

    # Extract paired indices for P1 and T1
    paired_indices_t1 = np.argwhere(transport_plan_t1 > 0)
    paired_p1_indices_t1 = paired_indices_t1[:, 0]
    paired_t1_indices = paired_indices_t1[:, 1]

    # Extract paired features and targets for T1
    paired_features_p1_t1 = X_p1[paired_p1_indices_t1]
    paired_features_t1 = X_t1[paired_t1_indices]
    paired_targets_p1_t1 = y_p1[paired_p1_indices_t1]
    paired_targets_t1 = y_t1[paired_t1_indices]

    # Combine paired features for T1
    X_regression_t1 = np.hstack([paired_features_p1_t1, paired_features_t1])
    y_regression_t1 = paired_targets_t1  # Target is T1 response

    # Combine S1 and T1 datasets
    X_combined = np.vstack([X_regression, X_regression_t1])  # Stack P1-S1 and P1-T1 features
    y_combined = np.hstack([y_s1_regression, y_regression_t1])  # Stack S1 and T1 targets