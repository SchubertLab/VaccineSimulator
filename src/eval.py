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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

class eval(object):

    def eval(self, data):

        k = 5

        self.unique_clonotypes_aggregated = data.unique_clonotypes_aggregated

        # Evaluate S1 predictions
        self.y_s1_test = data.y_s1_test
        self.y_s1_pred = data.y_s1_pred_aggregated

        self.y_t1_test = data.y_t1_test
        self.y_t1_pred = data.y_t1_pred_aggregated


        mse_s1 = mean_squared_error(self.y_s1_test, self.y_s1_pred_aggregated)
        mse_s1 = mse_clonotype(self.y_s1_test, self.y_s1_pred_aggregated)
        recall_ks1 = self.recall_at_k_clonotype(
            self.y_s1_test,
            self.y_s1_pred_aggregated,
            k
        )
        mean_clonotype_activation_s1_predicted = self.y_s1_pred_aggregated.mean()  # Predicted mean clonotype activation for the test set
        mean_clonotype_activation_s1 = self.y_s1_test.mean()# True mean clonotype activation for the test set



        print(f"Recall@{k} for S1: {recall_ks1:.2f}")
        print(f"MSE for S1: {mse_s1}")
        print(f"Mean Clonotype Activation (S1): {mean_clonotype_activation_s1:.4f}")
        print(f"Mean Clonotype Activation (S1) Predicted: {mean_clonotype_activation_s1_predicted:.4f}")

        # Evaluate T1 predictions
        mse_t1 = mean_squared_error(self.y_t1_test, self.y_t1_pred_aggregated)
        recall_kt1 = recall_at_k_clonotype(
            self.y_t1_test,
            self.y_t1_pred_aggregated,
            k
        )
        #ms2_t1_gbr = mean_squared_error(y_t1_test, y_t1_pred_gbr)
        mean_clonotype_activation_t1 = self.y_t1_test.mean()
        mean_clonotype_activation_t1_predicted = self.y_t1_pred_aggregated.mean()

        print(f"Recall@{k} for T1: {recall_kt1:.2f}")
        print(f"MSE for T1: {mse_t1}")
        print(f"Mean Clonotype Activation (T1): {mean_clonotype_activation_t1:.4f}")
        print(f"Mean Clonotype Activation (T1) Predicted: {mean_clonotype_activation_t1_predicted:.4f}")





def recall_at_k(self, y_true, y_pred, k):
    # Get the indices of the top-k predicted scores
    top_k_pred_indices = np.argsort(y_pred)[-k:]  # Indices of top-k predicted scores
    # Get the indices of the top-k true scores
    top_k_true_indices = np.argsort(y_true)[-k:]  # Indices of top-k true scores

    # Compute intersection of predicted and true top-k indices
    intersection = np.intersect1d(top_k_pred_indices, top_k_true_indices)

    # Compute Recall@k
    recall_k = len(intersection) / k
    return recall_k

def aggregate_responses_by_clonotype(self, y, clonotype_labels):
    """
    Aggregate responses by clonotype.

    Args:
        y (array-like): Responses (true or predicted) for each cell.
        clonotype_labels (array-like): Clonotype labels for each cell.

    Returns:
        dict: A dictionary mapping each clonotype to its average response.
    """
    y = np.array(y)
    clonotype_labels = np.array(clonotype_labels)
    unique_clonotypes = np.unique(clonotype_labels)

    clonotype_avg_responses = {}
    for clonotype in unique_clonotypes:
        # Identify cells belonging to this clonotype
        clonotype_mask = clonotype_labels == clonotype
        # Compute the average response for this clonotype
        clonotype_avg_responses[clonotype] = np.mean(y[clonotype_mask])

    return clonotype_avg_responses


def recall_at_k_clonotype(self, y_true, y_pred, k):
    """
    Compute recall at K based on clonotype-level responsiveness.

    Args:
        y_true (array-like): True responses for each cell.
        y_pred (array-like): Predicted responses for each cell.
        clonotype_labels (array-like): Clonotype labels for each cell.
        k (int): Number of top clonotypes to consider.

    Returns:
        float: Recall at K based on clonotypes.

    """

    clonotype_labels = self.unique_clonotypes_aggregated[self.test_mask]
    
    # Aggregate responses by clonotype
    true_responses_by_clonotype = self.aggregate_responses_by_clonotype(y_true, clonotype_labels)
    pred_responses_by_clonotype = self.aggregate_responses_by_clonotype(y_pred, clonotype_labels)

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

def mse_clonotype(self, y_true, y_pred):
    # List to store MSE for each clonotype
    clonotype_mse = []

    # Loop through each unique clonotype
    for clone in self.unique_clonotypes_aggregated:
        # Create a mask for the current clonotype in the test set
        mask = self.P1_clones == clone  # Use the appropriate clonotype variable

        # Get the true values (e.g., y_s1_test, y_t1_test, y_p1_test) and predicted values
        true_values = y_true[mask]  # True values for this clonotype (e.g., S1)
        predicted_values = y_pred[mask]  # Predicted values for this clonotype (e.g., S1)

        # Calculate MSE for this clonotype
        mse = mean_squared_error(true_values, predicted_values)
        
        # Append the MSE for this clonotype
        clonotype_mse.append(mse)

    # Calculate the average MSE across all clonotypes
    average_mse_clonotype = np.mean(clonotype_mse)

    # Print the average MSE on clonotype level
    print(f"Average MSE on clonotype level: {average_mse_clonotype}")



def plots(data):

    # Visualization for S1
    plt.figure(figsize=(6, 6))
    plt.scatter(data.y_s1_test, data.y_s1_pred, alpha=0.7, label="S1 Predictions")
    plt.plot([data.y_s1_test.min(), data.y_s1_test.max()], [data.y_s1_test.min(), data.y_s1_test.max()], 'r--', label="Ideal Fit")
    plt.xlabel("Actual IFN Response (S1)")
    plt.ylabel("Predicted IFN Response (S1)")
    plt.title(f"Prediction Performance for S1 (recall: {data.recall_ks1:.2f})")
    plt.legend()
    plt.show()

    # Visualization for T1
    plt.figure(figsize=(6, 6))
    plt.scatter(data.y_t1_test, data.y_t1_pred, alpha=0.7, label="T1 Predictions")
    plt.plot([data.y_t1_test.min(), data.y_t1_test.max()], [data.y_t1_test.min(),data.y_t1_test.max()], 'r--', label="Ideal Fit")
    plt.xlabel("Actual IFN Response (T1)")
    plt.ylabel("Predicted IFN Response (T1)")
    plt.title(f"Prediction Performance for T1 (recall: {data.recall_kt1:.2f})")
    plt.legend()
    plt.show()