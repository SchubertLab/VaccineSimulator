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


def eval(data):
    k = 5

    # Evaluate S1 predictions
    mse_s1 = mean_squared_error(data.y_s1_test, data.y_s1_pred)
    recall_ks1 = recall_at_k(data.y_s1_test, data.y_s1_pred, k)

    print(f"MSE for S1 (Linear Regression): {mse_s1}")
    print(f"Recall@{k} for S1: {recall_ks1:.2f}")

    # Evaluate T1 predictions
    mse_t1 = mean_squared_error(data.y_t1_test, data.y_t1_pred)
    recall_kt1 = recall_at_k(data.y_t1_test, data.y_t1_pred, k)
    print(f"MSE for T1 (Linear Regression): {mse_t1}")
    print(f"Recall@{k} for T1: {recall_kt1:.2f}")

    plots(data)


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