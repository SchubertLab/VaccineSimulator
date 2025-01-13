import pandas as pd
import numpy as np
import anndata as ad
import scanpy as sc
from matplotlib import rcParams
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from . import eval
from . import pipeline

# Main code
class run(object):

    def __init__(self, 
                adata, 
                embedding_method = "pca",
                outcome_variable = "IFN_Score",
                clone_pairing_method = "time_point",
                split = "timepoint",
                model = "linear_regression",
                ):
                """
                Raw data processing for further analysis

                adata: AnnData object
                    Annotated data matrix with rows corresponding to cells and columns to genes.
                    It is expected to have the following fields:
                    TODO 

                embedding_method: str
                    The method to use for dimensionality reduction. Options: "pca", "umap", "tsne"
                
                outcome_variable: str
                    The variable to use as the outcome for the model, a score indicating the response of clone to vaccination
                    Options: "IFN_Score", "Viral_Score", "Vaccine_Response"
                
                clone_pairing_method: str
                    The method to use for pairing clones across a variable. 
                    Options: "time_point", "optimal_transport"

                split: str
                    The method to use for splitting the data into training and testing sets. 
                    Options: "timepoint", "random", "donor"

                model: str
                    The model to use for training. 
                    Options: "linear_regression", "gradient_boosting", "pytorch"
                
                
                Returns
                ------- 
                trained model and model statistics indicating how clones respond to vaccinationin a late time point, using data from initial time points
                """



                # init method 
                self.adata= adata, 
                self.embedding_method = embedding_method
                self.outcome_variable = outcome_variable
                self.clone_pairing_method = clone_pairing_method
                self.split = split
                self.model = model
                self.results = None
                self.model_stats = None

    
                # quality check of parameters
                self.model_s1, self.model_t1, data = pipeline.pipeline(adata, self.embedding_method, self.outcome_variable, self.clone_pairing_method, self.split, self.model)

                # eval results
                eval.eval(data)
                    

                    
                    
    def get_results(self):
        return  self.results


    def get_model_stats(self):
        return self.model_stats
