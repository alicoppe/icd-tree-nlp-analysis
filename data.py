import torch
import pandas as pd

import numpy as np
from scipy.stats import spearmanr
from tqdm import tqdm

def create_batch_input_and_labels(icd_code_array, icd_tree, embedding_dict):
    """
    Prepares the batch input and label matrix for the TwoWordPSDProbe and L1DistanceLoss.

    Args:
        icd_code_array (list): A list of ICD codes for the batch.
        icd_tree (ICD_tree): An instance of the ICD_tree class, used to compute tree distances.
        embedding_dict (dict): A dictionary mapping ICD codes to their vector embeddings.

    Returns:
        embeddings (torch.Tensor): A tensor of shape (N, D), where N is the number of ICD codes
                                   and D is the embedding dimension. Represents the batch input embeddings.
        label_matrix (torch.Tensor): A tensor of shape (N, N) containing pairwise distances between all ICD codes in the batch,
                                     computed using the `get_tree_distance` method of the `ICD_tree` instance. 
                                     Diagonal entries are 0 (distance to itself).
    """
    # Extract embeddings from the dictionary
    embeddings = torch.stack([embedding_dict[code].clone() for code in icd_code_array])  # Shape: (N, D)
    num_codes = len(icd_code_array)

    # Initialize the label matrix for pairwise distances
    label_matrix = torch.zeros((num_codes, num_codes))

    # Compute pairwise distances for all ICD code combinations
    for i in range(num_codes):
        for j in range(num_codes):
            if i == j:
                label_matrix[i, j] = 0  # Distance to itself is zero
            else:
                label_matrix[i, j] = icd_tree.get_tree_distance(icd_code_array[i], icd_code_array[j])

    return embeddings, label_matrix



def calculate_spearman_correlations(predicted_distances, labelled_distances):
    """
    Calculates the Spearman correlation between predicted and labeled distances for multiple NxN matrices.

    Args:
        predicted_distances (list of np.ndarray): A list of L NxN matrices containing predicted distances.
        labelled_distances (list of np.ndarray): A list of L NxN matrices containing actual labeled distances.

    Returns:
        correlations (list): A list of Spearman correlation coefficients for each matrix pair.
        mean_correlation (float): The mean Spearman correlation across all matrices.
    """
    correlations = []

    for pred_matrix, label_matrix in zip(predicted_distances, labelled_distances):
        # Flatten the NxN matrices into 1D arrays for correlation calculation
        pred_flat = pred_matrix.flatten()
        label_flat = label_matrix.flatten()

        # Calculate Spearman correlation
        correlation, _ = spearmanr(pred_flat, label_flat)
        correlations.append(correlation)

    # Compute the mean Spearman correlation
    mean_correlation = np.nanmean(correlations)  # Handle NaNs gracefully

    return correlations, mean_correlation

def get_squared_distances(predictions):
  num_words, rank = predictions.size()
  # Expand for pairwise computation
  transformed = predictions
  transformed = transformed.unsqueeze(0).expand(num_words, num_words, rank)
  transposed = transformed.transpose(0, 1)

  # Compute squared distances
  diffs = transformed - transposed
  squared_diffs = diffs.pow(2)
  squared_distances = torch.sum(squared_diffs, -1)  # Shape: (N, N)
  return squared_distances

def evaluate_correlation(icd_tree, embedding_dict, code_subset):
    embeddings, label_matrix = create_batch_input_and_labels(code_subset, icd_tree, embedding_dict)
    pred_distances = get_squared_distances(embeddings).detach().cpu().numpy()
    label_distances = label_matrix.detach().cpu().numpy()
    _, mean_corr = calculate_spearman_correlations([pred_distances],[label_distances])
    return mean_corr
  
def evaluate_full_correlations(icd_tree, embedding_dict, icd_codes, batch_size=10, window_size=None, shuffle_icd=True):
    
    if shuffle_icd:
        if window_size is None:
            np.random.shuffle(icd_codes)
        else:
            icd_codes = locally_shuffle(icd_codes, window_size)
        
    # Create batches from icd codes
    batches = [icd_codes[i:i+batch_size] for i in range(0, len(icd_codes), batch_size)]
    batches = [b for b in batches if len(b) > 1]
    
    pred_distances_arr = []
    label_distances_arr = []
    
    for batch in tqdm(batches, desc="Evaluating batches"):
        embeddings, label_matrix = create_batch_input_and_labels(batch, icd_tree, embedding_dict)
        pred_distances = get_squared_distances(embeddings).detach().cpu().numpy()
        label_distances = label_matrix.detach().cpu().numpy()
        pred_distances_arr.append(pred_distances)
        label_distances_arr.append(label_distances)
        
    correlations, mean_correlations = calculate_spearman_correlations(pred_distances_arr, label_distances_arr)
    
    return mean_correlations

def locally_shuffle(arr, window_size):
    """
    Shuffle an array such that elements mostly stay within a given proximity (window_size).
    
    Parameters:
    arr (list): The input array to shuffle.
    window_size (int): The maximum window size within which most elements will be shuffled.

    Returns:
    list: The shuffled array.
    """
    n = len(arr)
    shuffled = arr.copy()
    for i in range(n):
        # Generate a random offset within the window size using a normal distribution
        offset = int(np.random.normal(loc=0, scale=window_size / 2))
        new_index = max(0, min(n - 1, i + offset))  # Keep the new index within bounds
        # Swap elements
        shuffled[i], shuffled[new_index] = shuffled[new_index], shuffled[i]
    
    return shuffled