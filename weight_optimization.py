import torch
import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data import create_batch_input_and_labels
from data import get_squared_distances
from data import calculate_spearman_correlations

def evaluate_correlation(icd_tree, embedding_dict, code_subset):
    embeddings, label_matrix = create_batch_input_and_labels(code_subset, icd_tree, embedding_dict)
    pred_distances = get_squared_distances(embeddings).detach().cpu().numpy()
    label_distances = label_matrix.detach().cpu().numpy()
    _, mean_corr = calculate_spearman_correlations([pred_distances],[label_distances])
    return mean_corr

def train_weights(icd_tree, embedding_dict, icd_codes,
                  batch_size = 32, learning_rate = 0.01, epsilon = 1e-4, max_epochs = 10, patience=3,
                  initial_weights = [1.0, 1.0, 1.0, 1.0, 1.0]):

    icd_codes_train, icd_codes_test = train_test_split(icd_codes, test_size=0.2, random_state=42)
    weights = np.array(initial_weights, dtype=float)
    best_weights = weights.copy()

    train_correlations_per_batch = []
    test_correlations_per_batch = []

    no_improvement_count = 0
    best_corr = -np.inf

    for epoch in range(max_epochs):
        np.random.shuffle(icd_codes_train)
        batches = [icd_codes_train[i:i+batch_size] for i in range(0, len(icd_codes_train), batch_size)]
        
        with tqdm(total=len(batches), desc=f"Epoch {epoch+1}/{max_epochs}", unit="batch") as pbar:
            for batch_codes_train in batches:
                if len(batch_codes_train) < 2:
                    pbar.update(1)
                    continue

                icd_tree.initialize_edge_weights(weights.tolist())

                base_corr_train = evaluate_correlation(icd_tree, embedding_dict, batch_codes_train)
                test_indices = np.random.choice(len(icd_codes_test), size=min(batch_size, len(icd_codes_test)), replace=False)
                batch_codes_test = icd_codes_test[test_indices]
                base_corr_test = evaluate_correlation(icd_tree, embedding_dict, batch_codes_test)

                train_correlations_per_batch.append(base_corr_train)
                test_correlations_per_batch.append(base_corr_test)

                # Approximate gradients via finite differences
                grad = np.zeros_like(weights)
                for j in range(len(weights)):
                    original_weight = weights[j]

                    # Positive perturbation
                    weights[j] = original_weight + epsilon
                    icd_tree.initialize_edge_weights(weights.tolist())
                    corr_pos = evaluate_correlation(icd_tree, embedding_dict, batch_codes_train)

                    # Negative perturbation
                    weights[j] = original_weight - epsilon
                    icd_tree.initialize_edge_weights(weights.tolist())
                    corr_neg = evaluate_correlation(icd_tree, embedding_dict, batch_codes_train)

                    # Restore original
                    weights[j] = original_weight

                    grad[j] = (corr_pos - corr_neg) / (2 * epsilon)

                # Update weights
                weights = weights + learning_rate * grad

                # Check improvement
                icd_tree.initialize_edge_weights(weights.tolist())
                new_corr_train = evaluate_correlation(icd_tree, embedding_dict, batch_codes_train)

                if new_corr_train > best_corr:
                    best_corr = new_corr_train
                    best_weights = weights.copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1

                if no_improvement_count > patience:
                    print("Early stopping triggered. No improvement after {} batches.".format(patience))
                    weights = best_weights
                    icd_tree.initialize_edge_weights(weights.tolist())
                    return best_weights, train_correlations_per_batch, test_correlations_per_batch

                pbar.update(1)

    icd_tree.initialize_edge_weights(best_weights.tolist())
    return best_weights, train_correlations_per_batch, test_correlations_per_batch





