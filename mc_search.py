import torch
import numpy as np
from tqdm import tqdm

from data import create_batch_input_and_labels
from data import get_squared_distances
from data import calculate_spearman_correlations
from data import evaluate_full_correlations
from data import evaluate_correlation

def train_weights_mc(icd_tree, embedding_dict, icd_codes_train, icd_codes_test,
                     batch_size=4, search_radius=0.1, num_samples=10,
                     max_epochs=10, patience=10, initial_weights=[1.0, 1.0, 1.0, 1.0, 1.0],
                     min_weight=0.05, num_batches_to_average=10, print_all_weights=False):

    weights = np.array(initial_weights, dtype=float)
    best_weights = weights.copy()

    train_correlations_per_batch = []
    test_correlations_per_batch = []
    max_diff_per_batch_group = []  # To store the max difference for each batch group

    no_improvement_count = 0
    best_corr = -np.inf
    
    print(f'Benchmark Correlation on Test Set with Initial Weights {weights}:')
    test_correlation = evaluate_full_correlations(icd_tree, embedding_dict, icd_codes_test, batch_size=batch_size, shuffle_icd=False)
    print(f"Test set correlation: {test_correlation}")
    best_corr = test_correlation
    test_correlations_per_batch.append(test_correlation)

    for epoch in range(max_epochs):
        train_batches = [icd_codes_train[i:i + batch_size] for i in range(0, len(icd_codes_train), batch_size)]
        train_batches = [tb for tb in train_batches if len(tb) > 1]

        np.random.shuffle(train_batches)

        # Group batches for averaging
        for group_start in range(0, len(train_batches), num_batches_to_average):
            batch_group = train_batches[group_start:group_start + num_batches_to_average]

            icd_tree.initialize_edge_weights(weights.tolist())

            # Calculate correlations for each batch in the group
            group_corrs = [evaluate_correlation(icd_tree, embedding_dict, batch) for batch in batch_group]
            base_corr_train = np.mean(group_corrs)
            train_correlations_per_batch.append(base_corr_train)

            print('\n-----------------------------------------------------------------')
            print(f'No Improvement Count --> {no_improvement_count}')
            print(f"\nEpoch {epoch + 1}, Batch Group {group_start + 1} to {group_start + len(batch_group)}:")
            print(f"  Before update: weights = {weights}, average correlation = {base_corr_train}")

            # Monte Carlo search for optimal weights
            sampled_weights = [weights + np.random.uniform(-search_radius, search_radius, size=weights.shape)
                               for _ in range(num_samples)]
            sampled_weights = [np.maximum(w, min_weight) for w in sampled_weights]  # Clamp weights to min_weight

            best_sample_corr = -np.inf
            worst_sample_corr = np.inf
            best_sample_weights = weights.copy()
            explored_weights = []  # To store all sampled weights and their correlations

            for sample in sampled_weights:
                # Evaluate the average correlation for the sample weights across the batch group
                icd_tree.initialize_edge_weights(sample.tolist())
                sample_corrs = [evaluate_correlation(icd_tree, embedding_dict, batch) for batch in batch_group]
                avg_sample_corr = np.mean(sample_corrs)

                # Store the sampled weight and its correlation for debugging
                explored_weights.append((sample, avg_sample_corr))

                if avg_sample_corr > best_sample_corr:
                    best_sample_corr = avg_sample_corr
                    best_sample_weights = sample
                    
                if avg_sample_corr < worst_sample_corr:
                    worst_sample_corr = avg_sample_corr

            largest_corr_diff = best_sample_corr - worst_sample_corr
            max_diff_per_batch_group.append(largest_corr_diff)

            if print_all_weights:
                # Print all explored weights and their correlations
                print("\nExplored weights and correlations:")
                for w, corr in explored_weights:
                    print(f"  Weights: {w}, Correlation: {corr}")

            print(f"Best correlation in group: {best_sample_corr}, Best weights: {best_sample_weights}")
            print(f'Largest difference in average correlation amongst weights groups: {largest_corr_diff}')

            if best_sample_corr > base_corr_train + base_corr_train / 1000:
                icd_tree.initialize_edge_weights(best_sample_weights.tolist())
                print(f'Batch performance improved with weights: {best_sample_weights}"')
                print(f'Changing weights...')
                weights = best_sample_weights
                
                print('Evaluating on the test set...')
                test_correlation = evaluate_full_correlations(icd_tree, embedding_dict, icd_codes_test, batch_size=batch_size, shuffle_icd=False)
                test_correlations_per_batch.append(test_correlation)
            else:
                print('------------ Batch performance did not improve --------------')
                test_correlations_per_batch.append(best_corr)

            if test_correlation > best_corr:
                prev_best = best_corr
                best_corr = test_correlation
                best_weights = best_sample_weights.copy()
                no_improvement_count = 0
                print(f'Test set spearman correlation improved from {prev_best} to {best_corr}')
                
            else:
                no_improvement_count += 1
                print(f'Test correlation did not improve')
                

            if no_improvement_count > patience:
                print("Early stopping triggered. No improvement.")
                
                return best_weights, train_correlations_per_batch, test_correlations_per_batch, max_diff_per_batch_group

    icd_tree.initialize_edge_weights(best_weights.tolist())
    return best_weights, train_correlations_per_batch, test_correlations_per_batch, max_diff_per_batch_group