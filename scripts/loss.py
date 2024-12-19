"""Custom loss classes for probing tasks."""

import torch
import torch.nn as nn

class L1DistanceLoss(nn.Module):
    """Custom L1 loss for a single tree's distance matrix."""
    def __init__(self, args):
        super(L1DistanceLoss, self).__init__()
        self.args = args

    def forward(self, predictions, label_matrix):
        """Computes L1 loss for a single distance matrix.

        Ignores entries where label_matrix = -1.

        Args:
            predictions: A tensor of shape (num_words, num_words) for predicted distances.
            label_matrix: A tensor of shape (num_words, num_words) for true distances.

        Returns:
            The L1 loss for the entire distance matrix.
        """
        # Mask invalid entries -- Though there should be no invalid entries in the matrix
        labels_1s = (label_matrix != -1).float()
        predictions_masked = predictions * labels_1s
        labels_masked = label_matrix * labels_1s

        # Compute L1 loss
        num_valid_pairs = torch.sum(labels_1s)
        if num_valid_pairs > 0:
            loss = torch.sum(torch.abs(predictions_masked - labels_masked)) / num_valid_pairs
        else:
            loss = torch.tensor(0.0, device=self.args['device'])
        return loss

