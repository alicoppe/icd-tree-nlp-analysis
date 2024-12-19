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


class L1DistanceLossOriginal(nn.Module):
  """Custom L1 loss for distance matrices."""
  def __init__(self, args):
    super(L1DistanceLoss, self).__init__()
    self.args = args
    self.word_pair_dims = (1,2)

  def forward(self, predictions, label_batch, length_batch):
    """ Computes L1 loss on distance matrices.

    Ignores all entries where label_batch=-1
    Normalizes first within sentences (by dividing by the square of the sentence length)
    and then across the batch.

    Args:
      predictions: A pytorch batch of predicted distances
      label_batch: A pytorch batch of true distances
      length_batch: A pytorch batch of sentence lengths

    Returns:
      A tuple of:
        batch_loss: average loss in the batch
        total_sents: number of sentences in the batch
    """
    labels_1s = (label_batch != -1).float()
    predictions_masked = predictions * labels_1s
    labels_masked = label_batch * labels_1s
    total_sents = torch.sum((length_batch != 0)).float()
    squared_lengths = length_batch.pow(2).float()
    if total_sents > 0:
      loss_per_sent = torch.sum(torch.abs(predictions_masked - labels_masked), dim=self.word_pair_dims)
      normalized_loss_per_sent = loss_per_sent / squared_lengths
      batch_loss = torch.sum(normalized_loss_per_sent) / total_sents
    else:
      batch_loss = torch.tensor(0.0, device=self.args['device'])
    return batch_loss, total_sents
