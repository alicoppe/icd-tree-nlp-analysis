import torch.nn as nn
import torch

class Probe(nn.Module):
  pass


class TwoWordPSDProbe(nn.Module):
  
  
  def __init__(self, model_dim, probe_rank, device='cpu'):
      """
      Initializes the TwoWordPSDProbe.

      Args:
          model_dim (int): Dimensionality of the input representations.
          probe_rank (int): Maximum rank for the projection matrix.
          device (str): Device to run the computations on ('cpu' or 'cuda').
      """
      print('Constructing TwoWordPSDProbe for a single large tree')
      super(TwoWordPSDProbe, self).__init__()
      self.model_dim = model_dim
      self.probe_rank = probe_rank
      self.proj = nn.Parameter(data=torch.zeros(self.model_dim, self.probe_rank))
      nn.init.uniform_(self.proj, -0.05, 0.05)
      self.to(device)

  def forward(self, representations):
      """Computes pairwise distances for a single large tree.

      Args:
          representations: A tensor of shape (num_words, representation_dim).

      Returns:
          A tensor of pairwise squared distances of shape (num_words, num_words).
      """
      transformed = torch.matmul(representations, self.proj)  # Shape: (N, probe_rank)
      num_words, rank = transformed.size()

      # Expand for pairwise computation
      transformed = transformed.unsqueeze(0).expand(num_words, num_words, rank)
      transposed = transformed.transpose(0, 1)

      # Compute squared distances
      diffs = transformed - transposed
      squared_diffs = diffs.pow(2)
      squared_distances = torch.sum(squared_diffs, -1)  # Shape: (N, N)
      return squared_distances