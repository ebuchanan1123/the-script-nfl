"""Numerical multilayer perceptron for WR receiving yards."""

import torch
from torch import nn


class NumericalWRModel(nn.Module):
	"""Small numerical-only network used before categorical embeddings."""

	def __init__(self, number_of_features: int, dropout: float = 0.2) -> None:
		super().__init__()
		self.network = nn.Sequential(
			nn.Linear(number_of_features, 64),
			nn.ReLU(),
			nn.Dropout(dropout),
			nn.Linear(64, 32),
			nn.ReLU(),
			nn.Linear(32, 1),
		)

	def forward(self, numeric: torch.Tensor) -> torch.Tensor:
		return self.network(numeric).squeeze(1)
