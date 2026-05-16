from typing import List, Dict, Optional, Tuple, Any, Callable

import numba

import torch
from torch.utils.data import Dataset, DataLoader

from .multilabelDataset import MultilabelDataset


class MultilabelTorchDataset(Dataset):
	"""
	Torch dataset for training. This is for multi-label samples.
	"""

	dev : torch.device
	dset : MultilabelDataset	
	
	def __init__(self, dset : MultilabelDataset):
		"""
		Constructor.
		
		:param dset: Dataset to wrap.
		"""

		super().__init__()

		#store inputs
		self.dset = dset

		#create torch device
		self.dev = torch.device('cpu')
	
	def __len__(self) -> int:
		return self.dset.currentIndex.shape[0]
	
	def __getitem__(self, itemIdx) -> Tuple[Tuple[torch.tensor, torch.tensor, int], torch.tensor]:
		#run the given item index through the current permutation
		sampleIdx = self.dset.currentIndex[itemIdx]

		#get index of text encoding for the sample
		txtEncIdx = self.dset.textEncodingIdxs[sampleIdx]

		#extract sample parts, convert to tensors
		if self.dset.embeds.shape[0] != 0:
			inputTknsIdsOrEmbeds = torch.tensor(self.dset.embeds[txtEncIdx], dtype=torch.float32, device=self.dev)
		else:
			inputTknsIdsOrEmbeds = torch.tensor(self.dset.tokenIds[txtEncIdx], dtype=torch.int32, device=self.dev)

		inputTknsMask = torch.tensor(self.dset.masks[txtEncIdx], dtype=torch.int8, device=self.dev)
		tknIdx = torch.tensor(self.dset.targetTokens[sampleIdx], dtype=torch.int32, device=self.dev)
		tknLbls = torch.tensor(self.dset.targetTokenLabels[sampleIdx], dtype=torch.float32, device=self.dev)

		#
		return [[inputTknsIdsOrEmbeds, inputTknsMask, tknIdx], tknLbls]