from typing import List, Dict, Optional, Tuple, Any, Callable
import pickle

import torch
from torch.utils.data import Dataset, DataLoader

from .multilabelTorchDataset import MultilabelTorchDataset


class MultilabelTorchDataLoader(DataLoader):
	"""
	Torch dataloader for working with torch datasets. This is for multi-label datasets.
	"""

	def __init__(self, ds : MultilabelTorchDataset, batchSize : int = 4, debug=False):
		"""
		Constructor.

		:param ds: Dataset to use.
		:param batchSize: Batch size to use.
		:param debug: True to initialize for debugging mode. This prevents parallel workers to prevent issues with VSCode multiprocess debugging.
		"""
		
		super().__init__(
			ds, 
			batch_size=batchSize,
			
			num_workers=4 if not debug else 0,
			persistent_workers=False,
		)