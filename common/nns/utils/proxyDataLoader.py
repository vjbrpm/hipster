from typing import List, Dict, Optional, Tuple, Any, Callable
import pickle

import torch
from torch.utils.data import DataLoader

class ProxyDataloader:
	"""
	Proxy dataloader. Used to replace dataloaders in trainer between epochs of training.
	"""
	
	"""
	Target of the proxy.
	"""
	target : DataLoader = None

	def __iter__(self):
		return self.target.__iter__()
	
	def __len__(self) -> int:
		return self.target.__len__()

	