from typing import Any, Optional, Self
import pickle
import numpy as np


class MultilabelDataset:
	"""
	A dataset of multilabel samples. This is optimized for RAM storage. 'tokenIds' and 'embeds' are both marked as optional, but at least one of them must exist in a valid dataset. In case of word based tokenization 'embeds' and 'masks' will represent words and 'tokenIds' will not be present.
	"""

	#these fields represent text encodings as table of (token-ids, embeddings, masks) tuples
	tokenIds : Optional[np.ndarray[tuple[int, int], np.int32]]
	embeds: Optional[np.ndarray[tuple[int, int, int], np.float32]]
	masks : np.ndarray[tuple[int, int], np.int8]

	#these fields represent a table of samples, each sample points to a row in text encodings table
	textIds : np.ndarray[tuple[int], np.int32]
	textEncodingIdxs : np.ndarray[tuple[int], np.int32]
	targetTokens : np.ndarray[tuple[int], np.int16]
	targetTokenLabels : np.ndarray[tuple[int, int], np.int8]

	#these fields represent first sample index and related RNG state
	firstIndex : np.ndarray[tuple[int], np.int32]
	firstIndexRngState : Any

	#these fields represent current sample index, and index permuation number
	currentIndex : np.ndarray[tuple[int], np.int32]
	currentIndexPermutation : int
	currentIndexRngState : Any

	def __init__(self):
		#initialize empty fields, this is used when empty dataset needs to be returned from somewhere
		self.tokenIds = None
		self.embeds = None
		self.masks = np.ndarray((0, 0), np.int8)

		self.textIds = np.ndarray((0,), np.int32)
		self.textEncodingIdxs = np.ndarray((0,), np.int32)
		self.targetTokens = np.ndarray((0,), np.int16)
		self.targetTokenLabels = np.ndarray((0, 0), np.int8)

		self.firstIndex = np.ndarray((0,), np.int32)
		self.firstIndexRngState = None

		self.currentIndex = np.ndarray((0,), np.int32)
		self.currentIndexPermutation = 0
		self.currentIndexRngState = None

	def saveToFile(self, path : str):
		"""
		Save this instance to file.

		:param path: File path to use.
		"""

		with open(path, mode="wb") as file:
			pickle.dump(self, file)

	def loadFromFile(path: str) -> "MultilabelDataset":
		"""
		Load an instance from file.
		
		:param path: File path to use.
		:return: An instance loaded.
		"""

		with open(path, mode="rb") as file:
			self = pickle.load(file)
			return self