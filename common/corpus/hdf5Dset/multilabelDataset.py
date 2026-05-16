from typing import Any, Optional, Self, Tuple
import os, pickle
import numpy as np
import h5py


class MultilabelDataset:
	"""
	A dataset of multilabel samples. 
	
	This uses hdf5 backend to store token ids, embeddings and masks in a bid to reduce RAM usage while still maintainig reasonable sample access times during training. 
	
	'tokenIds' and 'embeds' are both marked as optional, but at least one of them must exist in a valid dataset. In case of word based tokenization 'embeds' and 'masks' will represent words and 'tokenIds' will not be present.
	"""

	#handle for the hdf5 backing file
	_textEncBackingFile : h5py.File

	#name prefix for the files when saving
	_backingFilesNamePrefix : str	

	#how much to grow axis when resizing ndarrays for more storage
	_resizeAmount : int 

	#how many text encoding rows are actually stored
	_numTextEncodingRows : int #how many rows total

	#how many sample rows are actually stored
	_numSampleRows : int 
	
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

	def create(
			backingFilesNamePrefix : str,
			maxSeqLen : int,
			embedDimLen : int,
			numLbls : int
		) -> Self:
		"""
		Create a new instance of a dataset.

		:param backingFilesName: Name (path) prefix of the backing files.
		:param maxSeqLen: Maximum token sequence length.
		:param embedDimLen: Length of the embedding dimension.
		:param numLbls: Number of labels in one-hot encoding.
		:return: New instance of the dataset.
		"""

		#validate inputs
		if backingFilesNamePrefix is None:
			raise AssertionError("Argument 'backingFilesNamePrefix' is None.")

		#create new instance
		inst = MultilabelDataset()
		inst._backingFilesNamePrefix = backingFilesNamePrefix
		inst._resizeAmount = 1000
		inst._numTextEncodingRows = 0
		inst._numSampleRows = 0

		#open the backing file
		inst._textEncBackingFile = h5py.File(f"{backingFilesNamePrefix}.hdf5", "w")

		#create text encoding row storage
		inst.tokenIds = inst._textEncBackingFile.create_dataset(
			"tokenIds", (0, maxSeqLen), np.int32, maxshape=(None, maxSeqLen)
		)
		inst.embeds = inst._textEncBackingFile.create_dataset(
			"embeds", (0, maxSeqLen, embedDimLen), np.float32, maxshape=(None, maxSeqLen, embedDimLen)
		)
		inst.masks = inst._textEncBackingFile.create_dataset(
			"masks", (0, maxSeqLen), np.int8, maxshape=(None, maxSeqLen)
		)

		#create sample row storage
		inst.textIds = np.ndarray((0,), np.int32)
		inst.textEncodingIdxs = np.ndarray((0,), np.int32)
		inst.targetTokens = np.ndarray((0, ), np.int16)
		inst.targetTokenLabels = np.ndarray((0, numLbls), np.int8)

		#create permutation storage
		inst.firstIndex = np.ndarray((0,), np.int32)
		inst.firstIndexRngState = None

		inst.currentIndex = np.ndarray((0,), np.int32)
		inst.currentIndexPermutation = 0
		inst.currentIndexRngState = None

		#
		return inst
	
	def createLike(example : Self, backingFilesNamePrefix : str) -> Self:
		"""
		Create a dataset with the same shape of text encoding rows as the given one.

		:param example: Dataset to use as example.
		:param backinFilesNamePrefix:  Name (path) prefix of the backing files for the new dataset.

		:return: A new dataset having same shape of text encoding rows as the given one.
		"""

		#validate inputs
		if example is None:
			raise AssertionError("Argument 'example' is None.")

		if backingFilesNamePrefix is None:
			raise AssertionError("Argument 'backingFilesNamePrefix' is None.")
		
		#create new instance
		inst = MultilabelDataset()
		inst._backingFilesNamePrefix = backingFilesNamePrefix
		inst._resizeAmount = 1000
		inst._numTextEncodingRows = 0
		inst._numSampleRows = 0

		#get parameters from example dataset
		maxSeqLen = example.tokenIds.shape[1]
		embedDimLen = example.embeds.shape[2]
		numLbls = example.targetTokenLabels.shape[1]

		#open the backing file
		inst._textEncBackingFile = h5py.File(f"{backingFilesNamePrefix}.hdf5", "w")

		#create text encoding row storage
		inst.tokenIds = inst._textEncBackingFile.create_dataset(
			"tokenIds", (0, maxSeqLen), np.int32, maxshape=(None, maxSeqLen)
		)
		inst.embeds = inst._textEncBackingFile.create_dataset(
			"embeds", (0, maxSeqLen, embedDimLen), np.float32, maxshape=(None, maxSeqLen, embedDimLen)
		)
		inst.masks = inst._textEncBackingFile.create_dataset(
			"masks", (0, maxSeqLen), np.int8, maxshape=(None, maxSeqLen)
		)

		#create sample row storage
		inst.textIds = np.ndarray((0,), np.int32)
		inst.textEncodingIdxs = np.ndarray((0,), np.int32)
		inst.targetTokens = np.ndarray((0, ), np.int16)
		inst.targetTokenLabels = np.ndarray((0, numLbls), np.int8)

		#create permutation storage
		inst.firstIndex = np.ndarray((0,), np.int32)
		inst.firstIndexRngState = None

		inst.currentIndex = np.ndarray((0,), np.int32)
		inst.currentIndexPermutation = 0
		inst.currentIndexRngState = None

		#
		return inst


	def appendTextEncodingRow(
		self,
		tokenIds : Optional[np.ndarray[tuple[int, int], np.int32]], 
		embeds : Optional[np.ndarray[tuple[int, int, int], np.float32]], 
		mask : np.ndarray[tuple[int, int], np.int8]
	) -> int:
		"""
		Append a text encoding row. At least one of 'tokenIds' or 'embeds' must be provided. If any of the 'tokenIds' or 'embeds' are not provided, it must be done consistently for all rows, otherwise the state of the internal storage will become invalid.
		
		:param tokenIds: Token ids for the row. 
		:param embeds: Embeddings for the row.
		:param mask: Mask for the row.

		:return: Index of the row added.
		"""

		#validate inputs
		if tokenIds is None and embeds is None:
			raise AssertionError("At least one of 'tokenIds' or 'embeds' must be provided.")
		
		if mask is None:
			raise AssertionError("Argument 'mask' is None")
		
		#check for inconsistent provision of tokenIds and embeds between calls
		if not(self.tokenIds.shape[0] == 0 and self.embeds.shape[0] == 0):
			if(self.tokenIds.shape[0] > 0) and (tokenIds is None):
				raise AssertionError("Argument 'tokenIds' is not provided a value when at least one row was registered with it.")
			
			if (self.tokenIds.shape[0] == 0) and (tokenIds is not None):
				raise AssertionError("Argument 'tokenIds' is provided a value when at least one row was registered without it.")
			
			if (self.embeds.shape[0] > 0) and (embeds is None):
				raise AssertionError("Argument 'embeds' is not provided a value when at least one row was registered with it.")
			
			if (self.embeds.shape[0] == 0) and (embeds is not None):
				raise AssertionError("Argument 'embeds' is provided a value when at least one row was registered without it.")
		
		#append
		if tokenIds is not None:
			if self.tokenIds.shape[0] < (self._numTextEncodingRows + 1):
				self.tokenIds.resize(self.tokenIds.shape[0] + self._resizeAmount, 0)
			self.tokenIds[self._numTextEncodingRows] = tokenIds

		if embeds is not None:
			if self.embeds.shape[0] < (self._numTextEncodingRows + 1):
				self.embeds.resize(self.embeds.shape[0] + self._resizeAmount, 0)
			self.embeds[self._numTextEncodingRows] = embeds

		if self.masks.shape[0] < (self._numTextEncodingRows + 1):
			self.masks.resize(self.masks.shape[0] + self._resizeAmount, 0)
		self.masks[self._numTextEncodingRows] = mask

		self._numTextEncodingRows += 1

		#
		return (self._numTextEncodingRows - 1)

	def appendSampleRow(
		self,
		textId : int,
		textEncodingIdx : int,
		targetTokenIdx : int,
		targetTokenLabels : np.ndarray[tuple[int], np.int8]
	) -> int:
		"""
		Append sample row. This is only legal is dataset is configured for filling in with data.

		:param textId: Id of the source text.
		:param textEncodingIdx: Index of related text encoding.
		:param targetToken: Index of the target token.
		:param targetTokenLabels: Labels of the target token.

		:return: Index of the row added.
		"""

		#add to storage, if necessary
		if self.textIds.shape[0] < (self._numSampleRows + 1):
			shape = list(self.textIds.shape)
			shape[0] += self._resizeAmount
			self.textIds.resize(tuple(shape))

		if self.textEncodingIdxs.shape[0] < (self._numSampleRows + 1):
			shape = list(self.textEncodingIdxs.shape)
			shape[0] += self._resizeAmount
			self.textEncodingIdxs.resize(tuple(shape))

		if self.targetTokens.shape[0] < (self._numSampleRows + 1):
			shape = list(self.targetTokens.shape)
			shape[0] += self._resizeAmount
			self.targetTokens.resize(tuple(shape))

		if self.targetTokenLabels.shape[0] < (self._numSampleRows + 1):
			shape = list(self.targetTokenLabels.shape)
			shape[0] += self._resizeAmount
			self.targetTokenLabels.resize(tuple(shape))

		#append
		self.textIds[self._numSampleRows] = textId
		self.textEncodingIdxs[self._numSampleRows] = textEncodingIdx
		self.targetTokens[self._numSampleRows] = targetTokenIdx
		self.targetTokenLabels[self._numSampleRows] = targetTokenLabels

		self._numSampleRows += 1

		#
		return (self._numSampleRows - 1)

	def saveToFiles(self):
		"""
		Save the dataset to the files. The backingFilesNamePrefix will be used to derive the file names.
		"""

		#contract text encodings row storage to the actual size of the data
		if self.tokenIds.shape[0] > self._numTextEncodingRows:
			self.tokenIds.resize(self._numTextEncodingRows, 0)

		if self.embeds.shape[0] > self._numTextEncodingRows:
			self.embeds.resize(self._numTextEncodingRows, 0)

		if self.masks.shape[0] > self._numTextEncodingRows:
			self.masks.resize(self._numTextEncodingRows, 0)

		#contract sample row storage to he actual size of the data
		if self.textIds.shape[0] > self._numSampleRows:
			shape = list(self.textIds.shape)
			shape[0] = self._numSampleRows
			self.textIds.resize(tuple(shape))

		if self.textEncodingIdxs.shape[0] > self._numSampleRows:
			shape = list(self.textEncodingIdxs.shape)
			shape[0] = self._numSampleRows
			self.textEncodingIdxs.resize(tuple(shape))

		if self.targetTokens.shape[0] > self._numSampleRows:
			shape = list(self.targetTokens.shape)
			shape[0] = self._numSampleRows
			self.targetTokens.resize(tuple(shape))

		if self.targetTokenLabels.shape[0] > self._numSampleRows:
			shape = list(self.targetTokenLabels.shape)
			shape[0] = self._numSampleRows
			self.targetTokenLabels.resize(tuple(shape))

		#flush and close hdf5 storage
		self._textEncBackingFile.flush()
		self._textEncBackingFile.close()
		
		#save non-hdf5 backed fields as python pickle
		ignoreFields = [
			"_textEncBackingFile", 
			"_backingFilesNamePrefix",
			"tokenIds",
			"embeds",
			"masks"
		]

		with open(f"{self._backingFilesNamePrefix}.pickle", mode="wb") as file:
			saveFields = {k : v for k, v in self.__dict__.items() if k not in ignoreFields}
			pickle.dump(saveFields, file)

	def loadFromFiles(backingFilesNamePrefix : str) -> Self:
		"""
		Load dataset from files.

		:param backingFilesName: Name (path) prefix of the backing files.
		:return: A new instance of the dataset loaded from files.
		"""

		#load the pickled part
		with open(f"{backingFilesNamePrefix}.pickle", mode="rb") as file:
			savedFields = pickle.load(file)

		#create new instance, set the fields from the pickled part
		inst = MultilabelDataset()
		for k, v in savedFields.items():
			inst.__dict__[k] = v

		#load the hdf5 baced part
		inst._backingFilesNamePrefix = backingFilesNamePrefix
		inst._textEncBackingFile = h5py.File(f"{backingFilesNamePrefix}.hdf5", "r+")

		inst.tokenIds = inst._textEncBackingFile["tokenIds"]
		inst.embeds = inst._textEncBackingFile["embeds"]
		inst.masks = inst._textEncBackingFile["masks"]

		#
		return inst

	def deleteFiles(backingFilesNamePrefix : str):
		"""
		Delete dataset files.

		:param backingFilesName: Name (path) prefix of the backing files.
		"""

		os.remove(f"{backingFilesNamePrefix}.hdf5")
		os.remove(f"{backingFilesNamePrefix}.pickle")