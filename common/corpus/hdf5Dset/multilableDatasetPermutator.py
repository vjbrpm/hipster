from typing import List, Dict, Optional, Tuple, Any, Callable, override
import numpy as np
import pickle
from numpy.random import PCG64, Generator

from .multilabelDataset import MultilabelDataset

class MultilabelDatasetPermutator:
	"""
	Permutator (shuffler) for the MultilabelDataset
	"""

	def initializeIndices(self, dset : MultilabelDataset, rngSeed : int = 0):
		"""
		Initialize permutation indices of given dataset.
		
		:param dset: Dataset to initialize permutation idices for.
		:param rngSeed: Seed of random generator. Default is 0.
		"""

		rng = Generator(PCG64(rngSeed))

		dset.firstIndex = np.arange(dset.textIds.shape[0], dtype=np.int32)
		rng.shuffle(dset.firstIndex)
		rng.shuffle(dset.firstIndex)
		dset.firstIndexRngState = pickle.dumps(rng.bit_generator.state)

		dset.currentIndex = np.copy(dset.firstIndex)
		dset.currentIndexRngState = dset.firstIndexRngState
		dset.currentIndexPermutation = 0

	def buildNextShuffle(self, dset : MultilabelDataset):
		"""
		Shuffle permuation index of dataset once.
		
		:param dset: Dataset to work on.
		"""

		rng = Generator(PCG64(0))
		rng.bit_generator.state = pickle.loads(dset.currentIndexRngState)

		rng.shuffle(dset.currentIndex)
		dset.currentIndexRngState = pickle.dumps(rng.bit_generator.state)
		dset.currentIndexPermutation += 1

	def buildSpecificShuffle(self, dset : MultilabelDataset, shuffleIndex : int):
		"""
		Build a specific shuffle (permutation) of the dataset.
		
		:param dset: Dataset to work on.
		:param shuffleIndex: Index of the shuffle to go to. 0 based.
		"""

		#sanitize inputs
		shuffleIndex = max(0, shuffleIndex)

		#are we at desired shuffle? do nothing
		if dset.currentIndexPermutation == shuffleIndex:
			return
		
		#can we advance? do so
		if dset.currentIndexPermutation < shuffleIndex:
			while dset.currentIndexPermutation != shuffleIndex:
				self.buildNextShuffle(dset)
		#we need to reset and rebuild
		else:
			#reset
			dset.currentIndex = np.copy(dset.firstIndex)
			dset.currentIndexRngState = dset.firstIndexRngState
			dset.currentIndexPermutation = 0

			#advance
			while dset.currentIndexPermutation != shuffleIndex:
				self.buildNextShuffle(dset)


