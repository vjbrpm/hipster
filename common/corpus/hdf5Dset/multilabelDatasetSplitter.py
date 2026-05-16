from typing import List, Dict, Optional, Tuple, Any, Callable, override
import numpy as np
from random import Random

from numba import numba

from tqdm import tqdm

from .multilabelDataset import MultilabelDataset


class MultilabelDatasetSplitter:
	"""
	Splitter for masked token dataset. 
	
	Supports correct distribution of multilabel tokens between train and validate sets based on iterative stratification algorithm for multilabel data described in Sechidis, Konstantinos, Grigorios Tsoumakas, and Ioannis Vlahavas. "On the stratification of multi-label data." Joint European conference on machine learning and knowledge discovery in databases. Berlin, Heidelberg: Springer Berlin Heidelberg, 2011.
	"""

	def split(
		self, 
		dset: MultilabelDataset, 
		foldDist : List[float], 
		foldBackingFileNamePrefixes : List[str], 
		rng : Random | None = None
	) -> List[MultilabelDataset]:
		"""
		Split given dataset into given folds with iterative stratification.
		
		:param dset: Dataset to split.
		:param foldDist: Fold ratio distribution. Must sum to 1.
		:param foldBackingFileNamePrefixes: Name (path) prefix of the backing files for folds.
		:param rng: Random generator to use for splitting.
		:return: A corresponding dataset for each fold.
		"""

		#validate inputs
		foldDist = np.array(foldDist, np.float32)
		if np.sum(foldDist) != 1.0:
			raise AssertionError("Fold ratios in argument 'folds' must sum up to 1.")
		
		if len(foldDist) != len(foldBackingFileNamePrefixes):
			raise AssertionError("Each fold must be supplied a corresponding backing file name prefix in 'bacingFileNamePrefixes'.")

		#no RNG given? create one
		if rng == None:
			rng = Random(0)

		#get fold sample indices
		foldSampleSplits = self._split(dset, foldDist, rng)

		#build the fold datasets
		foldDsets : List[MultilabelDataset] = []

		for foldSampleIdxs, foldBackingFileNamePrefix in zip(foldSampleSplits, foldBackingFileNamePrefixes):
			#allocate storage for dataset of the fold
			foldDset = MultilabelDataset.createLike(dset, foldBackingFileNamePrefix)
			
			#transfer data from the samples of the fold
			foldDset.textIds = dset.textIds[foldSampleIdxs]			
			foldDset.textEncodingIdxs = dset.textEncodingIdxs[foldSampleIdxs]
			foldDset.targetTokens = dset.targetTokens[foldSampleIdxs]
			foldDset.targetTokenLabels = dset.targetTokenLabels[foldSampleIdxs]

			#...set sample row count in the fold dataset to the actual one
			foldDset._numSampleRows = foldDset.textIds.shape[0]

			#allocate text encoding index map
			textEncIdxMap = (
				np.zeros(dset.tokenIds.shape[0], np.int32)
				if dset.tokenIds.shape[0] != 0
				else np.zeros(dset.embeds.shape[0], np.int32)
			)
			textEncIdxMap[:] = -1

			#fix text encoding indices in the fold dataset
			for sampleIdx in tqdm(range(foldDset.textEncodingIdxs.shape[0]), desc="Copying text encoding data"):
				#get original index of the text encoding
				origTexEncIdx = foldDset.textEncodingIdxs[sampleIdx]

				#original index not mapped yet
				if textEncIdxMap[origTexEncIdx] == -1:
					#copy over data from the original text encoding into next free slot
					tokenIds = (
						dset.tokenIds[origTexEncIdx]
						if dset.tokenIds.shape[0] != 0
						else None
					)
					embeds = (
						dset.embeds[origTexEncIdx]
						if dset.embeds.shape[0] != 0
						else None
					)
					mask = dset.masks[origTexEncIdx]

					teRowIdx = foldDset.appendTextEncodingRow(tokenIds, embeds, mask)

					#set mapping
					textEncIdxMap[origTexEncIdx] = teRowIdx
					

				#resolve the text encoding index mapping and update the corresponding index for the sample in the fold dataset
				foldDset.textEncodingIdxs[sampleIdx] = textEncIdxMap[origTexEncIdx]

			#register fold dataset with the results
			foldDsets.append(foldDset)

		#
		return foldDsets

	@numba.njit
	def _findNextSampleIdx(
		sampleIsConsumed : np.ndarray[tuple[int], np.int8], 
		targetTokenLabels : np.ndarray[tuple[int, int], np.int8], 
		chosenLblIdx : int, 
		curSampleIdx : int
	) -> int:
		"""
		This is a jittable loop that searches for the next unconsumed sample having the chosen label. Look at the use inside _split(...) for more context. Speedup when compared with non-jitted version is in 100x of times range.
		"""

		sampleIdx = curSampleIdx
		while (
			sampleIsConsumed[sampleIdx] == 1 or
			targetTokenLabels[sampleIdx, chosenLblIdx] == 0						
		):
			sampleIdx += 1
		return sampleIdx

	def _split(self, dset: MultilabelDataset, foldDist : List[float], rng : Random) -> List[np.ndarray[tuple[int], np.int32]]:
		"""
		Split given dataset into given folds with iterative stratification. This will use an iterative stratification algorithm for multilabel data described in Sechidis, Konstantinos, Grigorios Tsoumakas, and Ioannis Vlahavas. "On the stratification of multi-label data." Joint European conference on machine learning and knowledge discovery in databases. Berlin, Heidelberg: Springer Berlin Heidelberg, 2011.
		
		:param dset: Dataset to split.
		:param foldDist: Fold ratio distribution. Must sum to 1.
		:return: A list of sample indexes for each fold.
		"""

		#get number of folds
		numFolds = len(foldDist)

		#get dataset stats
		numSamples = dset.textIds.shape[0]
		numLabels = dset.targetTokenLabels[0].shape[0]
		labelInstCnts = np.sum(dset.targetTokenLabels, axis=0, dtype=np.int32)
	
		#build sample capacities for each fold
		foldSampleCaps = self._getFoldSampleCounts(numTotalSamples=numSamples, folds=foldDist)

		#build label capacities for each fold
		foldLblCaps = np.zeros((numFolds, numLabels), np.int32)
		for lblIdx, lblInstCnt in enumerate(labelInstCnts):
			lblCapsByFold = self._getFoldSampleCounts(numTotalSamples=lblInstCnt, folds=foldDist)
			foldLblCaps[:, lblIdx] = lblCapsByFold #this writes label caps into a corresponding column

		#allocate sample storage, we allocate additional slot per label since exact final sample counts in a fold can't be known at this point
		foldSampleCnts = np.zeros(numFolds, np.int32)
		foldSamples : List[np.ndarray[tuple[int], np.int32]] = []
		for foldIdx in range(numFolds):
			foldSamples.append(np.zeros(foldSampleCaps[foldIdx], np.int32))

		#allocate consumed samples mask
		sampleIsConsumed = np.zeros(numSamples, np.int8)

		#distribute samples into folds
		with tqdm(total=numSamples, desc="Splitting samples into folds") as prg:
			while np.sum(foldSampleCnts) != numSamples:
				#find the label with the fewest (but at least one) remaining examples, breaking ties randomly
				chosenLblIdxs = np.nonzero(
					labelInstCnts == np.min(labelInstCnts[(labelInstCnts > 0)])
				)[0]
				chosenLblIdx = (
					rng.choice(chosenLblIdxs) 
					if chosenLblIdxs.shape[0] > 1 
					else chosenLblIdxs[0]
				)

				#distribute samples of the chosen label into the folds
				sampleIdx = 0
				
				while labelInstCnts[chosenLblIdx] > 0:
					#find index of next non consumed sample containing the chosen label
					sampleIdx = MultilabelDatasetSplitter._findNextSampleIdx(
						sampleIsConsumed, dset.targetTokenLabels, chosenLblIdx, sampleIdx
					)

					#find unfilled folds with the largest number of desired examples for this label, breaking ties by considering the largest number of desired examples, breaking further ties randomly
					foldLblCapsForChosenLbl = foldLblCaps[:, chosenLblIdx]
					chosenFoldIdxs = np.nonzero(
						(foldLblCapsForChosenLbl == np.max(foldLblCapsForChosenLbl[foldSampleCaps > 0])) &
						(foldSampleCaps > 0)
					)[0] 
					chosenFoldsCapForSamples = foldSampleCaps[chosenFoldIdxs] 
					chosenFoldIdxs = chosenFoldIdxs[chosenFoldsCapForSamples == np.max(chosenFoldsCapForSamples)]
					chosenFoldIdx = (
						rng.choice(chosenFoldIdxs)
						if chosenFoldIdxs.shape[0] > 1
						else chosenFoldIdxs[0]
					)

					#add sample to the fold
					foldSamples[chosenFoldIdx][foldSampleCnts[chosenFoldIdx]] = sampleIdx
					foldSampleCnts[chosenFoldIdx] += 1

					#mark sample consumed
					sampleIsConsumed[sampleIdx] = 1

					#decrease fold capacities
					foldSampleCaps[chosenFoldIdx] -=1
					foldLblCaps[chosenFoldIdx, dset.targetTokenLabels[sampleIdx] != 0] -= 1

					#cosume one instance of each label of the sample
					labelInstCnts[dset.targetTokenLabels[sampleIdx] != 0] -= 1

					#advance progress tracker
					prg.update(1)

		#
		return foldSamples

	def _getFoldSampleCounts(self, numTotalSamples : int, folds : np.ndarray[tuple[int], np.float32]) -> np.ndarray[tuple[int], np.int32]:
		"""
		Given a fold distribution, and number of total samples, calculate how many samples should go into each fold.

		:param numTotalSamples: Number of total samples.
		:param buckets: Bucket distribution. Must sum up to 1.		
		:return: A list with number of samples for each bucket.
		"""

		#validate total samples
		if numTotalSamples < 0:
			raise AssertionError("Argument 'numTotalSamples' is < 0.")

		#validate distribution sum
		if np.sum(folds) != 1.0:
			raise AssertionError("Distribution of 'foolds' must sum up to 1.")
		
		#distribute samples into folds
		numSamplesInFolds = np.ceil(folds * numTotalSamples).astype(np.int32)

		#fix rounding errors by removing samples form largest folds until total sum is correct
		while np.sum(numSamplesInFolds) > numTotalSamples:
			idxMax = np.argmax(numSamplesInFolds)
			numSamplesInFolds[idxMax] -= 1

		#
		return numSamplesInFolds
	