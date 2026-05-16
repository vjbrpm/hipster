from typing import List, Dict, Optional, Tuple, Any, Callable
import pickle
import numpy as np

from tqdm import tqdm

from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils_fast import EncodingFast

from corpus.chunkedCorpus import ChunkedCorpus, ChunkedText, Chunk
from .multilabelDataset import MultilabelDataset


class TextEncondingRow:
	"""
	Token ids/masks row during build process. 
	"""

	tokenIds : np.ndarray[tuple[int], np.int32]
	tokenEmbeds : np.ndarray[tuple[int, int], np.float32]
	tokenMasks : np.ndarray[tuple[int], np.int8]


class SampleRow:
	"""
	Sample row during build process.
	"""

	textId : int
	textEncodingIdx : int
	targetToken :int
	targetTokenLabels : np.ndarray[tuple[int], np.int8]


class DatasetBuild:
	"""
	An ongoing build of dataset
	"""

	textEncodingRows : List[TextEncondingRow]
	sampleRows : List[SampleRow]

	def __init__(self):
		self.textEncodingRows = []
		self.sampleRows = []


Embedder = Callable[[np.ndarray[tuple[int], np.int32], np.ndarray[tuple[int], np.int8]], np.ndarray[tuple[int, int], np.float32]]

class MultilabelDatasetBuilder:
	"""
	Builds a masked dataset from a ranged corpus. Supports multilabel tokens.
	"""

	tokenizer : PreTrainedTokenizerFast
	embedder :  Optional[Embedder]
	corpus : ChunkedCorpus
	
	def __init__(
		self, 
		tokenizer : PreTrainedTokenizerFast, 
		embedder: Optional[Embedder],
		corpus : ChunkedCorpus
	):
		"""
		Constructor.
		:param tokenizer: Tokenizer to use for text tokenization.
		:param embedder: Optional embedder to produce token embeddings. (tokenIds, tokenMask)->tokenEmbeddings.
		:param corpus: Ranged corpus to build the dataset from.
		"""

		self.tokenizer = tokenizer
		self.embedder = embedder
		self.corpus = corpus

	def buildTokenSamples(self, lblsForSpecialTkns : List[int], overlapRatio = 1/3) -> MultilabelDataset:
		"""
		Builds the dataset with token level samples.

		:param dsetName: Name to assign to the dataset in the database.
		:param lblsForSpecialTkns: Labels to assign to special tokens.
		:param overlapRatio: Inter-sequence overlap when generating sequences from texts that exceed maximum model sequence length set in tokenizer. Set to 0 to disable overlaps.
		
		:return: DB entity of the dataset built.
		"""
		
		#create new dataset build
		dsetBuild = DatasetBuild()

		#
		for ctIdx, ct in tqdm(enumerate(self.corpus.texts), "Building samples from corpus texts"):
			#current text is empty string? skip it
			if ct.text.strip() == "":
				continue

			#tokenize current text
			enc = self.tokenizer(
				text=ct.text, 
				padding='max_length', 
				max_length=self.tokenizer.model_max_length,
				return_attention_mask=True,
				add_special_tokens=False,
				truncation=True,
				return_overflowing_tokens=True,
				stride=int(self.tokenizer.model_max_length * overlapRatio),				
			).encodings[0]
			encs = [enc] + enc.overflowing

			#build samples from each model_max_length section of text
			for encIdx, enc in enumerate(encs):
				#extract common encoding components to avoid recomputing values
				inputTkns = enc.ids
				inputTknsMask = enc.attention_mask

				#embedder provided? extract embeddings
				if self.embedder is not None:
					inputTknEmbeds = self.embedder(np.array(inputTkns, np.int32), np.array(inputTknsMask, np.int8))
				else:
					inputTknEmbeds = None

				#build and register a text encoding row
				teRow = TextEncondingRow()
				teRow.tokenIds = np.array(inputTkns, np.int32)
				teRow.tokenEmbeds = inputTknEmbeds
				teRow.tokenMasks = np.array(inputTknsMask, np.int8)
				dsetBuild.textEncodingRows.append(teRow)

				#get labels for all tokens
				tknLbls = self._findTokenLabels(enc=enc, chunks=ct.chunks, tknPadLbls=lblsForSpecialTkns)

				#use attention mask to find how many non-pad tokens are available
				numNonPadTkns = np.array(inputTknsMask, np.int16).sum()

				#build the masked samples
				for tknIdx in range(numNonPadTkns):
					row = SampleRow()
					row.textId = ctIdx
					row.textEncodingIdx = len(dsetBuild.textEncodingRows) - 1
					row.targetToken = tknIdx

					row.targetTokenLabels = np.zeros(len(self.corpus.labelDefs), np.int8)
					for tknLbl in tknLbls[tknIdx]:
						row.targetTokenLabels[tknLbl] = 1

					dsetBuild.sampleRows.append(row)
		
		#convert the dataset build to the final dataset
		dset = self._convertToMultilabelDataset(dsetBuild)

		#
		return dset
	
	def buildWordSamples(self, lblsForSpecialTkns : List[int], overlapRatio = 1/3) -> MultilabelDataset:
		"""
		Builds the dataset with word level samples. This requires embedder to be set, as the words will be build at the embeddings level.

		:param dsetName: Name to assign to the dataset in the database.
		:param lblsForSpecialTkns: Labels to assign to special tokens.
		:param overlapRatio: Inter-sequence overlap when generating sequences from texts that exceed maximum model sequence length set in tokenizer. Set to 0 to disable overlaps. Range [0;0.9].
		
		:return: Dataset built.
		"""

		#make sure overlap ratio is reasonable
		if not(overlapRatio >= 0 and overlapRatio <= 0.9):
			raise AssertionError("Argument 'overlapRatio' must be in range [0;0.9].")

		#no embedder provided? cant build word samples
		if self.embedder is None:
			raise AssertionError("Building word samples requires embedder to be provided, but none is.")

		#create new dataset build
		dsetBuild = DatasetBuild()

		#
		for ctIdx, ct in tqdm(enumerate(self.corpus.texts), "Building samples from corpus texts"):
			#current text is empty string? skip it
			if ct.text.strip() == "":
				continue

			#tokenize current text into a list of 'model_max_length' size encodings
			enc = self.tokenizer(
				text=ct.text, 
				padding='max_length', 
				max_length=self.tokenizer.model_max_length,
				return_attention_mask=True,
				add_special_tokens=False,
				truncation=True,
				return_overflowing_tokens=True,
			).encodings[0]
			encs = [enc] + enc.overflowing

			#build embeddings for the pieces, collect token ids
			encTknEmbeds = []
			encTknIds = []
			encTknOffsets = []
			for encIdx, enc in enumerate(encs):
				#get token ids and attention mask for the current piece
				tknIds = np.array(enc.ids, np.int32)
				tknsMask = np.array(enc.attention_mask, np.int8)
				tknOffsets = np.array(enc.offsets, np.int32)
				
				#build embeddings remove padding tokens, store
				embeds = self.embedder(tknIds, tknsMask)
				embeds = embeds[tknsMask != 0]
				encTknEmbeds.append(embeds)

				#remove padding tokens from ids list, store
				tknIds = tknIds[tknsMask != 0]
				encTknIds.append(tknIds)

				#remove padding tokens from offset list, store
				tknOffsets = tknOffsets[tknsMask != 0]
				encTknOffsets.append(tknOffsets)

			#combine pieces into one sequence for word building
			tknIds = np.concatenate(encTknIds)
			tknOffsets = np.concatenate(encTknOffsets, axis=0)
			tknEmbeds = np.concatenate(encTknEmbeds, axis=0)

			#assign tokens tokens to words
			tknWrds = self._markWords(tknIds, self.tokenizer)

			#use word boundaries to combine token embeddings and offsets into word embeddings and offsets
			wrdEmbeds = np.ndarray((np.max(tknWrds) + 1, tknEmbeds.shape[1]), tknEmbeds.dtype)
			wrdOffsets = np.ndarray((np.max(tknWrds) + 1, tknOffsets.shape[1]), tknOffsets.dtype)
			for wrdIdx in tknWrds:
				#build embeddings for current word, store
				wrdTknEmbeds = tknEmbeds[tknWrds == wrdIdx]
				wrdEmbed = np.average(wrdTknEmbeds, axis=0)
				wrdEmbeds[wrdIdx, :] = wrdEmbed

				#build offsets for current word, store
				wrdTknOffsets = tknOffsets[tknWrds == wrdIdx]
				start = wrdTknOffsets[0, 0]
				end = wrdTknOffsets[-1, 1]
				wrdOffsets[wrdIdx, :] = np.array([start, end], wrdOffsets.dtype)
			
			#split everything back into 'model_max_length' pieces with given overlap, word wise
			numWrds = wrdEmbeds.shape[0]
			maxSeqLen = self.tokenizer.model_max_length

			#...calculate start indices of word sequences, taking overlap ratio into account
			seqStartIdxs : List[int] = []

			if numWrds <= maxSeqLen:
				seqStartIdxs.append(0)
			else:
				#we cant fit sequence into one chunk, and multiple chunks with overlap are needed
				seqStartIdx = 0
				while seqStartIdx < numWrds:
					seqStartIdxs.append(seqStartIdx)
					seqStartIdx += maxSeqLen - int(maxSeqLen * overlapRatio)

			#...build word sequences with offsets and masks
			seqWrdEmbedsAll : List[np.ndarray[tuple[int, int], np.float32]] = []
			seqWrdMasksAll : List[np.ndarray[tuple[int], np.int8]] = []
			seqWrdOffsetsAll : List[np.ndarray[tuple[int, int], np.int32]] = []

			for seqStartIdx in seqStartIdxs:
				#get length of the current sequence
				seqLen = min(numWrds - seqStartIdx, maxSeqLen)

				#extract word embeddings and offsets
				seqWrdEmbeds = wrdEmbeds[seqStartIdx:(seqStartIdx + seqLen)]
				seqWrdOffsets = wrdOffsets[seqStartIdx:(seqStartIdx + seqLen)]

				#pad sequence parts to match max sequence length
				if seqLen < maxSeqLen:					
					seqWrdEmbeds = np.concatenate(
						[
							seqWrdEmbeds,
							np.zeros((maxSeqLen-seqLen, seqWrdEmbeds.shape[1]), seqWrdEmbeds.dtype)
						],
						axis=0
					)
					seqWrdOffsets = np.concatenate(
						[
							seqWrdOffsets,
							np.zeros((maxSeqLen-seqLen, 2), seqWrdOffsets.dtype)
						],
						axis=0
					)
				
				#create word mask
				seqWrdMask = np.zeros((maxSeqLen,), np.int8)
				seqWrdMask[:seqLen] = 1

				#store
				seqWrdEmbedsAll.append(seqWrdEmbeds)
				seqWrdMasksAll.append(seqWrdMask)
				seqWrdOffsetsAll.append(seqWrdOffsets)

			#build samples from sequences
			for wrdEmbeds, wrdMask, wrdOffsets in zip(seqWrdEmbedsAll, seqWrdMasksAll, seqWrdOffsetsAll):
				#build and register a text encoding row
				teRow = TextEncondingRow()
				teRow.tokenIds = None
				teRow.tokenEmbeds = wrdEmbeds
				teRow.tokenMasks = wrdMask
				dsetBuild.textEncodingRows.append(teRow)

				#get labels for all tokens
				wrdLbls = self._findWordLabels(wrdOffsetsAll=wrdOffsets, chunks=ct.chunks, lblsForPad=lblsForSpecialTkns)

				#use attention mask to find how many non-pad words are available
				numNonPadWrds = wrdMask.sum(dtype=np.int16)

				#build the masked samples
				for wrdIdx in range(numNonPadWrds):
					row = SampleRow()
					row.textId = ctIdx
					row.textEncodingIdx = len(dsetBuild.textEncodingRows) - 1
					row.targetToken = wrdIdx

					row.targetTokenLabels = np.zeros(len(self.corpus.labelDefs), np.int8)
					for wrdLbl in wrdLbls[wrdIdx]:
						row.targetTokenLabels[wrdLbl] = 1

					dsetBuild.sampleRows.append(row)
		
		#convert the dataset build to the final dataset
		dset = self._convertToMultilabelDataset(dsetBuild)

		#
		return dset

	def _markWords(self, tkns : np.ndarray[tuple[int], np.int32], tokenizer : PreTrainedTokenizerFast) -> np.ndarray[tuple[int], np.int32]:
		"""
		Mark word in the given token sequence by assigning matching numbers to the tokens that belong to the same word. Special tokens will always be labeled as separate words, one word per token.
		
		:param tkns: A sequence of token ids.
		:param tokenizer: Tokenizer to use.
		
		:return: A sequence where each corresponding token index is marked by a corresponding word number. Word numbers are in range [0;inf), the sequence will be continous.
		"""

		#create result array
		wrds = np.zeros((tkns.shape[0], ), dtype=np.int32)

		#mark words, this may build non-continouse sequence if special tokens are encountered
		wrd = 0
		for tknIdx, tknId in enumerate(tkns):
			#find if current token is special token
			isSpecialToken = tknId in tokenizer.all_special_ids

			#current token is special? give it a separate word
			if isSpecialToken:
				wrd += 1 #end previous word, if any
				wrds[tknIdx] = wrd
				wrd += 1 #start new word
			#normal token
			else:
				#find if current token starts a word
				tknText = tokenizer.convert_ids_to_tokens([tknId])[0]
				isWordStart = tknText.startswith("▁")

				#current token starts a word? advance word number
				if isWordStart:
					wrd += 1

				#assign current token to current word
				wrds[tknIdx] = wrd

		#remap to continous word sequence
		contWrdSeq = np.zeros_like(wrds)
		curWrd = 0
		for idx in range(wrds.shape[0]):
			#word boundary detected? advance current word
			if idx != 0 and wrds[idx-1] != wrds[idx]:
				curWrd += 1
			
			#assign current word to current token
			contWrdSeq[idx] = curWrd

		#
		return contWrdSeq

	def _convertToMultilabelDataset(self, dsetBuild : DatasetBuild) -> MultilabelDataset:
		"""
		Convert the dataset build to multilabel dataset.
		:param dsetBuild: Dataset build to convert.
		:return: A corresponding multilabel dataset.
		"""

		dset = MultilabelDataset()

		#no data available? abort
		if len(dsetBuild.textEncodingRows) == 0:
			return dset

		#determine what parts of text encodings are available
		hasTokenIds = dsetBuild.textEncodingRows[0].tokenIds is not None
		hasEmbeds = dsetBuild.textEncodingRows[0].tokenEmbeds is not None

		#copy data from the build in to the dataset
		if hasTokenIds:
			dset.tokenIds = np.array([x.tokenIds for x in dsetBuild.textEncodingRows], np.int32)		
		if hasEmbeds:
			dset.embeds = np.array([x.tokenEmbeds for x in dsetBuild.textEncodingRows], np.float32)
		dset.masks = np.array([x.tokenMasks for x in dsetBuild.textEncodingRows], np.int8)

		dset.textIds = np.array([x.textId for x in dsetBuild.sampleRows], np.int32)
		dset.textEncodingIdxs = np.array([x.textEncodingIdx for x in dsetBuild.sampleRows], np.int32)
		dset.targetTokens = np.array([x.targetToken for x in dsetBuild.sampleRows], np.int16)
		dset.targetTokenLabels = np.array([x.targetTokenLabels for x in dsetBuild.sampleRows], np.int8)

		#
		return dset

	def _isItersectingRanges(self, sa : int, ea : int , sb : int , eb : int) -> bool:
		"""
		Tell if two numerical integer ranges intersect.
		Inputs.
			sa. Start of range A. Inclusive.
			ea. End of range A. Inclusive.
			sb. Start of range B. Inclusive.
			eb. End of range B. Inclusive.
		"""

		res = (
			#any end of B in A
			(sa <= sb and sb <= ea) or
			(sa <= eb and eb <= ea) or
			#any end of A in B
			(sb <= sa and sa <= eb) or
			(sb <= ea and ea <= eb)
		)
		return res

	def _findTokenLabels(self, enc : EncodingFast, chunks : List[Chunk], tknPadLbls : List[int])-> List[List[int]]:
		"""
		Computes which labels apply to individual tokens.
		Inputs.
			enc. Token encoding of some text.
			chunks. Labeled chunks for the same text.
			tknPadLbls. Labels to assign to special tokens.
		Returns.
			A list of list of token labels.
		"""

		#get ID of <unk> token, because we need to differentiate it from other special tokens
		idTknUnk = self.tokenizer.convert_tokens_to_ids(["<unk>"])[0]
	
		#define result list
		allTknLbls : List[List[int]] = []

		#go over each token
		for tknIdx, [tknOffsets, tknId] in enumerate(zip(enc.offsets, enc.ids)):
			#by default mark all possible token labels as not present
			tknLbls = []

			#determine if current token is special (except for <unk>)
			isTknSpecial = (
				#special tokens except unknown ones
				((enc.special_tokens_mask[tknIdx] == 1) and (tknId != idTknUnk)) or
				#any zero length tokens
				(tknOffsets[0] == tknOffsets[1])
			)

			#special token? assign it the given special token labels
			if isTknSpecial:
				for lbl in tknPadLbls:
					tknLbls.append(lbl)
			#set token labels from the chunks
			else:
				#get token start and end, inclusive
				tknStart = tknOffsets[0]
				tknEnd = tknOffsets[1] - 1

				for chunk in chunks:
					#get start and end of chunk, inclusive
					ckStart = chunk.start
					ckEnd = ckStart + chunk.len - 1

					#is token inside current chunk? add the chunk label to token label list
					isInRange = self._isItersectingRanges(ckStart, ckEnd, tknStart, tknEnd)
					if isInRange and (chunk.label not in tknLbls): #we prevent repeated adds of the same label, just in case
						tknLbls.append(chunk.label)

			#token has no label at all? something is wrong, print a warning
			if len(tknLbls) == 0:
				print(f"Warning. Could not find any label for token with id {tknId}. Set a breakpoint on this line to find more.")

			#add token labels to results
			allTknLbls.append(tknLbls)

		#
		return allTknLbls
	
	def _findWordLabels(self, wrdOffsetsAll : List[np.ndarray[tuple[int, int], np.int32]], chunks : List[Chunk], lblsForPad : List[int])-> List[List[int]]:
		"""
		Computes which labels apply to individual words.
		
		:param wrdOffsetsAll: Word character offsets. (start, end) end is excludive. If start==end we assume this is a padding word.
		:param chunks: Labeled chunks for the same text.
		:param lblsForPad: What labels to assign to the padding words.
		
		:return: A list of lists of word labels.
		"""
	
		#define result list
		wrdLblsAll : List[List[int]] = []

		#go over each token
		for wrdIdx, wrdOffsets in enumerate(wrdOffsetsAll):
			wrdLbls = []

			#determine if current word is padding
			isPadding = (wrdOffsets[0] == wrdOffsets[1])

			#padding word? assign it the given padding word labels
			if isPadding:
				for lbl in lblsForPad:
					wrdLbls.append(lbl)
			#set word labels from the chunks
			else:
				#get word start and end character idices, inclusive
				wrdStart = wrdOffsets[0]
				wrdEnd = wrdOffsets[1] - 1

				for chunk in chunks:
					#get start and end of chunk, inclusive
					ckStart = chunk.start
					ckEnd = ckStart + chunk.len - 1

					#is word intersecting current chunk? add the chunk label to word label list
					isInRange = self._isItersectingRanges(ckStart, ckEnd, wrdStart, wrdEnd)
					if isInRange and (chunk.label not in wrdLbls): #prevent multiples of the same label, just in case
						wrdLbls.append(chunk.label)

			#word has no label at all? something is wrong, print a warning
			if len(wrdLbls) == 0:
				print(f"Warning. Could not find any label for word with index {wrdIdx}. Set a breakpoint on this line to find more.")

			#add word labels to results
			wrdLblsAll.append(wrdLbls)

		#
		return wrdLblsAll