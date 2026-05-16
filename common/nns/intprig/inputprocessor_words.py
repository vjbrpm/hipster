from typing import Dict, List, Tuple, Any, Self
import torch
import numpy as np

from transformers import PreTrainedModel, PreTrainedTokenizerFast
from nns.embedders import Embedder


class ChunkDesc:
	idxOffset : int
	"""
	Chunk offset index. In words.
	"""
	
	embeds : torch.Tensor
	"""
	Chunk word embeddings.
	"""

	mask : torch.Tensor
	"""
	Attention mask.
	"""

	offsets: torch.Tensor
	"""
	Word character offsets. (start, length)
	"""


class InputProcessor:
	"""
	Word based input processor.
	"""
	
	def predictWinningLabels(
			self, 
			inputChunks: List[ChunkDesc],
			model : PreTrainedModel,
	) -> torch.Tensor:
		"""
		Predicts winning labels for each word of given text.

		:param inputChunks: Chunks of the text to predict the labels for.
		:param tokenizer: Tokenizer to use.
		:param model: Model to use.

		:return: A list of winning labels for each word. tensor-of(tensor-of(winning-labels)).
		"""

		#get head configuration from the model 
		numLblsPerHead = model.getNumLblsPerHead()

		#text is empty? return empty result
		if len(inputChunks) == 0:
			return torch.zeros((0, len(numLblsPerHead)), dtype=torch.int8)
		
		#find the total length of the input text in words
		inputLenInWrds = inputChunks[-1].idxOffset + inputChunks[-1].embeds.shape[0]
		
		#get word label confidences for each chunk
		allChunkLblConfs = self.predictLabelConfs(inputChunks=inputChunks, model=model)

		#allocate storage for merged results
		allLblConfs = torch.zeros((inputLenInWrds, sum(numLblsPerHead)), dtype=torch.float32)
		
		#merge chunk label confidences into one sequence
		numTokensPerPos = torch.zeros(allLblConfs.shape[0], dtype=torch.float32)

		for chunkIdx, (chunkLblConfs, chunk) in enumerate(zip(allChunkLblConfs, inputChunks)):
			idxOffset = chunk.idxOffset

			#add chunk label confidences onto final confidences
			allLblConfs[idxOffset:idxOffset+chunkLblConfs.shape[0]] += chunkLblConfs
			numTokensPerPos[idxOffset:idxOffset+chunkLblConfs.shape[0]] += 1

		#derive average label confidences for post-merge result
		allLblConfs /= numTokensPerPos.unsqueeze(0).transpose(0, 1).expand([-1, allLblConfs.shape[1]])

		#extract post-merge token labels
		winningLbls = self._extractWinningLabels(allLblConfs, numLblsPerHead)

		#
		return winningLbls

	def predictLabelConfs(
		self, 
		inputChunks : List[ChunkDesc], 
		model : PreTrainedModel, 
	) -> List[torch.Tensor]:
		"""
		Predict label confidences for each given chunk.

		:param text: Text to predict the labels for.
		:param model: Model to use.

		:return: A list of label confidence predictions. list-of(tensor-of(tensor-of(confidences))).
		"""

		result : List[torch.Tensor] = list()

		with torch.no_grad():
			for chunkIdx, chunk in enumerate(inputChunks):
				#convert inputs to tensors
				embeds = chunk.embeds.to(dtype=torch.float32, device=model.device)
				mask = chunk.mask.to(dtype=torch.int8, device=model.device)

				#add batch dimension
				embeds = embeds.unsqueeze(dim=0)			
				mask = mask.unsqueeze(dim=0)

				#invoke the model to get label confidences, move to cpu, remove batch dimension
				torch.compiler.set_stance("force_eager")
				try:
					lblConfs : torch.Tensor = model(inputTknIdsOrEmbeds=embeds, inputTknMask=mask)
					lblConfs = lblConfs.to(torch.device("cpu")).squeeze()
				finally:
					torch.compiler.set_stance("default")

				#add to results
				result.append(lblConfs)

		#
		return result
	
	def predicLabelConfsForChunk(self, embeds : torch.Tensor, mask : torch.Tensor, model : PreTrainedModel) -> torch.Tensor:
		"""
		Predict label confidences for given chunk. 
		
		The intended use of this function is in calculating integrated gradients. Do not use torch.no_grad() here otherwise gradient accumulation will not be done.

		:param embeds: Chunk word embeddings. Must contain a batch dimension.
		:param mask: Chunk word mask. Must contain a batch dimension.
		:param model: Model to use.

		:return: Label confidence predictions. shape=(batch, outSeqLen, numHeads). The result tensor will be on the same device as model tensor.
		"""

		#make sure inputs are in correct format and on correct device
		embeds = embeds.to(dtype=torch.float32, device=model.device)
		mask = mask.to(dtype=torch.int8, device=model.device)

		#invoke the model to get label confidences, move to cpu, remove batch dimension
		torch.compiler.set_stance("force_eager")
		try:
			lblConfs : torch.Tensor = model(inputTknIdsOrEmbeds=embeds, inputTknMask=mask)
		finally:
			torch.compiler.set_stance("default")

		#
		return lblConfs
	
	def buildInputChunks(self, 
		text : str, 
		tokenizer : PreTrainedTokenizerFast, 
		embedder: Embedder,
		overlapRatio = 1/3
	) -> List[ChunkDesc]:
		"""
		Build word embeddings for a given text. If text results in word sequence longer than the maximum input length of the model, this will return multiple embeddings chunks with given overlap ratio.

		:param text: Text to produce embeddings for.
		:param tokenizer: Tokenizer to use.
		:param embedder: Embedder to use.
		:param overlapRatio: Chunk overlap ratio for texts longer than the maximum input length of the model.

		:return: A list of chunks representing given text. Chunks will have no padding.
		"""

		results : List[ChunkDesc] = list()

		#tokenize current text into a list of 'model_max_length' size encodings
		enc = tokenizer(
			text=text, 
			padding='max_length', 
			max_length=tokenizer.model_max_length,
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
			embeds = embedder(tknIds, tknsMask)
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
		tknWrds = self._markWords(tknIds, tokenizer)

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
		maxSeqLen = tokenizer.model_max_length

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
		for seqStartIdx in seqStartIdxs:
			#get length of the current sequence
			seqLen = min(numWrds - seqStartIdx, maxSeqLen)

			#extract word embeddings and offsets
			seqWrdEmbeds = wrdEmbeds[seqStartIdx:(seqStartIdx + seqLen)]
			seqWrdOffsets = wrdOffsets[seqStartIdx:(seqStartIdx + seqLen)]

			#create word mask
			seqWrdMask = np.zeros((seqLen,), np.int8)
			seqWrdMask[:] = 1

			#recompute offsets into (start, length) format
			seqWrdOffsets[:, 1] -= seqWrdOffsets[:, 0]

			#store
			chunk = ChunkDesc()
			chunk.idxOffset = seqStartIdx
			chunk.embeds = torch.tensor(seqWrdEmbeds, dtype=torch.float32)
			chunk.mask = torch.tensor(seqWrdMask, dtype=torch.int8)
			chunk.offsets = torch.tensor(seqWrdOffsets, dtype=torch.int32)

			results.append(chunk)

		#
		return results

	def _extractWinningLabels(self, confs : torch.Tensor, numLblsPerHead : List[int]) -> torch.Tensor:
		"""
		Given a sequence of label confidences for each token and head configuration, extract winning labels for each head for each token.
		
		:param confs: A sequence of label confidences for each token. shape=(seqLen, sum(numLblsPerHead))
		:param numLblsPerHead: Number of labels per head.

		:return: A sequence of winning labels for each token. shape=(seqLen, len(numLblsPerHead))
		"""
		
		#define result storage, shape=(seqLen, len(numLblsPerHead))
		lbls = torch.zeros((confs.shape[0], len(numLblsPerHead)), dtype=torch.int8)

		#extract
		headOffset = 0
		for headIdx, numHeadLbls in enumerate(numLblsPerHead):
			#get indices of predicted labels for the current head, shape=(seqLen, sum(numLblsPerHead))
			predictedLbls = confs[:, headOffset:(headOffset + numHeadLbls)].argmax(dim=1)
			#offset indices back to the final vector space
			predictedLbls = predictedLbls + headOffset
			#assign the predicted label
			lbls[:, headIdx] = predictedLbls
			#move offset past current head
			headOffset += numHeadLbls

		#
		return lbls

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