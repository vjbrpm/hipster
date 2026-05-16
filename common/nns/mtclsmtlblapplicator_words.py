from typing import List, Tuple, Any, Self

import torch
import numpy as np

from transformers import PreTrainedModel, PreTrainedTokenizerFast
from nns.embedders import Embedder

class WordSeqLabelPreds:
	"""
	Label predictions for a sequence of words.
	"""

	"""
	Confidences of all labels for each word. shape=(numWords, numTotalLabels).
	"""
	confs : torch.Tensor

	"""
	Winning labels. One per head for each word. shape=(numWords, numHeads)
	"""
	lbls : torch.Tensor

	"""
	Word offsets. (start, length), shape=(numWords, 2)
	"""
	offsets : torch.Tensor


class MulticlassMultilabelApplicator:
	"""
	Helpers for applying multiclass-multilabel models to text analysis. This is for word based models that consume word level embeddings derived by averaging embeddings of corresponding tokens.
	"""

	def predictLbls(
			self, 
			text : str, 
			tokenizer : PreTrainedTokenizerFast, 
			model : PreTrainedModel,
			embedder: Embedder,
			overlapRatio = 1/3
	) -> WordSeqLabelPreds | None:
		"""
		Predict labels for each token of given text.

		:param text: Text to predict the labels for.
		:param tokenizer: Tokenizer to use.
		:param model: Model to use.
		:param embedder: Embedder to use.
		:param overlapRatio: Chunk overlap ratio for texts longer than the maximum input length of the model.

		:return: Token label predictions for given text. None if text has no tokens.
		"""

		#text is empty? return empty result
		if (text is None) or (text.strip() == ""):
			return None

		#get head configuration from the model 
		numLblsPerHead = model.getNumLblsPerHead()

		#get tokenization results for each chunk of the text
		chunkResults = self._chunkAndPredictWrdLbls(
			text=text, tokenizer=tokenizer, model=model, embedder=embedder, overlapRatio=overlapRatio
		)

		#find the upper limit for storage
		numWrdsUpperLimit = sum([cr.confs.shape[0] for cr in chunkResults])

		#allocate storage for merged results
		confs = torch.zeros((numWrdsUpperLimit, sum(numLblsPerHead)), dtype=torch.float32)
		offsets = torch.zeros((numWrdsUpperLimit, 2), dtype=torch.int32)
		
		#merge chunks into one sequence
		numTokensPerPos = torch.zeros(confs.shape[0], dtype=torch.float32)

		for chunkIdx, chunkRes in enumerate(chunkResults):
			#remove special tokens from chunk result, this should also remove padding tokens
			isSpecialTkn = (chunkRes.offsets[:, 1] == 0)
			chunkConfs = chunkRes.confs[isSpecialTkn == False]
			chunkOffsets = chunkRes.offsets[isSpecialTkn == False]

			#find offset of the first token in the chunk, use 0 for the first chunk
			idxOffset = 0
			if chunkIdx > 0:
				for idxOffset in range(confs.shape[0]):
					if offsets[idxOffset, 0] == chunkOffsets[0, 0]:
						break

			#add chunk label confidences onto final confidences
			confs[idxOffset:idxOffset+chunkConfs.shape[0]] += chunkConfs
			numTokensPerPos[idxOffset:idxOffset+chunkConfs.shape[0]] += 1

			#paste chunk word offsets onto final results
			offsets[idxOffset:idxOffset+chunkOffsets.shape[0]] = chunkOffsets

		#remove unused storage positions, 0 length indicates offset position was not written to
		isUsed = (offsets[:, 1] > 0)
		confs = confs[isUsed]
		numTokensPerPos = numTokensPerPos[isUsed]
		offsets = offsets[isUsed]

		#derive average label confidences for post-merge result
		confs /= numTokensPerPos.unsqueeze(0).transpose(0, 1).expand([-1, confs.shape[1]])

		#extract post-merge token labels
		lbls = self._extractWinningLabels(confs, numLblsPerHead)

		#
		res = WordSeqLabelPreds()
		res.confs = confs
		res.lbls = lbls
		res.offsets = offsets

		return res
	
	def _chunkAndPredictWrdLbls(
		self, 
		text : str, 
		tokenizer : PreTrainedTokenizerFast, 
		model : PreTrainedModel, 
		embedder: Embedder,
		overlapRatio = 1/3
	) -> List[WordSeqLabelPreds]:
		"""
		Predict labels for each token of given text. This will return results for each chunk.

		:param text: Text to predict the labels for.
		:param tokenizer: Tokenizer to use.
		:param model: Model to use.
		:param embedder: Embedder to use.
		:param overlapRatio: Chunk overlap ratio for texts longer than the maximum input length of the model.

		:return: A list of token label preditions, one for each chunk.
		"""

		#text is empty? return empty result
		if (text is None) or (text.strip() == ""):
			return list()

		#get head configuration from the model 
		numLblsPerHead = model.getNumLblsPerHead()

		#build chunked word embeddings with masks and offfsets
		chunkedEmbeds = self._chunkAndBuildWrdEmbeddings(
			text=text, tokenizer=tokenizer, embedder=embedder, overlapRatio=overlapRatio
		)

		#use the model to get label predictions for each chunk
		result : List[WordSeqLabelPreds] = []

		with torch.inference_mode():
			for chunkIdx, (embeds, mask, offsets) in enumerate(chunkedEmbeds):
				#convert inputs to tensors
				embeds = torch.tensor(embeds, dtype=torch.float32, device=model.device)
				mask = torch.tensor(mask, dtype=torch.int16, device=model.device)
				offsets = torch.tensor(offsets, dtype=torch.int32, device=torch.device("cpu"))

				#remove padding
				numWrds = mask.sum().item()
				
				mask = mask[:numWrds]
				embeds = embeds[:numWrds]
				offsets = offsets[:numWrds]

				#add batch dimension
				embeds = embeds.unsqueeze(dim=0)			
				mask = mask.unsqueeze(dim=0)

				#invoke the model to get label confidences, move to cpu, remove batch dimension
				wrdLblConfs : torch.Tensor = model(inputTknIdsOrEmbeds=embeds, inputTknMask=mask)
				wrdLblConfs = wrdLblConfs.to(torch.device("cpu")).squeeze()

				#convert label confidences to token labels, one winning label per head	
				tknLbls = self._extractWinningLabels(wrdLblConfs, numLblsPerHead)

				#add to results
				chunkRes = WordSeqLabelPreds()

				chunkRes.confs = wrdLblConfs
				chunkRes.lbls = tknLbls

				offsets[:, 1] -= offsets[:, 0] #convert second offset member to length instead of index
				chunkRes.offsets = offsets

				result.append(chunkRes)

		#
		return result
	
	def _chunkAndBuildWrdEmbeddings(self, 
		text : str, 
		tokenizer : PreTrainedTokenizerFast, 
		embedder: Embedder,
		overlapRatio = 1/3
	) -> List[Tuple[torch.tensor, torch.tensor, torch.tensor]]:
		"""
		Build word embeddings for a given text. If text results in word sequence longer than the maximum input length of the model, this will return multiple embeddings chunks with given overlap ratio.

		:param text: Text to produce embeddings for.
		:param tokenizer: Tokenizer to use.
		:param embedder: Embedder to use.
		:param overlapRatio: Chunk overlap ratio for texts longer than the maximum input length of the model.

		:return: A list of chunks where each chunk is tuple(embeddings, mask, offsets). Each 'offsets' is tensor(num-words, 2) where for each row, index 0 is start character index (0 based) and index 1 is end character index (0 based).
		"""

		results : List[Tuple[torch.tensor, torch.tensor, torch.tensor]] = list()

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
			results.append((seqWrdEmbeds, seqWrdMask, seqWrdOffsets))

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
