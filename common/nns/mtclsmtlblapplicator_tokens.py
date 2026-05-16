from typing import List, Tuple, Any, Self

import torch
import numpy as np

from transformers import PreTrainedModel, PreTrainedTokenizerFast


class TokenSeqLabelPreds:
	"""
	Label predictions for a sequence of tokens.
	"""

	"""
	Confidences of all labels for each token. shape=(numTokens, numTotalLabels).
	"""
	confs : torch.Tensor

	"""
	Winning labels. One per head for each token. shape=(numTokens, numHeads)
	"""
	lbls : torch.Tensor

	"""
	Token offsets. (start, length), shape=(numTokens, 2)
	"""
	offsets : torch.Tensor

	"""
	Token IDs. shape=(numTokens)
	"""
	ids : torch.Tensor


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

	"""
	Token IDs. shape=(numTokens)
	"""
	tokenIds : torch.Tensor

	"""
	Token-word assignments. shape=(numTokens)
	"""
	tokenWords : torch.Tensor


class MulticlassMultilabelApplicator:
	"""
	Helpers for applying multiclass-multilabel models to text analysis. This is for token based models.
	"""

	def predicWordLbls(self, 
			text : str, 
			tokenizer : PreTrainedTokenizerFast, 
			model : PreTrainedModel, 
			overlapRatio = 1/3
	) -> WordSeqLabelPreds | None:
		"""
		Predict labels for each word of given text.

		:param text: Text to predict the labels for.
		:param tokenizer: Tokenizer to use.
		:param model: Model to use.
		:param overlapRatio: Chunk overlap ratio for texts longer than the maximum input length of the model.

		:return: Word label predictions for given text. None if text has no words.
		"""
		
		#get head configuration from the model 
		numLblsPerHead = model.getNumLblsPerHead()

		#predict labels on tokens
		tknLblPreds = self.predictTknLbls(text=text, tokenizer=tokenizer, model=model, overlapRatio=overlapRatio)

		#text has no tokens? abort
		if tknLblPreds is None:
			return None

		#assign tokens to words
		tknWords = self._markWords(tknLblPreds.ids, tokenizer)
		numWords = tknWords.max().item()+1

		#merge tokens into words
		confs = torch.zeros((numWords, sum(numLblsPerHead)), dtype=torch.float32)
		offsets = torch.zeros((numWords, 2), dtype=torch.int32)

		for wrdIdx in range(numWords):
			#find which tokens correspond to current word, extract related information
			isWrdTkn = (tknWords == wrdIdx)
			wrdTknConfs = tknLblPreds.confs[isWrdTkn]
			wrdTknOffsets = tknLblPreds.offsets[isWrdTkn]

			#calculate word label confidences and character offsets
			wrdConf = wrdTknConfs.sum(dim=0) / torch.tensor(wrdTknConfs.shape[0], dtype=torch.float32)
			wrdStart = wrdTknOffsets[0, 0].item()
			wrdLen = wrdTknOffsets[-1, 0].item() + wrdTknOffsets[-1, 1].item() - wrdStart

			#write to result sequences
			confs[wrdIdx] = wrdConf
			offsets[wrdIdx] = torch.tensor([wrdStart, wrdLen], dtype=torch.int32)

		#extract word labels
		lbls = self._extractWinningLabels(confs, numLblsPerHead)

		#
		res = WordSeqLabelPreds()
		res.confs = confs
		res.lbls = lbls
		res.offsets = offsets
		res.tokenIds = tknLblPreds.ids
		res.tokenWords = tknWords

		return res

	def _markWords(self, tkns : torch.Tensor, tokenizer : PreTrainedTokenizerFast) -> torch.Tensor:
		"""
		Mark word in the given token sequence by assigning matching numbers to the tokens that belong to the same word. Special tokens will always be labeled as separate words, one word per token.
		
		:param tkns: A sequence of token ids.
		:param tokenizer: Tokenizer to use.
		
		:return: A sequence where each corresponding token index is marked by a corresponding word number. Word numbers are in range [0;inf), the sequence will be continous.
		"""

		#create result array
		wrds = torch.zeros(tkns.shape[0], dtype=torch.int32)

		#mark words, this may build non-continouse sequence if special tokens are encountered
		wrd = 0
		for tknIdx, tknId in enumerate(tkns):
			#find if current token is special token
			isSpecialToken = tknId.item() in tokenizer.all_special_ids

			#current token is special? give it a separate word
			if isSpecialToken:
				wrd += 1 #end previous word, if any
				wrds[tknIdx] = wrd
				wrd += 1 #start new word
			#normal token
			else:
				#find if current token starts a word
				tknText = tokenizer.convert_ids_to_tokens(tknId.item())
				isWordStart = tknText.startswith("▁")

				#current token starts a word? advance word number
				if isWordStart:
					wrd += 1

				#assign current token to current word
				wrds[tknIdx] = wrd

		#remap to continous word sequence
		contWrdSeq = torch.zeros_like(wrds)
		curWrd = 0
		for idx in range(wrds.shape[0]):
			#word boundary detected? advance current word
			if idx != 0 and wrds[idx-1] != wrds[idx]:
				curWrd += 1
			
			#assign current word to current token
			contWrdSeq[idx] = curWrd

		#
		return contWrdSeq


	def predictTknLbls(
			self, 
			text : str, 
			tokenizer : PreTrainedTokenizerFast, 
			model : PreTrainedModel, 
			overlapRatio = 1/3
	) -> TokenSeqLabelPreds | None:
		"""
		Predict labels for each token of given text.

		:param text: Text to predict the labels for.
		:param tokenizer: Tokenizer to use.
		:param model: Model to use.
		:param overlapRatio: Chunk overlap ratio for texts longer than the maximum input length of the model.

		:return: Token label predictions for given text. None if text has no tokens.
		"""

		#text is empty? return empty result
		if (text is None) or (text.strip() == ""):
			return None
		
		#measure input length in tokens
		enc = tokenizer(
			text=text, 
			padding='do_not_pad', 
			truncation=False,
			return_attention_mask=False
		).encodings[0]
		textLenInTkns = len(enc.ids)

		#get head configuration from the model 
		numLblsPerHead = model.getNumLblsPerHead()

		#get tokenization results for each chunk of the text
		chunkResults = self._chunkAndPredictTknLbls(text=text, tokenizer=tokenizer, model=model, overlapRatio=overlapRatio)

		#allocate storage for merged confidences
		confs = torch.zeros((textLenInTkns, sum(numLblsPerHead)), dtype=torch.float32)
		
		#extract token offsets, convert second offset member to length instead of index
		offsets = torch.tensor(enc.offsets, dtype=torch.int32)
		offsets[:, 1] -= offsets[:, 0]

		#etract token ids
		ids = torch.tensor(enc.ids, dtype=torch.int32)

		#remove entries for special tokens from results; special tokens have length of 0
		isSpecialTkn = (offsets[:, 1] == 0)
		confs = confs[isSpecialTkn == False]
		offsets = offsets[isSpecialTkn == False]
		ids = ids[isSpecialTkn == False]
		
		#merge token label confidences into results
		numTokensPerPos = torch.zeros(confs.shape[0], dtype=torch.float32)

		for chunkIdx, chunkRes in enumerate(chunkResults):
			#remove special tokens from chunk results, this should also remove padding tokens
			isSpecialTkn = (chunkRes.offsets[:, 1] == 0)
			chunkConfs = chunkRes.confs[isSpecialTkn == False]
			chunkOffsets = chunkRes.offsets[isSpecialTkn == False]

			#find offset of the first token in the chunk
			for idxOffset in range(confs.shape[0]):
				if offsets[idxOffset, 0] == chunkOffsets[0, 0]:
					break

			#add chunk label confidences onto final confidences
			confs[idxOffset:idxOffset+chunkConfs.shape[0]] += chunkConfs
			numTokensPerPos[idxOffset:idxOffset+chunkConfs.shape[0]] += 1

		#derive average label confidences for post-merge result
		confs /= numTokensPerPos.unsqueeze(0).transpose(0, 1).expand([-1, confs.shape[1]])

		#extract post-merge token labels
		lbls = self._extractWinningLabels(confs, numLblsPerHead)

		#
		res = TokenSeqLabelPreds()
		res.confs = confs
		res.lbls = lbls
		res.offsets = offsets
		res.ids = ids

		return res

	def _chunkAndPredictTknLbls(self, 
			text : str, 
			tokenizer : PreTrainedTokenizerFast, 
			model : PreTrainedModel, 
			overlapRatio = 1/3
	) -> List[TokenSeqLabelPreds]:
		"""
		Predict labels for each token of given text. This will return results for each chunk.

		:param text: Text to predict the labels for.
		:param tokenizer: Tokenizer to use.
		:param model: Model to use.
		:param overlapRatio: Chunk overlap ratio for texts longer than the maximum input length of the model.

		:return: A list of token label preditions, one for each chunk.
		"""

		#text is empty? return empty result
		if (text is None) or (text.strip() == ""):
			return list()
		
		#model in training mode? set it to evaluation mode
		if model.training:
			model.eval()

		#get maximum input sequence length from the model
		maxInputLenInTkns = model.getMaxInputLen()

		#get head configuration from the model 
		numLblsPerHead = model.getNumLblsPerHead()

		#tokenize text into chunks that model can process, with given overlap
		enc = tokenizer(
			text=text, 
			padding='max_length', 
			max_length=min(maxInputLenInTkns, tokenizer.model_max_length), #use the smaller input length to be safe
			truncation=True,
			return_overflowing_tokens=True,
			stride=int(min(maxInputLenInTkns, tokenizer.model_max_length) * overlapRatio)
		).encodings[0]
		encs = [enc] + enc.overflowing

		#use the model to get label predictions for each chunk
		result : List[TokenSeqLabelPreds] = []

		with torch.inference_mode():
			for encIdx, enc in enumerate(encs):
				#extract input tensors from the text encoding, add batch dimension
				tknIds = torch.tensor(enc.ids, dtype=torch.int32, device=model.device).unsqueeze(dim=0)			
				tknMask = torch.tensor(enc.attention_mask, dtype=torch.int8, device=model.device).unsqueeze(dim=0)

				#invoke the model to get label confidences, move to cpu, remove batch dimension
				tknLblConfs : torch.Tensor = model(inputTknIds=tknIds, inputTknMask=tknMask)
				tknLblConfs = tknLblConfs.to(torch.device("cpu")).squeeze()

				#convert label confidences to token labels, one winning label per head	
				tknLbls = self._extractWinningLabels(tknLblConfs, numLblsPerHead)

				#add to results
				chunkRes = TokenSeqLabelPreds()

				chunkRes.confs = tknLblConfs
				chunkRes.lbls = tknLbls
				
				offsets = torch.tensor(enc.offsets, dtype=torch.int32)
				offsets[:, 1] -= offsets[:, 0] #convert second offset member to length instead of index
				chunkRes.offsets = offsets

				chunkRes.ids = torch.tensor(enc.ids, dtype=torch.int32)

				result.append(chunkRes)

		#
		return result

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

	


