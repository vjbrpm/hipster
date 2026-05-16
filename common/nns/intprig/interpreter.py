from typing import Dict, List, Tuple, Any, Self
import torch
import torch.autograd.functional as taf

import captum

from transformers import PreTrainedModel, PreTrainedTokenizerFast
from nns.embedders import Embedder

from .inputprocessor_words import InputProcessor, ChunkDesc

class Interpreter:
	"""
	Integrated gradients based interpreter for multiclass-multilabel networks. Builds a token influence matrix where each row corresponds to an output token and each column corresponds to an input token. Cells correspond to an influence of a corresponding input token to a confidence of a winning label of a corresponding output token. Positive value is reinforcing, negative weakening and zero is neutral.

	In case of neural network with multiple heads, each head gets its own output matrix.
	"""

	numIgSteps : int
	"""
	How many interpolations steps should the IG method take.
	"""

	def __init__(self, numIgSteps : int = 100):
		"""
		Constructor.
		:param numIgSteps: How many interpolations steps should the IG method take.
		"""
		self.numIgSteps = numIgSteps

	def run(
		self,
		text : str, 
		inpproc : InputProcessor, 
		model : PreTrainedModel, 
		tokenizer : PreTrainedTokenizerFast, 
		embedder: Embedder,
		normalize = True,
		overlapRatio = 1/3
	) -> List[torch.tensor]:
		"""
		Runs the interpretter for a given model over given input.
		
		:param text: Text to produce embeddings for.
		:param inpproc: Input processor to use.
		:param model: Model to use.
		:param tokenizer: Tokenizer to use.
		:param embedder: Embedder to use. This should produce initial embeddings to reflect input tokens/words, not mid-level embeddings from hidden layers. Also model must be able to consume the embeddings produced.
		:param normalize: True to normalize in-row results to range [-1;1], False to return raw results.
		:param overlapRatio: Chunk overlap ratio for texts longer than the maximum input length of the model.

		:return: A list of influence matrices. One for each head of the model.
		"""

		#get head configuration from the model 
		numLblsPerHead = model.getNumLblsPerHead()

		#text is empty? do nothing
		if (text is None) or text.strip() == "":
			return [torch.zeros((0,0), dtype=torch.float32)] * len(numLblsPerHead)
		
		#build input chunks for the NN
		inpChunks = inpproc.buildInputChunks(
			text=text, tokenizer=tokenizer, embedder=embedder, overlapRatio=overlapRatio
		)

		#get winning labels for the whole input sequence
		lblsAll = inpproc.predictWinningLabels(inputChunks=inpChunks, model=model)

		#initialize output and working matrices
		inflAll : List[torch.tensor] = []
		perPosItemsAll : List[torch.tensor] = []

		seqLen = lblsAll.shape[0]
		for headIdx in range(len(numLblsPerHead)):
			#shape = (outSeqLen, inSeqLen)
			inflAll.append(torch.zeros((seqLen, seqLen), dtype=torch.float32))
			perPosItemsAll.append(torch.zeros((seqLen, seqLen), dtype=torch.float32))

		#accumulate each chunk information to influence matrices
		for chunkIdx, chunk in enumerate(inpChunks):
			#get winning labels for the chunk
			chunkLen = chunk.embeds.shape[0]
			lblsInpChunk = lblsAll[chunk.idxOffset:(chunk.idxOffset + chunkLen)]

			#run interpretation for the chunk
			headResAll = self._interpretChunk(
				chunk=chunk, inpproc=inpproc, model=model, winingLbls=lblsInpChunk
			)

			#accumulate influence stats for each head
			for headIdx in range(len(numLblsPerHead)):
				headRes = headResAll[headIdx]

				#additively paste chunk related part onto head influence matrix
				inflHead = inflAll[headIdx]
				inflHead[
					chunk.idxOffset:(chunk.idxOffset + chunkLen), 
					chunk.idxOffset:(chunk.idxOffset + chunkLen)
				] += headRes
				inflAll[headIdx] = inflHead

				perPosItemsHead = perPosItemsAll[headIdx]
				perPosItemsHead[
					chunk.idxOffset:(chunk.idxOffset + chunkLen), 
					chunk.idxOffset:(chunk.idxOffset + chunkLen)
				] += 1
				perPosItemsAll[headIdx] = perPosItemsHead

		#finalize results
		for headIdx in range(len(numLblsPerHead)):
			#read head result from storage
			inflHead : torch.Tensor = inflAll[headIdx]

			#average over overlapping accumulations			
			inflHead /= perPosItemsAll[headIdx]

			#normalize to range [-1;1] across rows (outputs)
			if normalize:
				colMax = inflHead.abs().max(dim=1)[0]
				colMax[colMax <= torch.finfo(colMax.dtype).tiny] = 1 #prevent division by zero and equivalent
				inflHead /= colMax.unsqueeze(0).transpose(0, 1).expand(inflHead.shape)

			#write back to storage
			inflAll[headIdx] = inflHead

		#
		return inflAll
		
	def _interpretChunk(
			self, 
			chunk : ChunkDesc,
			inpproc : InputProcessor, 
			model : PreTrainedModel, 
			winingLbls : torch.Tensor
	) -> List[torch.Tensor]:
		"""
		Calculate influence scores for a single input chunk.

		:param chunk: Chunk to calculate influence scores for.
		:param inpproc: Input processor to use.
		:param model: Model to use.
		:param winingLbls: Wining labels for the chunk. shape=(inSeqLen, numHeads)

		:return: A lisf of influence matrices, one for each head of the model.
		"""

		result : List[torch.Tensor] = list()

		#build a forward function
		def evalFunc(embeds : torch.tensor) -> torch.tensor:
			mask = chunk.mask.unsqueeze(0) #add batch dimension to a mask
			res = inpproc.predicLabelConfsForChunk(embeds=embeds, mask=mask, model=model)
			return res

		#run attribution for every head
		ig = captum.attr.IntegratedGradients(evalFunc)

		numHeads = winingLbls.shape[1]
		for headIdx in range(numHeads):
			#create empty influence matrix for the head, shape=(outSeqLen, inSeqLen)
			chunkLen = chunk.embeds.shape[0]
			headInfl = torch.zeros((chunkLen, chunkLen), dtype=torch.float32)

			#prepare data for IG passes by adding batch dimension and moving to correct device
			embeds = chunk.embeds.unsqueeze(0).to(device=model.device) 
			baselines = torch.zeros_like(embeds, device=model.device)

			#fill in influence matrix for the head, by accumulating relevant results from each output token
			for outTokenIdx in range(chunkLen):
				#get position of wining output label for current token and current head
				target = (
					outTokenIdx, 
					winingLbls[outTokenIdx, headIdx].to(dtype=torch.int32).item()
				)

				#run attribution
				attrs, delta = ig.attribute(
					embeds,
					baselines=baselines,
					target=target, 
					internal_batch_size=16,
					return_convergence_delta=True,
					n_steps=self.numIgSteps
				)

				#merge embedding element attributions into single value
				attrs = attrs.squeeze()
				attrs = attrs.mean(dim=-1)

				#store attribution result for current output token
				headInfl[outTokenIdx] = attrs

			#add to results
			result.append(headInfl)

		#
		return result