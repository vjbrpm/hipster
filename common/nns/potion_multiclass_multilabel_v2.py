from typing import List, Tuple, Any, Self, override
import os.path

import torch
from torch import nn as tnn
import torch.nn.functional as tnnf

import lightning as lit
import torchmetrics.classification as tmcls

from transformers import XLMRobertaTokenizerFast

import matplotlib
import matplotlib.pyplot as plt


class Tokenizer(XLMRobertaTokenizerFast):
	def load(modelMaxLength: int) -> Self:
		"""
		Load tokenizer and set given 'model_max_length'.
		:param modelMaxLength: Valueu of 'model_max_length' property to set.
		:return: A new instance of the tokenizer.
		"""

		#load the tokenizer
		loadPath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'potion-artifacts/tokenizer'))
		tokenizer = XLMRobertaTokenizerFast.from_pretrained(loadPath, local_files_only=True)

		#up the class to the typeof(Self) instead of base class
		tokenizer.__class__ = Tokenizer

		#set model max length as given
		tokenizer.model_max_length = modelMaxLength

		#
		return tokenizer
	
	def loadPreconfigured(loadPath : str):
		"""
		Load a preconfigured tokenizer saved with 'save_pretrained()'.
		:param loadPath: Path to load from.
		:return: A new instance of the tokenizer.
		"""

		tokenizer = XLMRobertaTokenizerFast.from_pretrained(loadPath, local_files_only=True)
		tokenizer.__class__ = Tokenizer

		#
		return tokenizer
	
	def __call__(self, *args, **kwargs):
		"""
		This a patch for XLMRobertaTokenizerFast.__call__ that adds 'add_special_tokens=False' to the kwargs because we do not support special tokens (except for <pad> and <unk>) in the embeddings of the model.
		"""
		
		#patch kwargs to disable generation of special tokens
		if kwargs == None:
			kwargs = dict()
		kwargs["add_special_tokens"] = False

		#delegate to superclass
		return super().__call__(*args, **kwargs)


class RoPE:
	"""
	Applies rotary positional encoding to given matrix. This is based on https://arxiv.org/pdf/2104.09864,  https://learnopencv.com/rope-position-embeddings .
	"""

	"""
	Device to store the cached tensors on.
	"""
	device : torch.device = torch.device("cpu")

	def __init__(self, dimHead : int, seqLen : int):
		"""
		Constructor.
		:param dimHead: Dimension of the head.
		:param seqLen: Sequence length.
		"""

		super().__init__()
		self.precompute(dimHead, seqLen)

	def precompute(self, dimHead : int, seqLen : int):
		"""
		Precompute cached tensors.
		:param dimHead: Dimension of the head.
		:param seqLen: Sequence length.
		"""

		if dimHead % 2 != 0:
			raise AssertionError("Argument 'dimHead' must be evenly divisible by 2.")
		
		if seqLen <= 0:
			raise AssertionError("Argument 'seqLen' must be >= 1.")

		#compute thetas
		base = torch.full((dimHead//2,), 10000, dtype=torch.float32)
		exponents = torch.arange(dimHead//2, dtype=torch.float32) * 2 / dimHead
		thetas = base.pow(exponents)

		#compute idx*thetas for each idx in sequence
		idxMatrix = (
			#create a vector of indices reshape to shape=(seqLen, 1)
			torch.arange(seqLen, dtype=torch.float32).reshape((seqLen, 1))
			#expand to shape=(seqLen, dimHead) by repeating collumns
			.expand((-1, dimHead))
		)
		thetaMatrix = (
			#repeat every theta value twice
			thetas.reshape((thetas.shape[0], 1)).expand((-1, 2)).flatten()
			#expand to shape=(seqLen, dimHead) by repeating rows
			.expand((seqLen, -1))
		)
		idxThetas = idxMatrix * thetaMatrix

		#compute cosines and sines
		self.cosines = idxThetas.cos()
		self.sines = idxThetas.sin()

		#apply negations to even sines, this will transfer to corresponding target elements when multilplying
		self.sines[:, torch.arange(0, dimHead, 2)] *= -1

		#build index to extract target elements from the rows of input for sine application
		self.rowIdxsSin = torch.arange(0, dimHead)
		for i in range(0, dimHead, 2):
			self.rowIdxsSin[[i, i+1]] = self.rowIdxsSin[[i+1, i]]

		#move new tensors to current devie
		self._moveToDevice()

	def _moveToDevice(self):
		"""
		Moves all cached tensors to current device.
		"""

		self.cosines = self.cosines.to(self.device)
		self.sines = self.sines.to(self.device)
		self.rowIdxsSin = self.rowIdxsSin.to(self.device)

	def to(self, dev: torch.device):
		"""
		Set device for cached tensors.
		"""
		self.device = dev
		self._moveToDevice()

	def apply(self, input : torch.Tensor) -> torch.Tensor:
		"""
		Apply RoPE to the given input.
		:param input: Input matrix. shape=(batch, seqLen, dimHead)
		:return: Input matrix with RoPE applied.
		"""
		seqLen = input.shape[1] #actual sequence length might be smaller than the maximum one
		cosPart = input * self.cosines[:seqLen]
		sinPart = input[:, :, self.rowIdxsSin] * self.sines[:seqLen]
		rope = cosPart + sinPart

		#
		return rope


class MultiHeadAttention(tnn.Module):
	"""
	Computes multi-head attention. This is adjusted from https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html . We are not using jagged tensors because they do not support jagged to jagged addition for two instance created via torch.nested.nested_tensor() even when shapes are compatible. Therefore jagged tensors can not be used to implement fast version of RoPE embeddings.
	"""

	def __init__(
		self,
		dimHidden: int,
		numHeads: int,		
		dropout: float,
		rope : RoPE
	):
		"""
		Constructor.
		
		:param dimHidden: Size of hidden dimension.
		:param numHeads: Number of heads. dimHidden % numHeads == 0 must hold.
		:param dropout: Dropout probability for internal dropout layers.
		:param rope: RoPE embeddings applicator.		
		"""

		super().__init__()
		
		if dimHidden % numHeads != 0:
			raise AssertionError("Value of argument 'dimHidden' must be divisible by value of argument 'dimHeads' without a reminder.")

		self.hiddenDim = dimHidden
		self.numHeads = numHeads
		self.dropout = dropout
		self.rope = rope
		
		self.qkvProjPacked = tnn.Linear(dimHidden, dimHidden * 3)
		
		self.outProj = tnn.Linear(dimHidden, dimHidden)

	def forward(self, input : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
		"""
		Run the forward pass.
		
		:param input: Embeddings. shape=(batch, seqLen, hiddenDim)
		:param mask: Attention mask. Must be of type torch.bool. True enables, False disables. shape=(batch, 1, 1, seqLen)		
		:return: Attention result. shape=(batch, seqLen, hiddenDim)
		"""
		
		#apply packed input projection to derive Q, K, V matrices
		qkvProjPackedRes = self.qkvProjPacked(input)
		query, key, value = torch.chunk(qkvProjPackedRes, 3, dim=-1)

		#apply RoPE to Q and K
		query = self.rope.apply(query)
		key = self.rope.apply(key)

		#split Q, K, V into heads; (batch, seqLen, dimHidden)->(batch, seqLen, numHeads, dimHead)->(batch, numHeads, seqLen, dimHead)
		dimHead = self.hiddenDim // self.numHeads
		query = query.unflatten(-1, [self.numHeads, dimHead]).transpose(1, 2)
		key = key.unflatten(-1, [self.numHeads, dimHead]).transpose(1, 2)
		value = value.unflatten(-1, [self.numHeads, dimHead]).transpose(1, 2)

		#run SDPA, (batch, numHeads, seqLen, dimHead)
		resSdpaPerHead = tnnf.scaled_dot_product_attention(
			query, key, value, attn_mask=mask, dropout_p=self.dropout
		)
		
		#concatenate head results (batch, numHeads, seqLen, dimHead)->(batch, seqLen, numHeads, dimHead)->(batch, seqLen, dimHidden)
		resSdpa = resSdpaPerHead.transpose(1, 2).flatten(-2)

		#apply output projection
		resOutProj = self.outProj(resSdpa)

		#
		return resOutProj


class TransformerEncoderLayer(tnn.Module):
	"""
	And encoder layer. This is adjusted from https://github.com/mikaylagawarecki/transformer_tutorial_accompaniment/blob/main/te_layer.py referenced by https://docs.pytorch.org/tutorials/intermediate/transformer_building_blocks.html .
	"""

	def __init__(
		self,
		dimHidden : int,
		numHeads : int,
		dropout : float,
		rope : RoPE
	):
		"""
		Constructor.
		
		:param dimHidden: Size of hidden dimension.
		:param numHeads: Number of heads. dimHidden % numHeads = 0 must hold.
		:param dropout: Dropout probability for internal dropout layers.
		:param rope: RoPE embeddings applicator.
		"""
		
		super().__init__()

		ffwExpansion = 2

		self.norm1 = tnn.LayerNorm(dimHidden)
		self.attn = MultiHeadAttention(dimHidden=dimHidden, numHeads=numHeads, dropout=dropout, rope=rope)

		self.norm2 = tnn.LayerNorm(dimHidden)
		self.ffwLin1 = tnn.Linear(dimHidden, dimHidden * ffwExpansion)
		self.ffwDropout = tnn.Dropout(dropout)
		self.ffwLin2 = tnn.Linear(dimHidden * ffwExpansion, dimHidden)

		self.gegluPackedProj = tnn.Linear(dimHidden * ffwExpansion, dimHidden * ffwExpansion * 2)

	def _ffw(self, input : torch.Tensor) -> torch.Tensor:
		"""
		Runs feed forward network.

		:param input: shape=(batch, seqLen, hiddenDim)
		:return: shape=(batch, seqLen, hiddenDim)
		"""

		resFfwLin1 = self.ffwLin1(input)
		resGeglu = self._geglu(resFfwLin1)		
		resFfwDropout = self.ffwDropout(resGeglu)
		resFfwLin2 = self.ffwLin2(resFfwDropout)

		#
		return resFfwLin2

	def _geglu(self, input : torch.Tensor) -> torch.Tensor:
		"""
		Runs GeGLU activation function. See https://arxiv.org/pdf/2002.05202, https://arxiv.org/pdf/1612.08083
		
		:param input: shape=(batch, seqLen, hiddenDim * ffwExpansion)
		:return: shape=(batch, seqLen, hiddenDim * ffwExpansion)
		"""

		resGegluPackedProj = self.gegluPackedProj(input)
		resGegluGateProj, resGegluLinProj = torch.chunk(resGegluPackedProj, 2, dim=-1)
		resGegluGate = tnnf.gelu(resGegluGateProj)
		resGeglu = resGegluGate * resGegluLinProj

		#
		return resGeglu

	def forward(self, input : torch.Tensor, mask : torch.Tensor) -> torch.Tensor:
		"""
		Run the forward pass.
		
		:param input: Embeddings. shape=(batch, seqLen, hiddenDim)
		:param mask: Attention mask. Must be of type torch.bool. True enables, False disables. shape=(batch, 1, 1, seqLen)
		:return: Encoder result. shape=(batch, seqLen, hiddenDim)
		"""
		
		#run self attention: prenorm, self-attn, shortcut
		resNorm1 = self.norm1(input)
		resSelfAttn = self.attn(input=resNorm1, mask=mask)
		resShortcut1 = input + resSelfAttn

		#run ffw network: prenorm, ffw, shortcut
		resNorm2 = self.norm2(resShortcut1)
		resFfw = self._ffw(resNorm2)
		resShortcut2 = resShortcut1 + resFfw
	
		#
		return resShortcut2


class MultiClassMultiLabelTokenLabeler(tnn.Module):
	"""
	This is an encoder only transformer built on top of 'minishlab/potion-multilingual-128M'. The model is both multiclass and multilabel with a separate head for each class. Inside each class, class labels are considered to be mutually-exclusive.
	"""

	"""
	Device used by this model.
	"""
	device : torch.device

	"""
	Number of labels supported.
	"""
	numLabels : List[int]

	"""
	Number of additional encoder layers.
	"""
	numAdditionalEncoderLayers : int

	"""
	Sequence length.
	"""
	seqLen : int

	def __init__(self, numLabels : List[int], numAdditionalEncoderLayers : int, seqLen : int):
		"""
		Constructor.
		:param numLabels: Number of labels to support for each class. Length of a list defines number of classes.
		:param numAdditionalEncoderLayers: Number of additional encoder layers to add on top of base model. Base model will be fixed during training, so only additional encoder layers and final classification head will get trained.
		:param seqLen: Sequence length. In tokens.
		"""

		super().__init__()

		#set defaults
		self.device = torch.device("cpu")
				
		#validate inputs
		if len(numLabels) < 1:
			raise AssertionError("Argument 'numLabels' must be a list with length >= 1.")
		
		if any([it <= 0 for it in numLabels]):
			raise AssertionError("Argument 'numLabels' must have all entries >= 1.")
		
		if numAdditionalEncoderLayers < 1:
			raise AssertionError("Argument 'numAdditionalEncoderLayers' must be >= 1.")
		
		if seqLen < 1:
			raise AssertionError("Argument 'seqLen' must be >= 1.")
		
		#store inputs
		self.numLabels = numLabels
		self.numAdditionalEncoderLayers = numAdditionalEncoderLayers
		self.seqLen = seqLen

		#load embeddings, make sure they a frozen
		loadPath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'potion-artifacts/embeddings/save.torch_save'))
		with open(loadPath, "rb") as file:
			self.embedding : tnn.Embedding = torch.load(file, weights_only=False)
		
		self.embedding.eval()
		for p in self.embedding.parameters(recurse=True):
			p.requires_grad = False

		#define internal configuration
		self.dimHidden = self.embedding.weight.shape[1]
		self.numEncoderHeads = 2
		self.dimEncoderHead = self.dimHidden // self.numEncoderHeads

		#check if internal configuration is compatible with base model
		if self.dimHidden % self.numEncoderHeads != 0:
			raise AssertionError(f"Chosen 'numEncoderHeads={self.numEncoderHeads}' is not compatible with 'dimHidden={self.dimHidden}'")

		#add post-transformation for embeddings, initialize to identity
		self.embProj = tnn.Linear(self.embedding.weight.shape[1], self.dimHidden)
		with torch.no_grad():
			self.embProj.weight.copy_(torch.eye(self.dimHidden, dtype=self.embProj.weight.dtype))
			self.embProj.bias.copy_(torch.zeros_like(self.embProj.bias))

		#create RoPE applicator
		self.rope = RoPE(self.dimHidden, self.seqLen)
		
		#add trainable additional encoder layers on top of base model
		self.addedEncoderLayers = tnn.ModuleList([
			TransformerEncoderLayer(
				dimHidden=self.dimHidden, numHeads=self.numEncoderHeads, dropout=0.1, rope=self.rope
			) 
			for _ in range(numAdditionalEncoderLayers)
		])

		#use custom classification heads for each set of mutually exclusive labels
		ffwExpansion = 4

		self.clsPackedFfwLin1 = tnn.Linear(self.dimHidden, self.dimHidden * ffwExpansion * len(numLabels))
		self.clsPackedFfwGelu = tnn.GELU()
		self.clsPackedFfwDropout = tnn.Dropout(p=0.1)
		
		self.clsFfw2Heads = tnn.ModuleList([
			tnn.Sequential(
				tnn.LayerNorm(self.dimHidden * ffwExpansion),
				tnn.Linear(self.dimHidden * ffwExpansion, numHeadLbls),
				tnn.Softmax(dim=-1)			
			)
			for numHeadLbls in numLabels
		])

	def _runClsHeads(self, input : torch.Tensor) -> torch.Tensor:
		"""
		Run classification heads.
		:param input: Sequence embeddings. shape=(batch, seqLen, dimHidden).
		:return: Token labels. shape=(batch, seqLen, sum(self.numLabels))
		"""

		#run first FFW layer for all heads in packed form
		resClsPackedFfwLin1 = self.clsPackedFfwLin1(input)
		resClsPackedGelu = self.clsPackedFfwGelu(resClsPackedFfwLin1)
		resPackedFfwDropout = self.clsPackedFfwDropout(resClsPackedGelu)

		#unpack results for each head from the first FFW layer
		resFfw1Unpacked = torch.chunk(resPackedFfwDropout, len(self.numLabels), dim=-1)

		#run second FFW layer for each head
		resClsHeads : List[torch.Tensor] = []
		for clsHeadIdx, (resFfw1Head, ffw2Head) in enumerate(zip(resFfw1Unpacked, self.clsFfw2Heads)):
			resFfw2Head = ffw2Head(resFfw1Head) #shape=(batchLen, seqLen, numHeadLabels)
			resClsHeads.append(resFfw2Head)

		#join results into a single tensor, shape=(batchLen, seqLen, sum(self.numLabels))
		resFinal = torch.cat(resClsHeads, dim=-1)

		#
		return resFinal

	def forward(self, inputTknIdsOrEmbeds : torch.Tensor, inputTknMask : torch.Tensor) -> torch.Tensor:
		"""
		Run a forward pass.
		
		:param inputTknIdsOrEmbeds: Input token ids or embeddings. Ids are indicated by shape=(batch, seqLen), embeddings by shape=(batch, seqLen, embedLen). Max sequence length limited by self.getMaxInputLen().
		:param inputTknMask: Input token mask. Same length as inputTknIdsOrEmbeds. 1 consider, 0 ignore.
		
		:return: Predictions for each label. A tensor in shape (batchLen, seqLen, sum(self.numLabels)). For each token a label vector is a concatenation of label vectors produced by individual heads. Heads are represented in the same order as in the self.numLabels list. Each head produces a softmax vector for the labels of the class it represents, same length as defined in a corresponding position of self.numLabels list.
		"""

		#ensure the tensors are on the correct device
		inputTknIdsOrEmbeds = inputTknIdsOrEmbeds.to(self.device)
		inputTknMask = inputTknMask.to(self.device)

		#embeddings not provided? run base model to produce them
		if len(inputTknIdsOrEmbeds.shape) == 2:
			resEmb = self.embedding(inputTknIdsOrEmbeds)
			tknEmbeds = self.embProj(resEmb)
		#embeddings provided, use them
		else:
			tknEmbeds = inputTknIdsOrEmbeds
		
		#make sure attention mask is of corect dtype and has shape=(batch, 1, 1, seqLen) to use in sdpa
		sdpaAttnMask = inputTknMask.to(dtype=torch.bool).unsqueeze(dim=1).unsqueeze(dim=1)
		
		#run added encoder layers, each layer has internal shortcut connections
		hiddenStates = tknEmbeds
		for encLayerIdx, encLayer in enumerate(self.addedEncoderLayers):
			hiddenStates = encLayer(input=hiddenStates, mask=sdpaAttnMask)

		#run classification heads
		resClsHeads = self._runClsHeads(hiddenStates)
		
		#
		return resClsHeads

	def train(self, mode : bool = True) -> "MultiClassMultiLabelTokenLabeler":		
		super().train(mode)

		#make sure embeddings are not in training mode, regardless
		self.embedding.eval()
		for p in self.embedding.parameters(recurse=True):
			p.requires_grad = False

		#
		return self

	def to(self, dev : torch.device) -> "MultiClassMultiLabelTokenLabeler":
		super().to(dev)

		#record which device we are on
		self.device = dev

		#make sure RoPE applicator is on correct device
		self.rope.to(dev)

		#
		return self
	
	def getMaxInputLen(self) -> int:
		"""
		Get maximum input length in tokens.		
		:return: Maximum input length in tokens.
		"""

		return self.seqLen
	
	def getNumLblsPerHead(self) -> List[int]:
		"""
		Get number of labels for each head.
		:return: A list with number of labels for each head.
		"""

		return self.numLabels


class MultiClassMultiLabelTokenLabelerLit(lit.LightningModule):
	"""
	This is a wrapper for MultiClassMultiLabelTokenLabeler for training with 'pytorch lightning'.
	"""

	"""
	Maximum lost boost coeficient for underrepresented labels.
	"""
	maxLossBoost : float

	def __init__(self, numLabels : List[int], numAdditionalEncoderLayers : int, seqLen : int, maxLossBoost : float = 2):
		"""
		Constructor.

		:param numLabels: Number of labels to support for each class. Length of a list defines number of classes.
		numAdditionalEncoderLayers. Number of additional encoder layers to add on top of base model. Base model will be fixed during training, so only additional encoder layers and final classification head will get trained.
		:param seqLen: Sequence length. In tokens.
		:param maxLossBoost: Maximum loss boost factor for heads of underrepresented labels. Must be >= 0. Head with full loss boost will have its loss scaled by (1+maxLossBoost).
		"""

		super().__init__()

		#validate inputs
		if maxLossBoost < 0:
			raise AssertionError("Argument 'maxLostBoost' is < 0.")

		#this is needed for constructor args to load correctly from checkpoints
		self.save_hyperparameters()

		#initialize base model
		self.model = MultiClassMultiLabelTokenLabeler(
			numLabels=numLabels, numAdditionalEncoderLayers=numAdditionalEncoderLayers, seqLen=seqLen
		)

		#add some metric trackers
		self.valConfMat = tmcls.MultilabelConfusionMatrix(num_labels=sum(numLabels))
		self.valF1 = tmcls.MultilabelF1Score(num_labels=sum(numLabels), average='none')

		#create metric history storage
		self.trainStepLosses = []
		self.valStepLosses = []
		self.histValF1 = []
		
		#store dimensions for figure logging
		px = 1/matplotlib.rcParams['figure.dpi']
		self.figSizeX = 1024 * 1.2 * px
		self.figSizeY = 768 * 1.2 * px

		#create storage for label sample counters
		self.numLblSamples = torch.zeros(sum(numLabels), dtype=torch.int32, requires_grad=False)

		#create loss scaling vector, set initial scaling to 1, since we do not have label sample counts yet
		self.headLossScaling = torch.ones(sum(numLabels), dtype=torch.float32, requires_grad=False)

		#save max loss boost parameter
		self.maxLossBoost = maxLossBoost

		#create scaling history storage
		self.histHeadLossScaling = []

	def to(self, dev : torch.device) -> "MultiClassMultiLabelTokenLabelerLit":
		super().to(dev)
		
		#this is not done automatically by pytorch
		self.model.to(dev)	
		self.numLblSamples = self.numLblSamples.to(dev)
		self.headLossScaling = self.headLossScaling.to(dev)

		#
		return self
	
	def configure_optimizers(self):
		optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
		return optimizer
	
	def training_step(self, batch, batchIdx):
		"""
		Runs a single training step/iteration.

		:param batch: A batch of training data. list(list(tknIds, tknMask, tknIdx), tknLbls). Where
			* tknIds is a tensor of token IDs, shape=(batchLen, seqLen) or tokenEmbeddings, shape=(batchLen, seqLen, embedDim);
			* tknMask is a tensor of attention masks, shape=(batchLen, seqLen);
			* tknIdx a tensor of target token indices, shape=(batchLen, 1);
			* tknLbl a tensor of labels for target tokens, shape=(batchLen, sum(numLabels)); presense of label is marked by 1 in a corresponding position, 0 marks non-presense; Neutral token is indicated by no labels being present.
			* batchIdx. Batch index in epoch.
			
		:return: Tensor for loss value.
		"""

		[[inputTknIdsOrEmbeds, inputTknMask, tknIdx], tknLbls] = batch

		#find out the longest sequence in the batch, trim everything to match and remove unnecessary padding
		maxSeqLen = inputTknMask.to(dtype=torch.int16).sum(dim=1).max()
		inputTknIdsOrEmbeds = inputTknIdsOrEmbeds[:, :maxSeqLen]
		inputTknMask = inputTknMask[:, :maxSeqLen]

		#ensure the tensors are on the correct device
		inputTknIdsOrEmbeds = inputTknIdsOrEmbeds.to(self.device)
		inputTknMask = inputTknMask.to(self.device)
		tknIdx = tknIdx.to(self.device)
		tknLbls = tknLbls.to(self.device)

		#make predictions for this batch
		resModel = self.model(inputTknIdsOrEmbeds=inputTknIdsOrEmbeds, inputTknMask=inputTknMask) #shape=(batchLen, seqLen, sum(self.numLabels))

		#extract confidence vectors for the labels of given tokens
		resTknLblConf = resModel[range(resModel.shape[0]), tknIdx, :] #shape=(batchLen, sum(self.numLabels))

		#find head losses, apply loss scaling (boosting), sum, derive batch average
		headLosses = tnnf.binary_cross_entropy(resTknLblConf, tknLbls, reduction='none')
		loss = (headLosses * self.headLossScaling).sum(dim=1).mean()

		#log and accumulate the loss
		lossForLog = loss.to("cpu").clone().detach() #we need clone here, in case model is running on cpu and to() does not make a copy	
		self.log("Training. BCE loss. Step.", lossForLog.item(), on_step=True, on_epoch=False)
		self.trainStepLosses.append(lossForLog)

		#add to label sample counts
		with torch.no_grad():
			self.numLblSamples = self.numLblSamples + tknLbls.sum(dim=0)

		#return the loss tensor for backpropagation
		return loss

	def on_train_epoch_end(self):
		loss = torch.stack(self.trainStepLosses).mean()
		self.log_dict({
			"Training. Mean BCE loss. Epoch.": loss.item(),
			"step" : self.current_epoch
		})
		self.trainStepLosses.clear()

	def validation_step(self, batch, batchIdx):
		[[inputTknIdsOrEmbeds, inputTknMask, tknIdx], tknLbls] = batch

		#find out the longest sequence in the batch, trim everything to match and remove unnecessary padding
		maxSeqLen = inputTknMask.to(dtype=torch.int16).sum(dim=1).max()
		inputTknIdsOrEmbeds = inputTknIdsOrEmbeds[:, :maxSeqLen]
		inputTknMask = inputTknMask[:, :maxSeqLen]

		#ensure the tensors are on the correct device
		inputTknIdsOrEmbeds = inputTknIdsOrEmbeds.to(self.device)
		inputTknMask = inputTknMask.to(self.device)
		tknIdx = tknIdx.to(self.device)
		tknLbls = tknLbls.to(self.device)

		#make predictions for this batch
		resModel = self.model(inputTknIdsOrEmbeds=inputTknIdsOrEmbeds, inputTknMask=inputTknMask) #shape=(batchLen, seqLen, sum(self.numLabels))

		#extract confidence vectors for the labels of given tokens
		resTknLblConf = resModel[range(resModel.shape[0]), tknIdx, :] #shape=(batchLen, sum(self.numLabels))

		#find the loss
		loss = tnnf.binary_cross_entropy(resTknLblConf, tknLbls)

		#compute label predictions in each head, for use in metrics
		resTknLbl = torch.zeros_like(resTknLblConf)
		headOffset = 0
		for numHeadLbls in self.model.numLabels:
			#get indices of predicted labels for the current head, shape=(batchLen, 1)
			predictedLbl = resTknLblConf[range(resModel.shape[0]), headOffset:(headOffset + numHeadLbls)].argmax(dim=1)
			#offset indices back to the final vector space
			predictedLbl += headOffset
			#assign the predicted label
			resTknLbl[range(resModel.shape[0]), predictedLbl] = 1
			#move offset past current head
			headOffset += numHeadLbls

		#update metrics
		self.valConfMat(resTknLbl, tknLbls.to(torch.int32))
		self.valF1(resTknLbl, tknLbls)

		#log and accumulate the loss
		if not self.trainer.sanity_checking:
			lossForLog = loss.to("cpu").detach()
			self.log("Validation. BCE loss. Step.", lossForLog.item(), on_step=True, on_epoch=False)
			self.valStepLosses.append(lossForLog)

	def on_validation_epoch_end(self):
		#log loss for the epoch
		if not self.trainer.sanity_checking:
			loss = torch.stack(self.valStepLosses).mean()
			self.log_dict({
				"Validation. Mean BCE loss. Epoch.": loss.item(),
				"step" : self.current_epoch
			})
		
		self.valStepLosses.clear()

		#log confusion matrix for the epoch
		if not self.trainer.sanity_checking:
			cm = self.valConfMat.compute().to('cpu').clone().detach()

			fig, _ = self.valConfMat.plot(cm)
			fig.set_size_inches(self.figSizeX, self.figSizeY)
			self.logger.experiment.add_figure("Validation. Confusion matrix. Epoch.", fig, global_step=self.current_epoch)

		#prepare to accumulate confusion matrix stats from next epoch
		self.valConfMat.reset()

		#log f1 score for the epoch
		f1 : torch.Tensor = None

		if not self.trainer.sanity_checking:
			f1 = self.valF1.compute().clone().detach()
			self.histValF1.append(f1.to('cpu').clone())
			
			fig, ax = plt.subplots(figsize=(self.figSizeX, self.figSizeY))
			for lblIdx in range(self.histValF1[0].shape[0]):
				lblVals = [it[lblIdx] for it in self.histValF1]
				plt.plot(range(len(self.histValF1)), lblVals, label=f"{lblIdx}")
			ax.set_xlabel("Step")
			ax.set_ylabel("F1 score")
			ax.legend()

			self.logger.experiment.add_figure("Validation. F1. Epoch.", fig)
		
		#prepare to accumulate f1 stats from next epoch
		self.valF1.reset()

		#log accuracy score for the epoch, log and update loss scaling
		if not self.trainer.sanity_checking:
			#log head loss scaling history for current epoch
			self.histHeadLossScaling.append(self.headLossScaling.to('cpu').clone())

			fig, ax = plt.subplots(figsize=(self.figSizeX, self.figSizeY))
			for lblIdx in range(self.histHeadLossScaling[0].shape[0]):
				lblVals = [it[lblIdx] for it in self.histHeadLossScaling]
				plt.plot(range(len(self.histHeadLossScaling)), lblVals, label=f"{lblIdx}")
			ax.set_xlabel("Step")
			ax.set_ylabel("Loss scaling")
			ax.legend()

			self.logger.experiment.add_figure("Head loss scaling. Epoch.", fig)

			#update loss scaling for next epoch
			with torch.no_grad():
				#handle degenerate case not having label sample counts to work with
				if self.numLblSamples.max() == 0:
					#no samples to reason from, no boosting of head losses
					self.headLossScaling[:] = 1.0
				#normal case
				else:
					#use relative frequency of each label to derive head loss boosting coeficient
					hlbCoef = self.numLblSamples.to(torch.float32) / self.numLblSamples.max() #relative frequency of each head label
					hlbCoef = hlbCoef.clamp(min=1/10, max=1) #clamp low frequency values to prevent range distortion
					hlbCoef = 1 / hlbCoef #transform to inverse relative frequencies
					hlbCoef = hlbCoef - hlbCoef.min() #subtract bias to bring values for most frequent label(s) to 0
					hlbCoef = hlbCoef / hlbCoef.max() #bring into range [0;1]

					#compute initial head loss scaling, without taking head performance into account
					self.headLossScaling = hlbCoef * self.maxLossBoost + 1
					
					#rescale boost factor by inverse of head performance
					invPerf = (1 - f1)
					self.headLossScaling = (self.headLossScaling - 1) * invPerf + 1

			#reset label sample counter to prepare for counting in next epoch
			self.numLblSamples[:] = 0
	
	# #XXX: this is for debugging model gradients
	# def on_before_optimizer_step(self, optimizer):
	# 	from lightning.pytorch.utilities import grad_norm
	# 	norms = grad_norm(self.model, norm_type=2)
	# 	self.log_dict(norms)

	def resetStats(self):
		"""
		Reset all accumulated statistics. This used when loading a pre-trained model for uptraining.
		"""

		#reset metric calculators
		self.valConfMat.reset()
		self.valF1.reset()

		#reset metric history storage
		self.trainStepLosses = []
		self.valStepLosses = []
		self.histValF1 = []

		#reset storage for label sample counters
		self.numLblSamples = torch.zeros_like(self.numLblSamples, requires_grad=False)

		#reset loss scaling vector, set initial scaling to 1, since we do not have label sample counts yet
		self.headLossScaling = torch.ones_like(self.headLossScaling, requires_grad=False)

		#reset scaling history storage
		self.histHeadLossScaling = []