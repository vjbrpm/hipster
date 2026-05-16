from typing import Any, List, Tuple, Callable
import numpy as np
import torch

#define embedder type
Embedder = Callable[[np.ndarray[tuple[int], np.int32], np.ndarray[tuple[int], np.int8]], np.ndarray[tuple[int, int], np.float32]]

#
def buildPotionEmbedderV2(tdev : torch.device) -> Embedder:
	"""
	Builds embedder for potion based models. This uses no_grad context, but results are still suitable for passing to caluclations that need gradient acuumulation.
	:param: tdev Device to run the embedder on.
	:return: Embedder instance.
	"""

	#create a dummy instance of the model, extract base embeddings
	from .potion_multiclass_multilabel_v2 import MultiClassMultiLabelTokenLabeler
	model = MultiClassMultiLabelTokenLabeler([1], 1, 1)
	embedding = model.embedding
	del model
	embedding.to(tdev)

	#create embedder
	def embedder(tknIds : np.ndarray[tuple[int], np.int32], tknMask :np.ndarray[tuple[int], np.int8]) -> np.ndarray[tuple[int, int], np.float32]:
		with torch.no_grad():
			tknIds = torch.tensor(tknIds, device=tdev, dtype=torch.int32).unsqueeze(dim=0)
			embeds = embedding(tknIds)
			embeds = embeds.squeeze().to("cpu").numpy()
			return embeds
		
	#
	return embedder

def buildPotionEmbedderV2ForInference(tdev : torch.device) -> Embedder:
	"""
	Builds embedder for potion based models. This uses inference mode context.
	:param: tdev Device to run the embedder on.
	:return: Embedder instance.
	"""

	#create a dummy instance of the model, extract base embeddings
	from .potion_multiclass_multilabel_v2 import MultiClassMultiLabelTokenLabeler
	model = MultiClassMultiLabelTokenLabeler([1], 1, 1)
	embedding = model.embedding
	del model
	embedding.to(tdev)

	#create embedder
	def embedder(tknIds : np.ndarray[tuple[int], np.int32], tknMask :np.ndarray[tuple[int], np.int8]) -> np.ndarray[tuple[int, int], np.float32]:
		with torch.inference_mode():
			tknIds = torch.tensor(tknIds, device=tdev, dtype=torch.int32).unsqueeze(dim=0)
			embeds = embedding(tknIds)
			embeds = embeds.squeeze().to("cpu").numpy()
			return embeds
		
	#
	return embedder

def buildMmbertEmbedderV2(tdev : torch.device) -> Embedder:
	"""
	Builds embedder for mmBert based models.
	:param: tdev Device to run the embedder on.
	:return: Embedder instance.
	"""

	#create a dummy instance of the model, extract base model
	from nns.mmbert_multiclass_multilabel_v2 import MultiClassMultiLabelTokenLabeler
	model = MultiClassMultiLabelTokenLabeler([1], 1, 1)
	baseModel = model.baseModel
	del model
	baseModel.to(tdev)

	#create embedder
	def embedder(tknIds : np.ndarray[tuple[int], np.int32], tknMask :np.ndarray[tuple[int], np.int8]) -> np.ndarray[tuple[int, int], np.float32]:
		torch.compiler.set_stance("force_eager")
		try:
			with torch.no_grad():
				tknIds = torch.tensor(tknIds, device=tdev, dtype=torch.int32).unsqueeze(dim=0)
				tknMask = torch.tensor(tknMask, device=tdev, dtype=torch.int8).unsqueeze(dim=0)

				embeds = baseModel(input_ids=tknIds, attention_mask=tknMask).last_hidden_state
				embeds = embeds.squeeze().to("cpu").numpy()

				return embeds
		finally:
			torch.compiler.set_stance("default")
	#
	return embedder
