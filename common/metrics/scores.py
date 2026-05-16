from typing import List, Tuple
import polars as pl
import numpy as np

import sklearn.metrics as sklmet


def extractLabelPreds(mlPreds : List[List[float]], lblIdx : int) -> List[float]:
	"""
	Given a list of multiclass predictions, extract predictions for a single class. Extracted predictions will maintain original
	order.
	Inputs.
		mlPreds. Multiclass predictions.
	Returns.
		A list of predictions for a chosen class.
	"""

	if len(mlPreds) == 0:
		return []
	
	result = np.zeros(len(mlPreds))
	for mlPredIdx, mlPred in enumerate(mlPreds):
		result[mlPredIdx] = mlPred[lblIdx]

	return result


def getBinaryClassBalance(expClss : List[float]) -> Tuple[float, float]:
	"""
	Get class balance from the expeceted prediction values.
	Inputs.
		expClss A list of expected classes. 0 negative, 1 positive.
	Returns.
		A tuple of normalized samples for (negative, positve). This will sum to 1, thus each part represents a portion of a class in samples in a set.
	"""

	expClss : np.ndarray = np.array(expClss, copy=False)
	
	num0s = np.where(expClss == 0, 1, 0).sum()
	num1s = np.where(expClss == 1, 1, 0).sum()
	
	total = num0s + num1s

	if total == 0:
		return (0.5, 0.5)
	
	return (num0s/total, num1s/total)


def f1scoreBaseline(expected : List[float]) -> float:
	"""
	Get macro averaged F1 score baseline.
	Inputs.
		expected A list of expected predictions. 0 negative, 1 positive.
	Returns.
		F1 score baseline.
	"""

	num0s, num1s = getBinaryClassBalance(expected)

	dominantCls = 1 if num1s > num0s else 0
	dominantExp = np.repeat(dominantCls, len(expected))
	baselineScore = sklmet.f1_score(expected, dominantExp, average='macro', zero_division=0)

	return baselineScore


def f1score(expected : List[float], predicted : List[float]) -> float:
	"""
	Get macro averaged F1 score.
	Inputs.
		expected A list of expected predictions. 0 negative, 1 positive.
		predicted A list of actual predictions. 0 negative, 1 positive.
	Returns.
		F1 score, macro averaged.
	"""
	
	score = sklmet.f1_score(expected, predicted, average='macro', zero_division=0)
	return score
