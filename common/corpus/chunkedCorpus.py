from typing import List, Dict, Optional
import builtins, json
from pydantic import BaseModel, Field


class ChunkSplitter:
	"""
	A helper for taking chunks out of range of indexes.
	"""

	"""
	A mask for indices. True means available, False means taken.
	"""
	_idxMask : List[bool]

	def __init__(self, lastIdx : int):
		"""
		Constructor.
		Inputs:
			lastIdx. Last index in the range.
		"""

		self._idxMask = [True for it in range(lastIdx+1)]


	def takeOut(self, start : int, len : int):
		"""
		Take out a given chunk out of the range.
		Inputs:
			start. Start index of the chunk.
			len. Length of the chunk.
		"""

		for i in range(start, start + len, 1):
			if i >= 0 and i < builtins.len(self._idxMask):
				self._idxMask[i] = False

	def getChunks(self, label : int) -> List['Chunk']:
		"""
		Get the remaining chunks.
		Inputs:
			label. Label to assign to the chunks.
		Returns:
			A list of chunks. Might be empty if there are no chunks remaining.
		"""

		#all sequence is masked, retur empty result
		allMasked = all([it == False for it in self._idxMask])
		if allMasked:
			return []
		
		#
		chunks : List['Chunk'] = []
		
		#build chunks
		idxStart = 0

		while True:
			#skip to the start of next chunk
			while idxStart < len(self._idxMask) and self._idxMask[idxStart] == False:
				idxStart += 1

			#are we past the end of sequence? end building
			if idxStart == len(self._idxMask):
				break

			#find the first index past the end of the chunk
			idxEnd = idxStart
			while idxEnd < len(self._idxMask) and self._idxMask[idxEnd] == True:
				idxEnd += 1

			#build the chunk and add to results
			chunk = Chunk(start=idxStart, len=(idxEnd-idxStart), label=label)
			chunks.append(chunk)

			#skip past the chunk
			idxStart = idxEnd
		
		#
		return chunks


class Chunk(BaseModel):
	"""
	A labeled range of text.
	"""

	"""
	Start index of the range.
	"""
	start : int

	"""
	Length of the range.
	"""
	len : int

	"""
	Label of the range.
	"""
	label : int

	def intersects(self, other : 'Chunk') -> bool:
		"""
		Tell if this range intersects other range.
		Inputs.
			other. Range to test.
		Returns.
			True if yes, false if no.
		"""
		return self.intersects(other.start, other.len)

	def intersects(self, start : int, len : int) -> bool:
		"""
		Tell if this range intersects other range.
		Inputs
			start. Start of the other range.
			len. Length of the other range.
		Returns.
			True if yes, false if no.
		"""
		
		#build interval end points
		myStart = self.start
		myEnd = self.start + max(0, self.len - 1)

		otherStart = start
		otherEnd = start + max(0, len - 1)

		#check for intersection
		answer = (
			(myStart <= otherStart and otherStart <= myEnd) or
			(myStart <= otherEnd and otherEnd <= myEnd) or
			(otherStart <= myStart and myStart <= otherEnd) or
			(otherStart <= myEnd and myEnd <= otherEnd)
		)

		#
		return answer


class ChunkedText(BaseModel):
	"""
	A piece of text that has labels assigned to its ranges.
	"""

	"""
	Entry ID in the host corpus.
	"""
	id : int

	"""
	The text itself.
	"""
	text : str

	"""
	Labeled chunks.
	"""
	chunks : List[Chunk] = Field(default = [])


class ChunkedCorpus(BaseModel):
	"""
	Corpus that contains range labeled text entries.
	"""
	
	"""
	Label definitions. id->labelName. ID's must form a continous set with min(id) being 0.
	"""
	labelDefs : Dict[int, str] = Field(default = {})

	"""
	Label groups. Labels in each group are mutually exclusive. This assumes that label indices both globally and in groups are contigous. Corresponds to labels-per-head in multilabel-multiclass classifier.
	"""
	labelGrps : List[int] = Field(default=[])

	"""
	Range labeled text entries.
	"""
	texts : List[ChunkedText] = Field(default = [])

	def nextId(self) -> int:
		"""
		Find ID that is larger that any ID used already for texts. This is a slow function, so once initial
		ID is found, it is best to generate the remaining ones manually by increments.

		Returns. 
			ID that is larger than any ID used already for texts.
		"""
		
		if len(self.texts) == 0:
			return 0
		
		maxId = max([x.id for x in self.texts])
		return maxId + 1

	def toJson(self) -> str:
		"""
		Serialize corpus to JSON.
		Returns.
			JSON serialized corpus.
		"""

		jsonStr =  json.dumps(self.model_dump())
		return jsonStr

	def saveToJson(self, path : str):
		"""
		Serialize corpus to JSON and save to given file.
		Inputs.
			path. Path of the file to save to.
		"""

		with open(path, mode="w", encoding="utf-8") as file:
			json.dump(self.model_dump(), file, indent="\t")

	def fromJson(src : str):
		"""
		Deserialize a new instance from given JSON string.
		Inputs.
			str. JSON dtring to deserialize from.
		Returns.
			The deserialized instance.
		"""

		jsonObj = json.loads(src)
		obj = ChunkedCorpus(**jsonObj)
		return obj

	def loadFromJson(path : str) ->  "ChunkedCorpus":
		"""
		Load contents of given file as JSON string and deserialize a new instance.
		Inputs.
			path. The path to file to read from.
		Returns.
			The deserialized instance.
		"""

		with open(path, mode="r", encoding="utf-8") as file:
			jsonObj = json.load(file)
			obj = ChunkedCorpus(**jsonObj)
			return obj


class ChunkedCorpusHTMLRederer:
	"""
	Renders instances of RangedCorpus as HTML code, for inspection.
	"""

	def renderLine(self, rlt : ChunkedText, labelDefs:Optional[Dict[int, str]]) -> str:
		"""
		Renders signgle line of range labeled text as HTML.
		Inputs.
			rlt. Text to render.
			lableDefs. Label definitions.
		Returns.
			A HTML representation of given text as string.		
		"""

		import matplotlib.colors as mcolors
		css4Colors = list(mcolors.XKCD_COLORS.values())

		res = ""
		for rng in rlt.chunks:
			text = rlt.text[rng.start : rng.start + rng.len]

			lbl = rng.label
			if not labelDefs is None:
				lbl = labelDefs[rng.label]

			title=f"st={rng.start},len={rng.len},lblId={rng.label},lbl={lbl}"

			color = css4Colors[rng.label % len(css4Colors)]
			style=f"border:solid 1px;color:{color}"

			htmlRng = f"<span title=\"{title}\" style=\"{style}\">{text}</span>"

			res += htmlRng

		return res

	def renderPage(self, corpus : ChunkedCorpus) -> str:
		"""
		Render corpus as HTML page.
		Inputs.
			corpus. Corpus to render.
		Returns.
			A corresponding HTML page as string.
		"""

		linesHtml = ""
		for line in corpus.texts:
			lineHtml = self.renderLine(line, corpus.labelDefs)
			linesHtml += f"<p>id = {line.id}, line = {lineHtml}</p>"

		pageHtml = \
			f"""
			<!DOCTYPE html>
			<html>
				<head></head>
				<body>
				{linesHtml}
				</body>
			</html>
			"""
		
		return pageHtml

	def renderSideBySidePage(self, refCorpus : ChunkedCorpus, resCorpus : ChunkedCorpus) -> str:
		"""
		Render HTML page for side by side comparison of given corpora. The corpora must have a matching
		amount of texts with matching id's.
		Inputs.
			refCorpus. Reference corpus.
			resCorpus. Result corpus.
		Returns.
			A HTML page for side by side comparison of given corpora. As string.
		"""

		linesHtml = ""
		for refLine in refCorpus.texts:
			htmlRefLine = self.renderLine(refLine, refCorpus.labelDefs)

			resLine = next(iter([x for x in resCorpus.texts if x.id == refLine.id]), None)
			htmlResLine = ""
			if not resLine is None:
				htmlResLine = self.renderLine(resLine, resCorpus.labelDefs)

			linesHtml += \
				f"""
				<tr>
					<td>id = {refLine.id}, line = {htmlRefLine}</td>
					<td>id = {resLine.id}, line = {htmlResLine}</td>
				</tr>
				"""
		
		pageHtml = \
			f"""
			<!DOCTYPE html>
			<html>
				<head></head>
				<body>
					<table>
						<tbody>{linesHtml}</tbody>
					</table>
				</body>
			</html>
			"""
		
		return pageHtml
	
	def renderJavascriptJson(self, refCorpus : ChunkedCorpus, resCorpus : ChunkedCorpus) -> str:
		"""
		Render javascript code with data for given corpora.
		Inputs.
			refCorpus. Reference corpus.
			resCorpus. Result corpus.
		Returns.
			Javascript code. As string.
		"""

		#define serialization model
		class SideBySideCorpora(BaseModel):
			refCorpus : ChunkedCorpus
			resCorpus : ChunkedCorpus

		#serialize corpora to JSON
		sideBySide = SideBySideCorpora(refCorpus=refCorpus, resCorpus=resCorpus)
		jsonStr = json.dumps(json.dumps(sideBySide.model_dump()))

		#render javascript
		js = \
			f"""
			var sideBySideCorpora = JSON.parse({jsonStr})
			"""

		#
		return js

		