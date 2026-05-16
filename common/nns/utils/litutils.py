from typing import List, Dict, Tuple, Optional
import glob, re, os


# erase all but 10
# get all versions
#get last version
#get checkpoint

#blend first time with resumes (in notebook)

class LitUtils:
	"""
	Helpers for traning with pytorch lightning.
	"""

	def __init__(self, basePath : str):
		"""
		Constructor.
		Inputs.
			basePath. Base path to the versioning logs. Same value as for 'default_root_dir' parameter in 'Trainer'.
		"""

		self.basePath = basePath

	def getVersions(self) -> List[int]:
		"""
		Get all available versions.
		Returns.
			A list of version numbers. Sorted ascending.
		"""

		gp = os.path.join(self.basePath, "lightning_logs/version_*/")
		versionPaths = glob.glob(gp)
		versions = [int(re.findall(r"(\d+)/$", x)[0]) for x in versionPaths]
		versions.sort()

		return versions
	
	def getVersionsWithCheckpoints(self) -> List[int]:
		"""
		Get all available versions that have checkpoint files.
		Returns.
			A list of version numbers. Sorted ascending.
		"""

		allVersions = self.getVersions()
		versionsWCpoints = [ver for ver in allVersions if self.getVersionCheckpointPath(ver) is not None]

		return versionsWCpoints
	
	def getLastVersion(self) -> Optional[int]:
		"""
		Get last version number.
		Returns
			Last version number of None if there are no versions.
		"""

		vrs = self.getVersions();
		if len(vrs) == 0:
			return None
		return max(vrs)
	
	def getVersionPath(self, version : int) -> str:
		"""
		Get path for the foolder of given version.
		Inputs.
			version. Version number.
		Returns.
			A corresponding path for version folder.
		"""

		return  os.path.join(self.basePath, f"lightning_logs/version_{version}/")
	
	def getVersionCheckpointPath(self, version: int) -> Optional[str]:
		"""
		Get path to the version checkpoint file.
		Inputs.
			version. Version number.
		Returns.
			A path to the checkpoint file of the version or None if no such checkpoint file exists.
		"""

		cptPaths = glob.glob(
			os.path.join(self.basePath, f"lightning_logs/version_{version}/checkpoints/*.ckpt")
		)
		if len(cptPaths) == 0 or (not os.path.isfile(cptPaths[0])):
			return None
		return cptPaths[0]

	def getLastEpoch(self, version: int, assumeEpochsPerVersion : int | None = None) -> int:
		"""
		Get epoch of the checpoint. If any of the intermediate checkpoints are not	available and assumeEpochsPerVesion is not set, will raise AssertionError. If checkpoint of given version is not available, will assume it epoch is 0.
		Inputs.
			version. Version number.
			assumeEpochsPerCheckpoint. How may epochs to assume per version when checkpoint is not available.
		Returns.
			Last epoch of the given version. 0 based.
		"""

		lastEpoch = 0

		#sum up epochs of intermediate versions, if any
		if version > 0:
			for intmdVer in range(version):
				verCptPath = self.getVersionCheckpointPath(intmdVer)
				
				#no checkpoint file for version? try using assumed epochs number
				if verCptPath is None:
					if assumeEpochsPerVersion is None:
						raise AssertionError(f"Version {intmdVer} has no checkpoint file and value of argument 'assumeEpochsPerVersion' is None.")
					lastEpoch += assumeEpochsPerVersion
				#parse last epoch from checkpoint file name
				else:
					cptFName = os.path.basename(verCptPath)
					cptEpoch = int(re.match(r"epoch=(\d+)-step=(\d+)\.ckpt", cptFName).group(1))
					lastEpoch += cptEpoch + 1

		#add epochs of given version, if available, to total epochs
		cptPath = self.getVersionCheckpointPath(version)
		if not(cptPath is None):
			cptFName = os.path.basename(cptPath)
			cptEpoch = int(re.match(r"epoch=(\d+)-step=(\d+)\.ckpt", cptFName).group(1))
			lastEpoch += cptEpoch
		
		#
		return lastEpoch
		
	def eraseAllButNLastCheckpoints(self, n : int):
		"""
		Erase checkpoint files for all, but N last versions.
		Inputs.
			n. How many of the latest versions to spare.
		"""

		vrs = self.getVersions()
		vrs.sort()

		if n >= len(vrs):
			return
		
		vrsToErase = vrs[0:-n]
		for vr in vrsToErase:
			cptFilePath = self.getVersionCheckpointPath(vr)
			if cptFilePath is not None:
				os.remove(cptFilePath)
