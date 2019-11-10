from __future__ import division
from __future__ import print_function
import numpy as np


class BeamEntry:
	"information about one single beam at specific time-step"
	def __init__(self):
		self.prTotal = 0 # blank and non-blank
		self.prNonBlank = 0 # non-blank
		self.prBlank = 0 # blank
		self.prText = 1 # LM score
		self.lmApplied = False # flag if LM was already applied to this beam
		self.labeling = () # beam-labeling


class BeamState:
	"information about the beams at specific time-step"
	def __init__(self):
		self.entries = {}

	def norm(self):
		"length-normalise LM score"
		for (k, _) in self.entries.items():
			labelingLen = len(self.entries[k].labeling)
			self.entries[k].prText = self.entries[k].prText ** (1.0 / (labelingLen if labelingLen else 1.0))

	def sort(self):
		"return beam-labelings, sorted by probability"
		beams = [v for (_, v) in self.entries.items()]
		sortedBeams = sorted(beams, reverse=True, key=lambda x: x.prTotal*x.prText)
		return [x.labeling for x in sortedBeams]


def applyLM(parentBeam, childBeam, classes, lm):
	"calculate LM score of child beam by taking score from parent beam and bigram probability of last two chars"
	if lm and not childBeam.lmApplied:
		#매칭되는 글자 하나씩 추출하는 모양
		c1 = parentBeam.labeling[-1] if parentBeam.labeling else classes.index(' ') # first char
		c2 = childBeam.labeling[-1] # second char
		lmFactor = 0.01 # influence of language model
		bigramProb = lm.getCharBigram(c1, c2) ** lmFactor # probability of seeing first and second char next to each other
		childBeam.prText = parentBeam.prText * bigramProb # probability of char sequence
		childBeam.lmApplied = True # only apply LM once per beam entry


def addBeam(beamState, labeling):
	"add beam if it does not yet exist"
	if labeling not in beamState.entries:
		beamState.entries[labeling] = BeamEntry()


def ctcBeamSearch(mat, classes, lm, beamWidth=25):
	"beam search as described by the paper of Hwang et al. and the paper of Graves et al."

	blankIdx = len(classes)-1
	maxT, maxC = mat.shape

	# initialise beam state
	last = BeamState()
	labeling = ()
	last.entries[labeling] = BeamEntry()
	last.entries[labeling].prBlank = 1
	last.entries[labeling].prTotal = 1

	# go over all time-steps
	for t in range(maxT):
		curr = BeamState()

		# get beam-labelings of best beams
		bestLabelings = last.sort()[0:beamWidth]

		# go over best beams
		for labeling in bestLabelings:

			# probability of paths ending with a non-blank
			prNonBlank = 0
			# in case of non-empty beam
			if labeling:
				# probability of paths with repeated last char at the end
				prNonBlank = last.entries[labeling].prNonBlank * mat[t, labeling[-1]]

			# probability of paths ending with a blank
			prBlank = (last.entries[labeling].prTotal) * mat[t, blankIdx]

			# add beam at current time-step if needed
			addBeam(curr, labeling)

			# fill in data
			curr.entries[labeling].labeling = labeling
			curr.entries[labeling].prNonBlank += prNonBlank
			curr.entries[labeling].prBlank += prBlank
			curr.entries[labeling].prTotal += prBlank + prNonBlank
			curr.entries[labeling].prText = last.entries[labeling].prText # beam-labeling not changed, therefore also LM score unchanged from
			curr.entries[labeling].lmApplied = True # LM already applied at previous time-step for this beam-labeling

			# extend current beam-labeling
			for c in range(maxC - 1):
				# add new char to current beam-labeling
				newLabeling = labeling + (c,)

				# if new labeling contains duplicate char at the end, only consider paths ending with a blank
				if labeling and labeling[-1] == c:
					prNonBlank = mat[t, c] * last.entries[labeling].prBlank
				else:
					prNonBlank = mat[t, c] * last.entries[labeling].prTotal

				# add beam at current time-step if needed
				addBeam(curr, newLabeling)
				
				# fill in data
				curr.entries[newLabeling].labeling = newLabeling
				curr.entries[newLabeling].prNonBlank += prNonBlank
				curr.entries[newLabeling].prTotal += prNonBlank
				
				# apply LM
				applyLM(curr.entries[labeling], curr.entries[newLabeling], classes, lm)

		# set new beam state
		last = curr

	# normalise LM scores according to beam-labeling-length
	last.norm()

	 # sort by probability
	bestLabeling = last.sort()[0] # get most probable labeling

	# map labels to chars
	res = []
	for l in bestLabeling:
		res.append(l)

	return res

import torch
import copy
def testBeamSearch(classes, mats):
	"test decoder"
	#classes = '''가나다라마'''
	#mat = np.array([[0.4, 0, 0.3,0.1,0.1,0.1], [0.3, 0, 0.4,0.1,0.1,0.1],[0.4, 0, 0.3,0.1,0.1,0.1],[0.4, 0, 0.3,0.1,0.1,0.1],[0.4, 0, 0.3,0.1,0.1,0.1]])
	#mat = np.random.rand(len(expected), len(classes))
	#print('Test beam search')
	#expected = '가나다라마'
	#문자열 대신에 사전으로 바꿔야됨 or 글자 저장된 리스트?
	classes = copy.deepcopy(classes)
	temp = classes[0]
	classes[0]=classes[len(classes)-1]
	classes[len(classes)-1] = temp
	mats = mats.contiguous().cpu().data.numpy()
	temp = np.copy(mats[:,0])
	mats[:,0] = mats[:,-1]
	mats[:-1] = temp
	all_actual = []
	for mat in mats:

		actual = ctcBeamSearch(mat, classes, None, beamWidth = 5)
		all_actual.append(actual)
	#print('Expected: "' + expected + '"')
	#print('Actual: "' + str(actual) + '"')
	#print('OK' if expected == actual else 'ERROR')
	return torch.LongTensor(all_actual)
def BeamSearchDecoder(classes, mats):
	"test decoder"
	#classes = '''가나다라마'''
	#mat = np.array([[0.4, 0, 0.3,0.1,0.1,0.1], [0.3, 0, 0.4,0.1,0.1,0.1],[0.4, 0, 0.3,0.1,0.1,0.1],[0.4, 0, 0.3,0.1,0.1,0.1],[0.4, 0, 0.3,0.1,0.1,0.1]])
	#mat = np.random.rand(len(expected), len(classes))
	#print('Test beam search')
	#expected = '가나다라마'
	#문자열 대신에 사전으로 바꿔야됨 or 글자 저장된 리스트?
	all_actual = []
	for mat in mats:
		actual = ctcBeamSearch(mat.contiguous().cpu().data.numpy(), classes, None, beamWidth = 5)
		all_actual.append(actual)
	#print('Expected: "' + expected + '"')
	#print('Actual: "' + str(actual) + '"')
	#print('OK' if expected == actual else 'ERROR')
	return torch.LongTensor(all_actual)
if __name__ == '__main__':
	testBeamSearch()
