# covermatchersv.py  2016-08-04
import numpy as np
import cv2
import os
import time

def ct():  # current local 'H:M:S'
	ltt=time.localtime()[3:6]
	return ':'.join(map(str,ltt))


class CoverMatcher:
	def __init__(self, descriptor, coverPaths, ratio = 0.7, minMatches = 40,
		useHamming = True):
		# store the descriptor, book cover paths, ratio and minimum
		# number of matches for the homography calculation, then
		# initialize the distance metric to be used when computing
		# the distance between features
		self.descriptor = descriptor
		self.coverPaths = coverPaths
		self.ratio = ratio
		self.minMatches = minMatches
		self.distanceMethod = "BruteForce"
		# for storing descriptors for reuse
		self.savedirname=os.path.split(os.path.abspath('.'))[1]
		self.savefilebase='covermatcher-{}-'.format(self.savedirname)

		# if the Hamming distance should be used, then update the
		# distance method
		if useHamming:
			self.distanceMethod += "-Hamming"

	def svcache(self):
		# save temlates to descr cache only
		# loop over the book cover images
		for n,coverPath in enumerate(self.coverPaths):
			# load the query image, convert it to grayscale, and
			# extract keypoints and descriptors
			cover = cv2.imread(coverPath)
			gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
			t0=time.time()
			(kps, descs) = self.descriptor.describe(gray)
			t1=time.time()
			dt=t1-t0
			print('img# {:03d} descr in {:4.2f} sec'.format(n,dt))
			fid,fin=os.path.split(coverPath)
			fib=os.path.splitext(fin)[0]
			fip=self.savefilebase+fib+'.npz'
			with open(fip,'wb') as ofh:
                                np.savez(ofh, kps=kps, descs=descs)

	def search(self, queryKps, queryDescs):
		# initialize the dictionary of results
		results = {}
		# loop over the book cover images
		for n,coverPath in enumerate(self.coverPaths):
			# load the query image, convert it to grayscale, and
			# extract keypoints and descriptors
			cover = cv2.imread(coverPath)
			gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
			(kps, descs) = self.descriptor.describe(gray)
			dctup=(kps, descs)
			fid,fin=os.path.split(coverPath)
			fib=os.path.splitext(fin)[0]
			fip=self.savefilebase+fib+'.npz'
			with open(fip,'wb') as ofh:
                                np.savez(ofh, kps=kps, descs=descs)
			# determine the number of matched, inlier keypoints,
			# then update the results
			score = self.match(queryKps, queryDescs, kps, descs)
			results[coverPath] = score
			print('{:03d} {:4.2f}'.format(n,time.time()))
		# if matches were found, sort them
		print('Done in covermatchersv:search')
		if len(results) > 0:
			results = sorted([(v, k) for (k, v) in results.items() if v > 0],
				reverse = True)

		# return the results
		return results

	def match(self, kpsA, featuresA, kpsB, featuresB):
		# compute the raw matches and initialize the list of actual
		# matches
		matcher = cv2.DescriptorMatcher_create(self.distanceMethod)
		rawMatches = matcher.knnMatch(featuresB, featuresA, 2)
		matches = []

		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other
			if len(m) == 2 and m[0].distance < m[1].distance * self.ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# check to see if there are enough matches to process
		if len(matches) > self.minMatches:
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (i, _) in matches])
			ptsB = np.float32([kpsB[j] for (_, j) in matches])

			# compute the homography between the two sets of points
			# and compute the ratio of matched points
			(_, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)

			# return the ratio of the number of matched keypoints
			# to the total number of keypoints
			return float(status.sum()) / status.size

		# no matches were found
		return -1.0
