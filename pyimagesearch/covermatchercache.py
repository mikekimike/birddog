# import the necessary packages
import numpy as np
import cv2
import os
import glob
import time

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
		self.savedir=os.path.abspath('.')
		self.savedirname=os.path.split(self.savedir)[1]
		self.savefilebase='covermatcher-{}-'.format(self.savedirname)
		print('savedir={}'.format(self.savedir))
		print('savedirname={}'.format(self.savedirname))
		print('savefilebase={}'.format(self.savefilebase))
		# if the Hamming distance should be used, then update the
		# distance method
		if useHamming:
			self.distanceMethod += "-Hamming"

	def search(self, queryKps, queryDescs):
		# initialize the dictionary of results
		results = {}
		# load the template descriptors into td dictionary
		td={}
		tfd=self.savedir
		tfpat='covermatcher-*.npz'
		tpat=os.path.join(tfd,tfpat)
		print('tpat={}'.format(tpat))
		tpathlist=glob.glob(tpat)
		print('{} cached templates found in {}'.format(len(tpathlist),tfd))
		for n,t in enumerate(tpathlist):
                        tbp=os.path.split(t)[1]
                        tbn=os.path.splitext(tbp)[0]
                        tbi=tbn.split('-')[-1]
                        if n==0: print('loaded key[0]={}'.format(tbi))
                        npzf=np.load(t)
                        k=npzf['kps']
                        d=npzf['descs']
                        if n==0:
                                print('loaded k[0].shape={}'.format(k.shape))
                                print('loaded d[0].shape={}'.format(d.shape))
                                print('loaded k[0][0].shape={}'.format(k[0].shape))
                                print('loaded d[0][0].shape={}'.format(d[0].shape))
                        td[tbi]=(k,d)
		# loop over the book cover images
		firstcalc=True
		firstchached=True
		for n,coverPath in enumerate(self.coverPaths):
                        bp=os.path.split(coverPath)[1]
                        bn=os.path.splitext(bp)[0]
                        #print('bp={}'.format(bp))
                        #print('bn={}'.format(bn))
                        if bn not in td:
                                # calc & save
                                if firstcalc:
                                        # load the query image, convert it to grayscale, and
                                        # extract keypoints and descriptors
                                        cover = cv2.imread(coverPath)
                                        gray = cv2.cvtColor(cover, cv2.COLOR_BGR2GRAY)
                                        (kps, descs) = self.descriptor.describe(gray)
                                        print('first calc kps.shape={}'.format(kps.shape))
                                        print('first calc descs.shape={}'.format(descs.shape))
                                        print('first calc kps[0].shape={}'.format(kps[0].shape))
                                        print('first calc descs[0].shape={}'.format(descs[0].shape))
                                        firstcalc=False
                        else:
                                # get the cached descriptor
                                (kps, descs) = td[bn]
                                if firstchached:
                                        print('first cached kps.shape={}'.format(kps.shape))
                                        print('first cached descs.shape={}'.format(descs.shape))
                                        print('first cached kps[0].shape={}'.format(kps[0].shape))
                                        print('first cached descs[0].shape={}'.format(descs[0].shape))
                                        firstchached=False
			# determine the number of matched, inlier keypoints,
			# then update the results
                        t0=time.time()
                        score = self.match(queryKps, queryDescs, kps, descs)
                        t1=time.time()
                        print('try matching bp={}, score={:6.3f} in {:6.3f} sec'.format(bp,score,(t1-t0)),flush=False)
                        results[coverPath] = score
		# if matches were found, sort them
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
			if status.any()==None:
                                return -1.0
                        else:
                                return float(status.sum()) / status.size

		# no matches were found
		return -1.0
