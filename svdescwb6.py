# USAGE
# python svdecrwb6.py
# save to disk, bottle descriptors

# import the necessary packages
import sys
import time
#next only needed for python 2
#from __future__ import print_function
from pyimagesearch.coverdescriptor import CoverDescriptor
from pyimagesearch.covermatchersv import CoverMatcher
#import argparse
import glob
import os
import csv
import cv2
#import wx

def ct():  # current local 'H:M:S'
	ltt=time.localtime()[3:6]
	return ':'.join(map(str,ltt))

bp='/home/pi/winepy/bottles/test6'
if os.path.abspath(os.curdir)!=bp:
	os.chdir(bp)
	print('current dir changed to {}'.format(bp))
else:
	print('current dir is {}'.format(os.path.abspath(os.curdir)))

bottlesfd='/home/pi/winepy/bottles/test6'

# initialize the default parameters using BRISK is being used
print('using BRISK')
useSIFT =  0  #  just manually set to NO
useHamming = (useSIFT == 0)
ratio = 0.7
minMatches = 40

# if SIFT is to be used, then update the parameters
if useSIFT:
	minMatches = 50

# initialize the cover descriptor and cover matcher
print('starting CoverDescriptor() at {}'.format(ct()))
cd = CoverDescriptor(useSIFT = useSIFT)

# watch out for case sensitive ext  jpg
bottlesfp=os.path.join(bottlesfd,'*.JPG')
imgsp=glob.glob(bottlesfp)
print('{} bottle templates'.format(len(imgsp)))

cv = CoverMatcher(cd, imgsp,
	ratio = ratio, minMatches = minMatches, useHamming = useHamming)

print('starting CoverMatcher:svcache at {}'.format(ct()))
cv.svcache()
print('ending CoverMatcher:svcache at {}'.format(ct()))
