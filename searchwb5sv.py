# USAGE
# python searchwb5.py
# use disk-cached bottle descriptors,  use eighth-size pics

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
'''
def get_path(fp,wildcard):
    app = wx.App(None)
    style = wx.FD_OPEN | wx.FD_FILE_MUST_EXIST
    dialog = wx.FileDialog(None, 'Open', fp, wildcard=wildcard, style=style)
    if dialog.ShowModal() == wx.ID_OK:
        path = dialog.GetPath()
    else:
        path = None
    dialog.Destroy()
    return path
'''

bp='/home/pi/winepy/bottles/eighth_res'
if os.path.abspath(os.curdir)!=bp:
	os.chdir(bp)
	print('current dir changed to {}'.format(bp))
else:
	print('current dir is {}'.format(os.path.abspath(os.curdir)))

#winesdb=None  # no wine db yet
bottlesfd='/home/pi/winepy/bottles/eighth_res'
queryfd='/home/pi/winepy/queries'  # 
query1='IMG_8446eighth.JPG'  #  case sens
print('going to search for match to {}'.format(query1))


'''
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--db", required = True,
	help = "path to the book database")
ap.add_argument("-c", "--covers", required = True,
	help = "path to the directory that contains our book covers")
ap.add_argument("-q", "--query", required = True,
	help = "path to the query book cover")
ap.add_argument("-s", "--sift", type = int, default = 0,
	help = "whether or not SIFT should be used")
args = vars(ap.parse_args())


# initialize the database dictionary of covers
db = {}

# loop over the database
csvrdr=csv.reader(open(args["db"]))
print('{} csv lines read'.format(len(csvlines)))
for l in  csvrdr:
	# update the database using the image ID as the key
	db[l[0]] = l[1:]
'''

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
print('starting CoverMatcher() at {}'.format(ct()))

# watch out for case sensitive ext  jpg
bottlesfp=os.path.join(bottlesfd,'*.JPG')
imgsp=glob.glob(bottlesfp)
print('{} bottle templates'.format(len(imgsp)))

cv = CoverMatcher(cd, imgsp,
	ratio = ratio, minMatches = minMatches, useHamming = useHamming)

print('starting query match at {}'.format(ct()))
# load the query image, convert it to grayscale, and extract
# keypoints and descriptors

#queryfp=get_path(queryfd,'*.jpg')
queryfp=os.path.join(queryfd,query1)  #  query1=img8250.jpg

queryImage = cv2.imread(queryfp)
gray = cv2.cvtColor(queryImage, cv2.COLOR_BGR2GRAY)
(queryKps, queryDescs) = cd.describe(gray)
print('finish query match at {}'.format(ct()))

print('starting search')
# try to match the book cover to a known database of images
results = cv.search(queryKps, queryDescs)
print('finish search at {}'.format(ct()))

# show the query cover
cv2.imshow("Query", queryImage)
cv2.waitKey(0)


# check to see if no results were found
nr=len(results)
if nr == 0:
	print("I could not find a match for that cover!")
	cv2.waitKey(0)
# otherwise, matches were found
else:
	print('search produced {} results'.format(nr))
	print('the top four:')
	# loop over the top 4 results
	for (i, (score, coverPath)) in enumerate(results[0:4]):
		# grab the book information
		resultnum=i+1
		print("{}. {:.2f}% : {}".format(resultnum, score * 100,os.path.basename(coverPath)))

		# load the result image and show it
		result = cv2.imread(coverPath)
		cv2.imshow("Result#%d"%(resultnum,), result)
		cv2.waitKey(0)
