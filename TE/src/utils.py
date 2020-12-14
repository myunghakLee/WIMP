#!/usr/bin/env python

from   mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import numpy as np
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

category_to_idx = {
  u'car':        1,
  u'pedestrian': 2,
  u'truck':      3,
  u'vehicle':    4, #other vehicle
  u'van':        5,
  u'person':     6,
  u'cyclist':    7,
  u'tram':       8,
  u'misc':       9,
  u'bus': 		 10,
  u'coche':      11,
  u'motorcyclist': 12,
  u'animals':    13
}

idx_to_category = {
  1: u'car',
  2: u'pedestrian',
  3: u'truck',
  4: u'vehicle',
  5: u'van',
  6: u'person',
  7: u'cyclist',
  8: u'tram',
  9: u'misc',
  10: u'bus',
  11: u'coche',
  12: u'motorcyclist',
  13: u'animals'
}

type_to_idx = {
  u'static':  1,
  u'dynamic': 2
}

idx_to_type = {
  1: u'static',
  2: u'dynamic'
}

class bcolors:
    HEADER  = '\033[95m'
    BLUE    = '\033[94m'
    GREEN   = '\033[92m'
    WARNING = '\033[93m'
    FAIL    = '\033[91m'
    ENDC    = '\033[0m'
    BOLD    = '\033[1m'
    HIGHL   = '\x1b[6;30;42m'
    UNDERLINE = '\033[4m'

class dict2(dict):
    def __init__(self, **kwargs):
        dict.__init__(self, kwargs)
        self.__dict__ = self

def get_center_pos(bb, ax, edgecolor='red'):	
	m = [bb[0],bb[1],bb[0]+bb[3],bb[1],bb[0]+bb[3],bb[1]+bb[4],bb[0],bb[1]+bb[4]]
	m = [bb[0],bb[1],bb[0]+bb[3],bb[1],bb[0]+bb[3],bb[1]+bb[4],bb[0],bb[1]+bb[4]]

	m = np.asarray(m)
	m = m.reshape(4,2)

	m[0,0] -= bb[3]/2
	m[1,0] -= bb[3]/2
	m[2,0] -= bb[3]/2
	m[3,0] -= bb[3]/2

	m[0,1] -= bb[4]/2
	m[1,1] -= bb[4]/2
	m[2,1] -= bb[4]/2
	m[3,1] -= bb[4]/2

	t = mpl.transforms.Affine2D().rotate_deg_around((m[0,0]+m[1,0])/2,(m[0,1]+m[3,1])/2,np.rad2deg(bb[6]))+ax.transData

	rect = patches.Polygon([m[0,0:2], m[1,0:2], m[2,0:2], m[3,0:2]], fill=False, edgecolor=edgecolor)
	rect.set_transform(t)

	#print(rect.get_transform())

	return rect
	
	
	