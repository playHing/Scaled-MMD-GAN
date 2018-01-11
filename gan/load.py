from __future__ import division, print_function
from glob import glob
import os

import numpy as np
import scipy.misc
from six.moves import xrange
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
import lmdb
from IPython.display import display

import mmd as MMD
from ops import batch_norm, conv2d, deconv2d, linear, lrelu
from utils import unpickle

