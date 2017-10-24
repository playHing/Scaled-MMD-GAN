#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 21:10:57 2017

@author: mikolajbinkowski
"""

import os, sys
import scipy.misc
import numpy as np

from utils import pp, visualize, to_json

import tensorflow as tf

saver = tf.train.Saver()#max_to_keep=2)