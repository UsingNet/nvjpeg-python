#!/usr/bin/env python3

import sys
import os
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '../out'))

import nvjpeg

fp = open(os.path.join(os.path.dirname(__file__), 'test.jpg'), "rb")
img = fp.read()
fp.close()


nj = nvjpeg.NvJpeg()
img_np = nj.decode(img)
jpg = nj.encode(img_np)
fp = open(os.path.join(os.path.dirname(__file__), 'out', 'python-test.jpg'), "wb")
fp.write(jpg)
fp.close()