import sys
import os
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "lib"))

from nvjpeg import NvJpeg as _NvJpeg

class NvJpeg:
    def __init__():
        self._handle = _NvJpeg()

    def encode(self, numpy_array, quality=70):
        return self._handle.encode(numpy_array, quality)

    def decode(self, jpegData):
        return self._handle.decode(jpegData)
