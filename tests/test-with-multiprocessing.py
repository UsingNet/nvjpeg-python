#!/usr/bin/env python3

import sys
import os
import numpy as np
import cv2
import glob
from concurrent.futures import ThreadPoolExecutor, wait, ALL_COMPLETED, FIRST_COMPLETED

for lib in glob.glob(os.path.join(os.path.dirname(__file__), "../build/lib.*")):
    sys.path.append(lib)

from nvjpeg import NvJpeg
nj = NvJpeg()


image_dir = os.path.join(os.path.dirname(__file__), "test-image")
image_dir_out = os.path.join(os.path.dirname(__file__), "out")

def test_process(id):
    nj = NvJpeg()
    # print("Process %d" %(id,))
    fp = open(os.path.join(image_dir, "%d.jpg" % (id,)), "rb")
    img = fp.read()
    fp.close()
    nj_np = nj.decode(img)
    print("Decode Image %d Size:" % (id,), nj_np.shape)
    nj_jpg = nj.encode(nj_np)
    print('Jpeg %d Size: %d' % (id, len(nj_jpg)))
    fp = open(os.path.join(image_dir_out, "nv-mp-test-%d.jpg" % (id,)), "wb")
    fp.write(nj_jpg)
    fp.close()

 
executor = ThreadPoolExecutor(max_workers=10)
task_ids = range(10)
print("submit test")
all_task = [executor.submit(test_process, (id)) for id in task_ids]
wait(all_task, return_when=ALL_COMPLETED)
print("test-finished")