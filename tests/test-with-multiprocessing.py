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

success_times = 0

def test_process_global(id):
    global nj,success_times
    for run_time in range(id+1):
        # print("Process %d run time %d" %(id, run_time))
        fp = open(os.path.join(image_dir, "%d.jpg" % (id%10,)), "rb")
        img = fp.read()
        fp.close()
        try:
            nj_np = nj.decode(img)
        except e:
            print(e)
        # print("Decode Image %d Size:" % (id,), nj_np.shape)
        nj_jpg = nj.encode(nj_np)
        # print('Jpeg %d Size: %d' % (id, len(nj_jpg)))
        fp = open(os.path.join(image_dir_out, "nv-mp-test-%d.jpg" % (id,)), "wb")
        fp.write(nj_jpg)
        fp.close()
    success_times+=1
    # print("\tProcess %d test finished" % (id,))


def test_process_in_threads(id):
    global success_times
    nj = NvJpeg()
    for run_time in range(id+1):
        # print("Process %d run time %d" %(id, run_time))
        fp = open(os.path.join(image_dir, "%d.jpg" % (id%10,)), "rb")
        img = fp.read()
        fp.close()
        nj_np = nj.decode(img)
        # print("Decode Image %d Size:" % (id,), nj_np.shape)
        nj_jpg = nj.encode(nj_np)
        # print('Jpeg %d Size: %d' % (id, len(nj_jpg)))
        fp = open(os.path.join(image_dir_out, "nv-mp-test-%d.jpg" % (id,)), "wb")
        fp.write(nj_jpg)
        fp.close()
    del nj
    success_times+=1
    # print("\tProcess %d test finished" % (id,))

TEST_TIMES = 50

executor = ThreadPoolExecutor(max_workers=10)
task_ids = range(TEST_TIMES)

print("submit global test")
success_times = 0
all_task = [executor.submit(test_process_global, (id)) for id in task_ids]
wait(all_task, return_when=ALL_COMPLETED)
if success_times == TEST_TIMES:
    print("global test finished")
else:
    print("global test with error %d" % (TEST_TIMES-success_times))

print("submit in threads test")
success_times = 0
all_task = [executor.submit(test_process_in_threads, (id)) for id in task_ids]
wait(all_task, return_when=ALL_COMPLETED)
if success_times == TEST_TIMES:
    print("in threads test finished")
else:
    print("in threads test with error %d" % (TEST_TIMES-success_times))
