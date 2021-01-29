NvJpeg - Python
---------------------------

## Require
* nvjpeg
* cuda >= 10.2
* numpy >= 1.7
* python >= 3.6
* gcc >= 7.5

## Build
```shell
make
```

## Install
```shell
make install
```

## Usage
```python
#!/usr/bin/env python3

from pynvjpeg import NvJpeg 

# read file
fp = open("input-image.jpg", "rb")
jpegData = fp.read()
fp.close()

# decode
nj = NvJpeg()
img_np = nj.decode(jpegData)

# use opencv show numpy image data
cv2.imshow("Demo", img_np)
cv2.waitKey(0)

# encode numpy image data
jpg = nj.encode(img_np)

# write file
fp = open("output-image.jpg", "wb")
fp.write(jpg)
fp.close()
```