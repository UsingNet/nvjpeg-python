NvJpeg - Python
---------------------------

## Require
* nvjpeg
* cuda >= 10.2
* numpy >= 1.7
* python >= 3.6
* gcc >= 7.5
* make >= 4.1

## System
Linux(x86, x86_64)
Nvidia Jetson OS


## Install
```shell
pip install pynvjpeg
```

## Usage

### 0. Init PyNvJpeg
```python
from nvjpeg import NvJpeg
nj = NvJpeg()
```

### 1. Use PyNvJpeg

#### Read Jpeg File to Numpy
```python
img = nj.read("_JPEG_FILE_PATH_")
# like cv2.imread("_JPEG_FILE_PATH_")
```

#### Write Numpy to Jpeg File
```python
nj.write("_JPEG_FILE_PATH_", img)
# or nj.write("_JPEG_FILE_PATH_", quality)
# int quality default 70, mean jpeg quality
# like cv2.imwrite("_JPEG_FILE_PATH_", img)
```

#### Decode Jpeg bytes in variable
```python
img = nj.decode(jpeg_bytes)
# like cv2.imdecode(variable)
```

#### Encode image numpy array to bytes
```python
jpeg_bytes = nj.encode(img)
# or with jpeg quality
# jpeg_bytes = nj.encode(img, 70)
# int quality default 70, mean jpeg quality

# like cv2.imencode(".jpg", variable)[1]
```
