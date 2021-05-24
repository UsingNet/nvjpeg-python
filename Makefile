ifndef PYTHON_VERSION
PYTHON_VERSION=$(shell python3 -c "import sys; print('%d.%d' % (sys.version_info.major, sys.version_info.minor,))")
endif

ifndef PYTHON_BIN
PYTHON_BIN=python${PYTHON_VERSION}
endif

all: pynvjpeg
out:
	mkdir out

test:
	${PYTHON_BIN} tests/test.py
	${PYTHON_BIN} tests/test-with-multiprocessing.py

pynvjpeg: build/lib/libcolor_space.a
	${PYTHON_BIN} setup.py build

lib_cuda: build/lib/libcolor_space.a

build/lib/libcolor_space.a: src/jetson/Utils/ColorSpace.cu
	mkdir -p build/lib
	nvcc -DCUDNN  --compiler-options "-fPIC -lstdc++ -pthread -lm" -c src/jetson/Utils/ColorSpace.cu -o build/lib/libcolor_space.a

clean:
	rm -Rf out build dist pynvjpeg.egg-info

release: clean pynvjpeg
	${PYTHON_BIN} setup.py sdist
	${PYTHON_BIN} -m twine upload dist/*
