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

pynvjpeg:
	${PYTHON_BIN} setup.py build

clean:
	rm -Rf out build dist pynvjpeg.egg-info

release: clean pynvjpeg
	${PYTHON_BIN} setup.py sdist
	${PYTHON_BIN} -m twine upload dist/*
