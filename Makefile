CFLAGS=-Wall -Werror -O2
ifdef DEBUG
CFLAGS=-g -Wall -Werror -O0
endif

ifndef CUDA_PATH
CUDA_PATH=/usr/local/cuda
endif

ifndef PYTHON_VERSION
PYTHON_VERSION=3.6
endif

ifndef PYTHON_BIN
PYTHON_BIN=python${PYTHON_VERSION}
endif

ifndef PYTHON_INCLUDE_PATH
PYTHON_INCLUDE_PATH=/usr/include/python${PYTHON_VERSION}
endif

ifndef PYTHON_LIB_PATH
PYTHON_LIB_PATH=$(shell ldconfig -p | grep python${PYTHON_VERSION} | head -n 1 | xargs dirname | tail -n 1)
endif

ifndef PYTHON_DYNLOAD_PATH
PYTHON_DYNLOAD_PATH=$(shell ${PYTHON_BIN} -c "import sys; print(list(filter(lambda x: 'lib-dynload' in x, sys.path))[0])")
endif

PYTHON_LIB_NAME=$(shell ${PYTHON_BIN} -c "import sys; print('nvjpeg.cpython-%d%dm' % (sys.version_info.major, sys.version_info.minor,))")-x86_64-linux-gnu.so

all: out/nvjpeg-test out/${PYTHON_LIB_NAME}
out:
	mkdir out

out/nvjpeg-test.o: out nvjpeg-python.c
	gcc -o out/nvjpeg-test.o -c nvjpeg-python.c -I${CUDA_PATH}/include -I${PYTHON_INCLUDE_PATH} -D BUILD_TEST ${CFLAGS}

out/nvjpeg-test: out/nvjpeg-test.o
	gcc -o out/nvjpeg-test out/nvjpeg-test.o -L${CUDA_PATH}/lib64 -lnvjpeg -lcudart -L${PYTHON_LIB_PATH} -lpython${PYTHON_VERSION}m ${CFLAGS}

out/nvjpeg.o: out nvjpeg-python.c
	gcc -fPIC -o out/nvjpeg.o -c nvjpeg-python.c -I${CUDA_PATH}/include -I${PYTHON_INCLUDE_PATH} ${CFLAGS}

out/${PYTHON_LIB_NAME}: out/nvjpeg.o
	gcc --shared -fPIC -o out/${PYTHON_LIB_NAME} out/nvjpeg.o -L${CUDA_PATH}/lib64 -lnvjpeg -lcudart -L${PYTHON_LIB_PATH} -lpython${PYTHON_VERSION}m ${CFLAGS}

clean:
	rm -Rf out

install: out/${PYTHON_LIB_NAME}
	cp -f out/${PYTHON_LIB_NAME} ${PYTHON_DYNLOAD_PATH}