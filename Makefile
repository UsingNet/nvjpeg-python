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

ifndef PYTHON_INCLUDE_PATH
PYTHON_INCLUDE_PATH=/usr/include/python${PYTHON_VERSION}
endif

ifndef PYTHON_LIB_PATH
PYTHON_LIB_PATH=$(shell ldconfig -p | grep python${PYTHON_VERSION} | head -n 1 | xargs dirname | tail -n 1)
endif

all: out/nvjpeg-test out/nvjpeg.so
out:
	mkdir out

out/nvjpeg-test.o: out nvjpeg-python.c
	gcc -o out/nvjpeg-test.o -c nvjpeg-python.c -I${CUDA_PATH}/include -I/usr/include/python3.6 -D BUILD_TEST ${CFLAGS}

out/nvjpeg-test: out/nvjpeg-test.o
	gcc -o out/nvjpeg-test out/nvjpeg-test.o -L${CUDA_PATH}/lib64 -lnvjpeg -lcudart -L${PYTHON_LIB_PATH} -lpython${PYTHON_VERSION}m ${CFLAGS}

out/nvjpeg.o: out nvjpeg-python.c
	gcc -fPIC -o out/nvjpeg.o -c nvjpeg-python.c -I${CUDA_PATH}/include -I${PYTHON_INCLUDE_PATH} ${CFLAGS}

out/nvjpeg.so: out/nvjpeg.o
	gcc --shared -fPIC -o out/nvjpeg.so out/nvjpeg.o -L${CUDA_PATH}/lib64 -lnvjpeg -lcudart -L${PYTHON_LIB_PATH} -lpython${PYTHON_VERSION}m ${CFLAGS}


clean:
	rm -Rf out