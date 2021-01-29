CFLAGS=-Wall -Werror -O2
ifdef DEBUG
CFLAGS=-g -Wall -Werror -O0
endif

all: out/nvjpeg-test out/nvjpeg.so
out:
	mkdir out

out/nvjpeg-test.o: out nvjpeg-python.c
	gcc -o out/nvjpeg-test.o -c nvjpeg-python.c -I/usr/local/cuda/include -I/usr/include/python3.6 -D BUILD_TEST ${CFLAGS}

out/nvjpeg-test: out/nvjpeg-test.o
	gcc -o out/nvjpeg-test out/nvjpeg-test.o -L/usr/local/cuda/lib64 -lnvjpeg -lcudart -L/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu -lpython3.6 ${CFLAGS}

out/nvjpeg.o: out nvjpeg-python.c
	gcc -fPIC -o out/nvjpeg.o -c nvjpeg-python.c -I/usr/local/cuda/include -I/usr/include/python3.6 ${CFLAGS}

out/nvjpeg.so: out/nvjpeg.o
	gcc --shared -fPIC -o out/nvjpeg.so out/nvjpeg.o -L/usr/local/cuda/lib64 -lnvjpeg -lcudart -L/usr/lib/python3.6/config-3.6m-x86_64-linux-gnu -lpython3.6 ${CFLAGS}


clean:
	rm -Rf out