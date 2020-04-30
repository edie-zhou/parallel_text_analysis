CC=g++
NVCC=nvcc
CFLAGS=-O3 -m64 -arch compute_30


all: match.exe

.PHONY: clean

clean: 
	rm -rf match.exe match.exe_debug *.o

match.exe: main.cu input.cpp
	$(NVCC) $(CFLAGS) main.cu input.cpp -o match.exe

