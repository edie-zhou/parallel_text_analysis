CC=g++
NVCC=nvcc
CFLAGS=-O3 -m64 -arch compute_30


all: nfa 

.PHONY: clean

clean: 
	rm -rf nfa nfa_debug *.o

nfa: main.cu input.cpp
	$(NVCC) $(CFLAGS) main.cu input.cpp -o nfa

