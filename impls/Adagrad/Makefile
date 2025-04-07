# Makefile for compiling adagrad.cu with selectable architecture

# Source and output
SRC = adagrad.cu
OUT = libadagrad.so

# Default architecture (can be overridden on command line)
ARCH ?= sm_70  # Default to V100

# Compiler and flags
NVCC = nvcc
CFLAGS = -Xcompiler -fPIC -shared

all: $(OUT)

$(OUT): $(SRC)
	$(NVCC) $(CFLAGS) -gencode arch=compute_$(ARCH:sm_%=%),code=$(ARCH) $(SRC) -o $(OUT)

clean:
	rm -f $(OUT)
