# Makefile for compiling adam.cu with selectable architecture

# Source and output
SRC = adam.cu
OUT = libadam.so

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
