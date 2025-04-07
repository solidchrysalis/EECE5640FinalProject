# Compiler settings
CXX = gcc
NVCC = nvcc
CUDA_HOME = /shared/EL9/explorer/cuda/12.3.0
TF_CFLAGS := $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell python3 -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')
TF_INC := $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB := $(shell python3 -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')

# Output
TARGET = sgd_op.so

# Sources
SRC_CU = sgd_op.cu.cc
SRC_CC = sgd_op.cc
OBJ_CU = $(SRC_CU:.cu.cc=.cu.o)

# Compiler flags
CXXFLAGS = -fPIC -O2 -D_GLIBCXX_USE_CXX11_ABI=1 $(TF_CFLAGS)
NVCCFLAGS = -arch=$(ARCH) -c -O2 -x cu -Xcompiler -fPIC -I$(TF_INC)

LDFLAGS = -shared -fPIC -L$(CUDA_HOME) -lcudart -L$(TF_LIB) $(TF_LFLAGS)

all: $(TARGET)

$(OBJ_CU): $(SRC_CU)
	$(NVCC) $(NVCCFLAGS) $< -o $@

$(TARGET): $(SRC_CC) $(OBJ_CU)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

clean:
	rm -f *.o *.so
