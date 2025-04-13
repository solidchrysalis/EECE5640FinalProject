EXT_NAME = stochastic_cuda

.PHONY: all clean

all:
	python3 build_wrapper.py $(ARCH)

clean:
	rm -rf build *.so dist/$(EXT_NAME).whl __pycache__