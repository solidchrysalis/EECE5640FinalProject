EXT_NAME = adagrad_cuda

.PHONY: all clean

all:
	python setup.py build_ext --inplace

clean:
	rm -rf build *.so $(EXT_NAME).egg-info __pycache__