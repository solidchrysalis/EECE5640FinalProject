EXT_NAME = stochastic_cuda

.PHONY: all clean

all:
	python setup.py bdist_wheel

clean:
	rm -rf build *.so dist/$(EXT_NAME).whl __pycache__