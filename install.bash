#!/bin/bash
cd impls/Stochastic/dist
pip install --force-reinstall stochastic_cuda-0.0.0-cp39-cp39-linux_x86_64.whl
cd ../../ADAM/dist
pip install --force-reinstall adam_cuda-0.0.0-cp39-cp39-linux_x86_64.whl
cd ../../Adagrad/dist
pip install --force-reinstall adagrad_cuda-0.0.0-cp39-cp39-linux_x86_64.whl
cd ../../../