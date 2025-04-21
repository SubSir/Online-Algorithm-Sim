#!/bin/bash

# Shell script to compile the required .so files for the python code

# Install the required packages

# pip install pybind11
# pip install zstandard
# pip install tqdm

# Run this script from the root directory of the project

cd src 
python3 setup.py build_ext --inplace 
cd scheduler
python3 setup.py build_ext --inplace 