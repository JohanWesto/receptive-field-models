#!/bin/bash

# Compile the modified Libilienar library and wrapper 
cd liblinear/
python setup.py build_ext --inplace
cd ../

# Compile the custon cross-correlation function
cd rf_models/cython/
python setup.py build_ext --inplace
cd ../
