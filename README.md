# receptive-field-models

Python code for training and testing various receptive field (RF) models. These are intended to extract interpretable quantizations of neural behaviour from single cell recordings. Included model types are:
 * Traditional linear-nonlinear (LN) models
 * Multi-filter LN models
 * Low-rank quadratic-nonlinear (QN) models
 * Context models
 * Subunit convolutional models

## Getting Started

The library includes a simulator for generating data from various standard neuron models, which later can be used for estimating RF models. Separate sample scripts are provided for both stages. Hence, in order to get started run:
1. example_simulate_and_store_data.py
2. example_load_data_and_train_models.py

Real data sets can be obtained from the [Collaborative Research in Computational Neuroscience](https://crcns.org/). 
The included matlab script Yang_Dan_Lab_create_x_and_y_data.m extracts and saves data into a suitable format from one particular [data set](https://crcns.org/data-sets/vc/pvc-2/about). Once saved, the 'example_load_data_and_train_models.py' script can be used to estiamte varying RF models to measurments from real neurons in primary visual cortex.

### Prerequisites

```
numpy, v. 1.11.1 or higher
matplotlib, v 1.5.1 or higher
scipy, v 0.17.0 or higher
scikit-learn, v. 0.18 or higher 
cython, v. 0.24 or higher
```

### Installing

Part of the code is written in C/C++ and needs to be compiled. This is accomplished by running
```
./compile_cython_wrappers.sh
```
