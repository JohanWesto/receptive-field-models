#!/usr/bin/python
"""
" @section DESCRIPTION
" Script that loads recorded or simulated data and trains selected RF models
"""

from rf_models.rf_configurations import *
from rf_models.rf_model_tester import ModelTester
from rf_models.rf_helper import load_mat_dat_file


# STEP 1: Load data
"""
The load directory (load_dir) indicates where the data files are located,
while the file name (file_name) defines which data file you want to load.

The file name might hence be, for example, "energy_model_1.0e+05.dat" if
simulated data is loaded, or "020321.A.i02.mat" if real data from the
Yang Dan Lab dataset is used. The load_mat_dat_file function automatically
uses an appropriate routine for loading .dat or .mat files. Assuming that
these are saved using either the "example_simulate_and_store_data.py" script
or the included matlab script for the Yang Dan Lab's dataset.
"""
load_dir = "data/"
file_name = "energy_model_1.0e+05.dat"
file_path = load_dir + file_name
data = load_mat_dat_file(file_path)


# STEP 2: Create a model tester and train all selected models
"""
Various RF models are predefined in 'rf_configurations.py'
The models used in Westo and May (2017) correspond to

Traditional LN models
-------------------------------------
LinRer:         glm_linreg (original_basis)
LogReg:         glm_logreg (original_basis)
PoiReg:         glm_poireg (original_basis)
iSTAC1:         ln_istac1 (original_basis)
MID1_iSTAC1:    mid1_istac_init (original_basis)

Two-filter LN models
-------------------------------------
STC2:           ln_stc2 (original_basis)
iSTAC2:         ln_istac2 (original_basis)
MNE2:           qn_mne2_C_2 (original_basis)
MID_STC2:       mid2_stc_init (original_basis)
MID_iSTAC2:     mid2_istac_init (original_basis)
MID_MNE2:       mid2_mne_init (original_basis)

Four-filter LN models
-------------------------------------
MNE4:           qn_mne4_C_2 (original_basis)

Single-CF context models
-------------------------------------
LinRegCtx:      glm_linreg_ctx_same (original_basis)
LogRegCtx:      glm_logreg_ctx_same (original_basis)
PoiRegCtx:      glm_poireg_ctx_same (original_basis)
MID_PoiRegCtx:  mid1_ctx_same_poireg_init

Single-CF context models (space-time-intensity RF)
-------------------------------------
LinRegCtxInt:   glm_linreg_ctx_same (binary_basis)
LogRegCtxInt:   glm_logreg_ctx_same (binary_basis)
PoiRegCtxInt:   glm_poireg_ctx_same (binary_basis)

Two-CF context models (space-time-intensity RF)
-------------------------------------
LinRegCtx2Int:  glm_linreg_ctx_dim2_init (binary_basis)
LogRegCtx2Int:  glm_logreg_ctx_dim2_init (binary_basis)
PoiRegCtx2Int:  glm_poireg_ctx_dim2_init (binary_basis)

Observe that some of MID models as well as the two-CF context models requires
that a corresponding simpler model has been estiamted and saved first so that
it can be used as an initial solution. These are loaded by providing the
train_models method with load_path as follows:
models_load_dir = ""
model_tester.train_models(load_path=models_load_dir)

By default, the first 20 % of the provided data is used as a test set and the
remaining 80 % for estimating the models and for setting potential
hyper-parameters. Five separate models can also be trained by setting:
model_tester.train_models(first_fold_only=False)
This will create five different models, where different parts of the original
data is used for training/testing. The first model has the first 20 % of the
data as its test set, whereas the fifth model uses the last 20 % of the data
for testing.

Estimated models as well as the MI values can also be automatically saved as
images by providing the plot methods with a path.
model_tester.plot_models(path=save_dir)
model_tester.plot_evaluation(path=save_dir)
"""

model_params = {
    'rf_win_size': 11,  # the RF's time window size (time bins)
    'basis_fun': 'original_basis',  # scales inputs to [0, 1]
    'rf_models': [mid1,
                  glm_logreg,
                  glm_logreg_ctx_same_c,
                  ln_stc2,
                  ln_istac2,
                  qn_mne2_C_2]
}
model_tester = ModelTester(model_params)
model_tester.add_raw_xy_data(data)
model_tester.train_models(load_path='/home/johan/repos/receptive-field-models/models/energy_model/')
model_tester.print_mi_values()
model_tester.print_r_values()
model_tester.plot_models()
model_tester.plot_evaluation()
model_tester.save_models(path="models/")




