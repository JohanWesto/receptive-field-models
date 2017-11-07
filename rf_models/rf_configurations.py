#!/usr/bin/python
"""
" @section DESCRIPTION
" Configurations for different RF models
"""


def _get_model_params(name='glm',
                      solver='linreg',
                      multilin=False,
                      rf_type='lin_rfs',
                      rf_truncated=False,
                      cf_type=None,
                      cf_mapping=None,
                      cf_alignment='edge',
                      cf_act_fun=None,
                      pos_sol=True,
                      reg_c_init=1e-0,
                      reg_type='smooth',
                      init=None,
                      n_rfs=1):

    config = {'name': name,
              'params': {'solver': solver,
                         'multilin': multilin,
                         'rf_type': rf_type,
                         'rf_truncated':rf_truncated,
                         'cf_type': cf_type,
                         'cf_mapping': cf_mapping,
                         'cf_alignment': cf_alignment,
                         'cf_act_fun': cf_act_fun,
                         'pos_sol': pos_sol,
                         'reg_c_init': reg_c_init,
                         'reg_type':reg_type,
                         'init': init,
                         'n_rfs': n_rfs
                         }
              }

    return config

# GLMs
glm_linreg = \
    _get_model_params(name='glm',
                      rf_type='max_rfs',
                      multilin=False)
glm_linreg_multilin = \
    _get_model_params(name='glm',
                      rf_type='max_rfs',
                      multilin=True)
glm_logreg = \
    _get_model_params(name='glm',
                      rf_type='max_rfs',
                      solver='logreg',
                      multilin=False)
glm_logreg_multilin = \
    _get_model_params(name='glm',
                      rf_type='max_rfs',
                      solver='logreg',
                      multilin=True)
glm_logreg2 = \
    _get_model_params(name='glm',
                      n_rfs=2,
                      rf_type='max_rfs',
                      solver='logreg',
                      multilin=False)
glm_logreg3 = \
    _get_model_params(name='glm',
                      n_rfs=3,
                      rf_type='max_rfs',
                      solver='logreg',
                      multilin=False)
glm_logreg4 = \
    _get_model_params(name='glm',
                      n_rfs=4,
                      rf_type='max_rfs',
                      solver='logreg',
                      multilin=False)
glm_poireg = \
    _get_model_params(name='glm',
                      rf_type='max_rfs',
                      solver='poireg',
                      multilin=False)
glm_poireg_multilin = \
    _get_model_params(name='glm',
                      rf_type='max_rfs',
                      solver='poireg',
                      multilin=True)

# Context models, including subunit models
glm_linreg_ctx_same = \
    _get_model_params(name='ctx',
                      cf_type='ctx',
                      multilin=False,
                      cf_mapping='same')
glm_linreg_ctx_same_ml = \
    _get_model_params(name='ctx',
                      cf_type='ctx',
                      multilin=True,
                      cf_mapping='same')
glm_linreg_ctx_nl_same_adaptive = \
    _get_model_params(name='ctx',
                      cf_type='ctx_nl',
                      solver='linreg',
                      multilin=False,
                      cf_mapping='same',
                      cf_act_fun='adaptive',
                      init = 'LinRegCtx_')
glm_linreg_subunit_same_adaptive = \
    _get_model_params(name='ctx',
                      cf_type='subunit',
                      solver='linreg',
                      multilin=False,
                      cf_mapping='same',
                      cf_act_fun='adaptive',
                      init = 'LinRegCtx_')
glm_linreg_subunit_t_same_adaptive = \
    _get_model_params(name='ctx',
                      rf_truncated=True,
                      cf_type='subunit',
                      solver='linreg',
                      multilin=False,
                      cf_mapping='same',
                      cf_act_fun='adaptive',
                      init = 'LinRegCtx_')
glm_logreg_ctx_same = \
    _get_model_params(name='ctx',
                      cf_type='ctx',
                      solver='logreg',
                      multilin=False,
                      cf_mapping='same')
glm_logreg_ctx_same_ml = \
    _get_model_params(name='ctx',
                      multilin=True,
                      cf_type='ctx',
                      solver='logreg',
                      cf_mapping='same',
                      init='LogRegCtx_')
glm_logreg_ctx_same_c = \
    _get_model_params(name='ctx',
                      cf_type='ctx',
                      solver='logreg',
                      multilin=False,
                      cf_mapping='same',
                      cf_alignment='center')
glm_logreg_ctx_nl_same_adaptive = \
    _get_model_params(name='ctx',
                      cf_type='ctx_nl',
                      solver='logreg',
                      multilin=False,
                      cf_mapping='same',
                      cf_act_fun='adaptive',
                      init = 'LogRegCtx_')
glm_logreg_ctx_nl_same_rectified = \
    _get_model_params(name='ctx',
                      cf_type='ctx_nl',
                      solver='logreg',
                      multilin=False,
                      cf_mapping='same',
                      cf_act_fun='rectified',
                      init = 'LogRegCtx_')
glm_logreg_subunit_same_adaptive = \
    _get_model_params(name='ctx',
                      cf_type='subunit',
                      solver='logreg',
                      multilin=False,
                      cf_mapping='same',
                      cf_act_fun='adaptive',
                      init = 'LogRegCtx_')
glm_logreg_subunit_t_same_adaptive = \
    _get_model_params(name='ctx',
                      rf_truncated=True,
                      cf_type='subunit',
                      solver='logreg',
                      multilin=False,
                      cf_mapping='same',
                      cf_act_fun='adaptive',
                      init = 'LogRegCtx_')
glm_poireg_ctx_same = \
    _get_model_params(name='ctx',
                      cf_type='ctx',
                      solver='poireg',
                      multilin=False,
                      cf_mapping='same',
                      init='PoiRegCtx_')
glm_poireg_ctx_same_ml = \
    _get_model_params(name='ctx',
                      cf_type='ctx',
                      solver='poireg',
                      multilin=True,
                      cf_mapping='same',
                      init='PoiRegCtx_')
glm_poireg_ctx_nl_same_adaptive = \
    _get_model_params(name='ctx',
                      cf_type='ctx_nl',
                      solver='poireg',
                      multilin=False,
                      cf_mapping='same',
                      cf_act_fun='adaptive',
                      init = 'PoiRegCtx_')
glm_poireg_subunit_same_adaptive = \
    _get_model_params(name='ctx',
                      cf_type='subunit',
                      solver='poireg',
                      multilin=False,
                      cf_mapping='same',
                      cf_act_fun='adaptive',
                      init = 'PoiRegCtx_')
glm_poireg_subunit_t_same_adaptive = \
    _get_model_params(name='ctx',
                      rf_truncated=True,
                      cf_type='subunit',
                      solver='poireg',
                      multilin=False,
                      cf_mapping='same',
                      cf_act_fun='adaptive',
                      init = 'PoiRegCtx_')
glm_poireg_ctx_temporal = \
    _get_model_params(name='ctx',
                      cf_type='ctx',
                      solver='poireg',
                      multilin=False,
                      cf_mapping='temporal')
glm_poireg_subunit_temporal_adaptive = \
    _get_model_params(name='ctx',
                      cf_type='subunit',
                      solver='poireg',
                      multilin=False,
                      cf_mapping='temporal',
                      cf_act_fun='adaptive',
                      init = 'PoiRegCtx_')

# Multi-filter LN models
ln_stc1 = _get_model_params(name='ln',
                            n_rfs=1,
                            solver='stc')
ln_stc2 = _get_model_params(name='ln',
                            n_rfs=2,
                            solver='stc')
ln_stc3 = _get_model_params(name='ln',
                            n_rfs=3,
                            solver='stc')
ln_stc4 = _get_model_params(name='ln',
                            n_rfs=4,
                            solver='stc')
ln_stc6 = _get_model_params(name='ln',
                            n_rfs=6,
                            solver='stc')
ln_stc1_w = _get_model_params(name='ln',
                              n_rfs=1,
                              solver='stc_w')
ln_stc2_w = _get_model_params(name='ln',
                              n_rfs=2,
                              solver='stc_w')
ln_stc3_w = _get_model_params(name='ln',
                              n_rfs=3,
                              solver='stc_w')
ln_stc4_w = _get_model_params(name='ln',
                              n_rfs=4,
                              solver='stc_w')

ln_istac1 = \
    _get_model_params(name='ln',
                      n_rfs=1,
                      solver='istac')
ln_istac2 = \
    _get_model_params(name='ln',
                      n_rfs=2,
                      solver='istac')
ln_istac3 = \
    _get_model_params(name='ln',
                      n_rfs=3,
                      solver='istac')
ln_istac4 = \
    _get_model_params(name='ln',
                      n_rfs=4,
                      solver='istac')
ln_istac6 = \
    _get_model_params(name='ln',
                      n_rfs=6,
                      solver='istac')

qn_mne1_C_0 = \
    _get_model_params(name='qn',
                      n_rfs=1,
                      solver='mne',
                      reg_c_init=1e-0,
                      init='MNE2_C1.0')
qn_mne2_C_0 = \
    _get_model_params(name='qn',
                      n_rfs=2,
                      solver='mne',
                      reg_c_init=1e-0,
                      init='MNE2_C1.0')
qn_mne3_C_0 = \
    _get_model_params(name='qn',
                      n_rfs=3,
                      solver='mne',
                      reg_c_init=1e-0,
                      init='MNE2_C1.0')
qn_mne4_C_0 = \
    _get_model_params(name='qn',
                      n_rfs=4,
                      solver='mne',
                      reg_c_init=1e-0,
                      init='MNE2_C1.0')
qn_mne1_C_1 = \
    _get_model_params(name='qn',
                      n_rfs=1,
                      solver='mne',
                      reg_c_init=1e-1,
                      init='MNE2_C0.1')
qn_mne2_C_1 = \
    _get_model_params(name='qn',
                      n_rfs=2,
                      solver='mne',
                      reg_c_init=1e-1,
                      init='MNE2_C0.1')
qn_mne3_C_1 = \
    _get_model_params(name='qn',
                      n_rfs=3,
                      solver='mne',
                      reg_c_init=1e-1,
                      init='MNE2_C0.1')
qn_mne4_C_1 = \
    _get_model_params(name='qn',
                      n_rfs=4,
                      solver='mne',
                      reg_c_init=1e-1,
                      init='MNE2_C0.1')
qn_mne1_C_2 = \
    _get_model_params(name='qn',
                      n_rfs=1,
                      solver='mne',
                      reg_c_init=1e-2,
                      init='MNE2_C0.01')
qn_mne2_C_2 = \
    _get_model_params(name='qn',
                      n_rfs=2,
                      solver='mne',
                      reg_c_init=1e-2,
                      init='MNE2_C0.01')
qn_mne3_C_2 = \
    _get_model_params(name='qn',
                      n_rfs=3,
                      solver='mne',
                      reg_c_init=1e-2,
                      init='MNE2_C0.01')
qn_mne4_C_2 = \
    _get_model_params(name='qn',
                      n_rfs=4,
                      solver='mne',
                      reg_c_init=1e-2,
                      init='MNE2_C0.01')
qn_mne1_C_3 = \
    _get_model_params(name='qn',
                      n_rfs=1,
                      solver='mne',
                      reg_c_init=1e-3,
                      init='MNE2_C0.001')
qn_mne2_C_3 = \
    _get_model_params(name='qn',
                      n_rfs=2,
                      solver='mne',
                      reg_c_init=1e-3,
                      init='MNE2_C0.001')
qn_mne3_C_3 = \
    _get_model_params(name='qn',
                      n_rfs=3,
                      solver='mne',
                      reg_c_init=1e-3,
                      init='MNE2_C0.001')
qn_mne4_C_3 = \
    _get_model_params(name='qn',
                      n_rfs=4,
                      solver='mne',
                      reg_c_init=1e-3,
                      init='MNE2_C0.001')
qn_mne1_C_4 = \
    _get_model_params(name='qn',
                      n_rfs=1,
                      solver='mne',
                      reg_c_init=1e-4,
                      init='MNE2_C0.0001')
qn_mne2_C_4 = \
    _get_model_params(name='qn',
                      n_rfs=2,
                      solver='mne',
                      reg_c_init=1e-4,
                      init='MNE4_C0.0001')
qn_mne3_C_4 = \
    _get_model_params(name='qn',
                      n_rfs=3,
                      solver='mne',
                      reg_c_init=1e-4,
                      init='MNE2_C0.0001')
qn_mne4_C_4 = \
    _get_model_params(name='qn',
                      n_rfs=4,
                      solver='mne',
                      reg_c_init=1e-4,
                      init='MNE2_C0.0001')
qn_mne4_C_5 = \
    _get_model_params(name='qn',
                      n_rfs=4,
                      solver='mne',
                      reg_c_init=1e-5,
                      init='MNE2_C0.0001')
qn_mne2_best = \
    _get_model_params(name='qn',
                      n_rfs=2,
                      solver='mne',
                      reg_c_init=1e-1,
                      init='MNE2')
qn_mne4_best = \
    _get_model_params(name='qn',
                      n_rfs=4,
                      solver='mne',
                      reg_c_init=1e-1,
                      init='MNE4')

# MID models (Models that maximize the single spike information directly)
mid1 = \
    _get_model_params(name='mid')
mid1_ml = \
    _get_model_params(name='mid',
                      multilin=True)
mid1_ml_lin_init = \
    _get_model_params(name='mid',
                      multilin=True,
                      init='LinReg_')
mid1_ml_poi_init = \
    _get_model_params(name='mid',
                      multilin=True,
                      init='PoiReg_')
mid1_istac_init = \
    _get_model_params(name='mid',
                      init='iSTAC1_')
mid1_lin_init = \
    _get_model_params(name='mid',
                      init='LinReg_')
mid1_log_init = \
    _get_model_params(name='mid',
                      init='LogReg_')
mid1_poi_init = \
    _get_model_params(name='mid',
                      init='PoiReg_')
mid1_pos_inv = \
    _get_model_params(name='mid',
                      rf_type='pos_inv_rfs')
mid1_ctx_same = \
    _get_model_params(name='mid',
                      cf_mapping='same')
mid1_ctx_same_linreg_init = \
    _get_model_params(name='mid',
                      cf_mapping='same',
                      init='LinRegCtx_')
mid1_ctx_same_logreg_init = \
    _get_model_params(name='mid',
                      cf_mapping='same',
                      init='LogRegCtx_')
mid1_ctx_same_poireg_init = \
    _get_model_params(name='mid',
                      cf_mapping='same',
                      init='PoiRegCtx_')
mid2 = \
    _get_model_params(name='mid',
                      init='none',
                      n_rfs=2)
mid2_stc_init = \
    _get_model_params(name='mid',
                      init='STC2_',
                      n_rfs=2)
mid2_istac_init = \
    _get_model_params(name='mid',
                      init='iSTAC2_',
                      n_rfs=2)
mid2_mne_init = \
    _get_model_params(name='mid',
                      init='MNE2',
                      n_rfs=2)
mid2_mid_init = \
    _get_model_params(name='mid',
                      init='MID2_STC2__ori',
                      n_rfs=2)