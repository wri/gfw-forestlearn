
from gfw_forestlearn import fl_regression as fl
import pandas as pd
import numpy as np
import os
from sklearn.utils import resample
import random

num_seed = 30
random.seed(num_seed)

y_column = 'carbon_seqr_rate_Mg_ha_yr'
one_hot_feats = ['BiomesMask_b1']
xy = ['Lon_1km','Lat_1km']
predictors = ['BDRLOG_M_1km_ll_b1', 'BDTICM_M_1km_ll_b1', 'BLDFIE_M_sl_b1', 'BiomesMask_b1', 'CECSOL_M_sl_b1', 
                'CLYPPT_M_sl_b1', 'CM10_1975H_Bio20_V1_2_b1', 'CM10_1975H_Bio21_V1_2_b1', 'CM10_1975H_Bio22_V1_2_b1', 'CM10_1975H_Bio23_V1_2_b1', 
                'CM10_1975H_Bio24_V1_2_b1', 'CM10_1975H_Bio25_V1_2_b1', 'CM10_1975H_Bio26_V1_2_b1', 'CM10_1975H_Bio27_V1_2_b1', 'CM10_1975H_Bio28_V1_2_b1', 
                'CM10_1975H_Bio29_V1_2_b1', 'CM10_1975H_Bio30_V1_2_b1', 'CM10_1975H_Bio31_V1_2_b1', 'CM10_1975H_Bio32_V1_2_b1', 'CM10_1975H_Bio33_V1_2_b1',
                'CM10_1975H_Bio34_V1_2_b1', 'CM10_1975H_Bio35_V1_2_b1', 'CRFVOL_M_sl_b1', 'GMTEDAspect_b1', 'GMTEDElevation_b1', 'GMTEDHillShade_b1', 
                'GMTEDSlope_b1', 'NHx_avg_dep_1980_2009_b1', 'NOy_avg_dep_1980_2009_b1', 'OCDENS_M_sl_b1', 'OCSTHA_M_sd_b1', 'ORCDRC_M_sl_b1', 
                'PHIHOX_M_sl_b1', 'PHIKCL_M_sl_b1', 'SLTPPT_M_sl_b1', 'SNDPPT_M_sl_b1', 'WWP_M_sl_b1', 'ai_et0_b1', 'et0_yr_b1', 
                'shortwave_radiaton_1982_2015_b1', 'wc_v2_bio_30s_01_b1', 'wc_v2_bio_30s_02_b1', 'wc_v2_bio_30s_03_b1', 'wc_v2_bio_30s_04_b1', 
                'wc_v2_bio_30s_05_b1', 'wc_v2_bio_30s_06_b1', 'wc_v2_bio_30s_07_b1', 'wc_v2_bio_30s_08_b1', 'wc_v2_bio_30s_09_b1',
                'wc_v2_bio_30s_10_b1', 'wc_v2_bio_30s_11_b1', 'wc_v2_bio_30s_12_b1', 'wc_v2_bio_30s_13_b1', 'wc_v2_bio_30s_14_b1', 
                'wc_v2_bio_30s_15_b1', 'wc_v2_bio_30s_16_b1', 'wc_v2_bio_30s_17_b1', 'wc_v2_bio_30s_18_b1', 'wc_v2_bio_30s_19_b1']

in_csv = '/Users/kristine/WRI/MachineLearning/CarbonAI/2021/Testing/TrainwCovar.csv'
training_data = pd.read_csv(in_csv)


def main(run_num):

    model_out_file = 'model_{}.pkl'.format(run_num)
    
    learning = fl.ForestLearn(predictors=predictors, y_column=y_column, xy = xy, one_hot_feats = one_hot_feats)
    
    bootstrapped_samples = resample(training_data,n_samples=len(training_data), replace=True, 
                                      stratify=training_data['Stratify_column'].values, random_state=run_num)
    model = learning.setup_rf_model_scale()
    params = {'max_depth': 10, 
              'max_features': 0.4, 
              'min_samples_leaf': 15, 
              'min_samples_split': 3, 
              'n_estimators': 200}
              
    learning.fit_model_with_params(training_data, model_out_file, in_params=params, in_modelfilename=None)
    #learning.tune_param_set(training_data, params, model_out_file, cv_results_filename)
    learning = None

num_splits=1
start = 0
stop=1#100
for i in np.arange(start,stop): 
    print(i)
    main(i)

