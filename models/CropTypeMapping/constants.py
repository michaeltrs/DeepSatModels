import numpy as np
import torch.nn as nn
import torch
import os

"""
Constants for file paths
"""

SPLITS = ['train', 'val', 'test']
NON_DL_MODELS = ['logreg', 'random_forest']
DL_MODELS = ['bidir_clstm','fcn', 'unet', 'fcn_crnn', 'mi_clstm', 'unet3d', 'only_clstm_mi']
MULTI_RES_MODELS = ['fcn_crnn']

S1_NUM_BANDS = 3
PLANET_NUM_BANDS = 4

LABEL_DIR = "raster_npy"
S1_DIR = "s1_npy"
S2_DIR = "s2_npy"
NROW = 5

# FILE PATHS: 
BASE_DIR = os.getenv("HOME")

GCP_DATA_DIR = BASE_DIR + '/croptype_data/data'
LOCAL_DATA_DIR = BASE_DIR + '/croptype_data_local/data'
#LOCAL_DATA_DIR = 'data'

HDF5_PATH = { 'ghana': LOCAL_DATA_DIR + '/ghana/final_data.hdf5_32',
              'southsudan': LOCAL_DATA_DIR + '/southsudan/final_data.hdf5_32',
              'tanzania': LOCAL_DATA_DIR + '/tanzania/data_w_planet.hdf5',
              'germany': LOCAL_DATA_DIR + '/germany/data.hdf5'}

GRID_DIR = { 'ghana': LOCAL_DATA_DIR + "/ghana", 
             'southsudan': LOCAL_DATA_DIR + "/southsudan", 
             'tanzania': LOCAL_DATA_DIR + "/tanzania",
             'germany': LOCAL_DATA_DIR + "/germany"}

GHANA_RASTER_DIR = GCP_DATA_DIR + '/ghana/raster/'
GHANA_RASTER_NPY_DIR = GCP_DATA_DIR + '/ghana/raster_npy/'
GHANA_S1_DIR = GCP_DATA_DIR + '/ghana/s1_npy'
GHANA_S2_DIR = GCP_DATA_DIR + '/ghana/s2_npy'


PRETRAINED_GERMANY_PATH = '/home/roserustowicz/crop-type-mapping/runs/20190303_fcncrnn_germany_s2_15dayagg_weightdecay01_noclouds_yesdoy_hiddendims128_avghiddenstatesyes_s2numbands10_yesearlyfeats_yesvegindices_best'

# HYPERPARAMETER SEARCH
INT_POWER_EXP = ["hidden_dims"]
REAL_POWER_EXP = ["weight_decay", "lr"]
INT_HP = ['batch_size', 'crnn_num_layers']
FLOAT_HP = ['weight_scale', 'percent_of_dataset']
STRING_HP = ['crnn_model_name']
BOOL_HP = ['use_s1', 'use_s2', 'include_clouds', 'bidirectional', 'least_cloudy',
           'avg_hidden_states', 'early_feats']
INT_CHOICE_HP = ['num_timesteps', 's2_num_bands']


# INT_POWER_EXP = []
# REAL_POWER_EXP = ["weight_decay", "lr"]
# INT_HP = ['batch_size']
# FLOAT_HP = ['weight_scale', 'dropout']
# STRING_HP = []
# BOOL_HP = ['use_s1', 'use_s2', 'include_clouds']
# INT_CHOICE_HP = ['s2_num_bands', 'num_timesteps']

HPS = [INT_POWER_EXP, REAL_POWER_EXP, INT_HP, FLOAT_HP, STRING_HP, BOOL_HP, INT_CHOICE_HP]

# LOSS WEIGHTS
GHANA_LOSS_WEIGHT = 1 - np.array([.17, .56, .16, .11])
GHANA_LOSS_WEIGHT = torch.tensor(GHANA_LOSS_WEIGHT, dtype=torch.float32).cuda()

SSUDAN_LOSS_WEIGHT = 1 - np.array([.72, .11, .10, .07])
SSUDAN_LOSS_WEIGHT = torch.tensor(SSUDAN_LOSS_WEIGHT, dtype=torch.float32).cuda()

TANZ_LOSS_WEIGHT = 1 - np.array([.64, .14, .12, .05, .05])
TANZ_LOSS_WEIGHT = torch.tensor(TANZ_LOSS_WEIGHT, dtype=torch.float32).cuda()
          
GERMANY_LOSS_WEIGHT = 1 - np.array([.02, .01, .07, .05, .03, .01, .02, .01, .01, .04, .01, .01, .27, .10, .01, .03, .32])
GERMANY_LOSS_WEIGHT = torch.tensor(GERMANY_LOSS_WEIGHT, dtype=torch.float32).cuda()

LOSS_WEIGHT = { 'ghana': GHANA_LOSS_WEIGHT, 
                'southsudan': SSUDAN_LOSS_WEIGHT,
                'tanzania': TANZ_LOSS_WEIGHT,
                'germany': GERMANY_LOSS_WEIGHT }

# BAND STATS

BANDS = { 's1': { 'VV': 0, 'VH': 1, 'RATIO': 2},
          's2': { '10': {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'RDED1': 3, 'RDED2': 4, 'RDED3': 5, 'NIR': 6, 'RDED4': 7, 'SWIR1': 8, 'SWIR2': 9},
                   '4': {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'NIR': 3}}, 
          'planet': { '4': {'BLUE': 0, 'GREEN': 1, 'RED': 2, 'NIR': 3}}}

MEANS = { 's1': { 'ghana': np.array([-10.50, -17.24, 1.17]), 
                  'southsudan': np.array([-9.02, -15.26, 1.15]), 
                  'tanzania': np.array([-9.80, -17.05, 1.30])},
          's2': { 'ghana': np.array([2620.00, 2519.89, 2630.31, 2739.81, 3225.22, 3562.64, 3356.57, 3788.05, 2915.40, 2102.65]),
                  'southsudan': np.array([2119.15, 2061.95, 2127.71, 2277.60, 2784.21, 3088.40, 2939.33, 3308.03, 2597.14, 1834.81]),
                  'tanzania': np.array([2551.54, 2471.35, 2675.69, 2799.99, 3191.33, 3453.16, 3335.64, 3660.05, 3182.23, 2383.79]),
                  'germany': np.array([1991.37, 2026.92, 2136.22, 6844.82, 9951.98, 11638.58, 3664.66, 12375.27, 7351.99, 5027.96])},
          'planet': { 'ghana': np.array([1264.81, 1255.25, 1271.10, 2033.22]),
                      'southsudan': np.array([1091.30, 1092.23, 1029.28, 2137.77]),
                      'tanzania': np.array([1014.16, 1023.31, 1114.17, 1813.49])},
          's2_cldfltr': { 'ghana': np.array([1362.68, 1317.62, 1410.74, 1580.05, 2066.06, 2373.60, 2254.70, 2629.11, 2597.50, 1818.43]),
                  'southsudan': np.array([1137.58, 1127.62, 1173.28, 1341.70, 1877.70, 2180.27, 2072.11, 2427.68, 2308.98, 1544.26]),
                  'tanzania': np.array([1148.76, 1138.87, 1341.54, 1517.01, 1937.15, 2191.31, 2148.05, 2434.61, 2774.64, 2072.09])} }

STDS = { 's1': { 'ghana': np.array([3.57, 4.86, 5.60]),
                 'southsudan': np.array([4.49, 6.68, 21.75]),
                 'tanzania': np.array([3.53, 4.78, 16.61])},
         's2': { 'ghana': np.array([2171.62, 2085.69, 2174.37, 2084.56, 2058.97, 2117.31, 1988.70, 2099.78, 1209.48, 918.19]),
                 'southsudan': np.array([2113.41, 2026.64, 2126.10, 2093.35, 2066.81, 2114.85, 2049.70, 2111.51, 1320.97, 1029.58]), 
                 'tanzania': np.array([2290.97, 2204.75, 2282.90, 2214.60, 2182.51, 2226.10, 2116.62, 2210.47, 1428.33, 1135.21]),
                 'germany': np.array([1943.62, 1755.82, 1841.09, 5703.38, 5104.90, 5136.54, 1663.27, 5125.05, 3682.57, 3273.71])},
         'planet': { 'ghana': np.array([602.51, 598.66, 637.06, 966.27]),
                     'southsudan': np.array([526.06, 517.05, 543.74, 1022.14]),
                     'tanzania': np.array([492.33, 492.71, 558.90, 833.65])},
         's2_cldfltr': { 'ghana': np.array([511.19, 495.87, 591.44, 590.27, 745.81, 882.05, 811.14, 959.09, 964.64, 809.53]),
                 'southsudan': np.array([548.64, 547.45, 660.28, 677.55, 896.28, 1066.91, 1006.01, 1173.19, 1167.74, 865.42]),
                 'tanzania': np.array([462.40, 449.22, 565.88, 571.42, 686.04, 789.04, 758.31, 854.39, 1071.74, 912.79])} }

# OTHER PER COUNTRY CONSTANTS
NUM_CLASSES = { 'ghana': 4,
                'southsudan': 4,
                'tanzania': 5,
                'germany': 17 }

GRID_SIZE = { 'ghana': 32, 
              'southsudan': 32, 
              'tanzania': 32, 
              'germany': 48 }

CM_LABELS = { 'ghana': [0, 1, 2, 3], 
              'southsudan': [0, 1, 2, 3], 
              'tanzania': [0, 1, 2, 3, 4],
              'germany': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] }

CROPS = { 'ghana': ['groundnut', 'maize', 'rice', 'soya bean'], 
          'southsudan': ['sorghum', 'maize', 'rice', 'groundnut'], 
          'tanzania': ['maize', 'beans', 'sunflower', 'chickpeas', 'wheat'],
          'germany': ['sugar beet', 'summer oat', 'meadow', 'rapeseed', 'hop', 'winter spelt', 
                      'winter triticale', 'beans', 'peas', 'potato', 'soybeans', 'asparagus', 
                      'winter wheat', 'winter barley', 'winter rye', 'summer barley', 'maize']}
