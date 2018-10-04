import os
import re
import numpy as np
import pandas as pd
import healpy as hp
import opsimsummary as oss

wfd_nheal = 16420
FOV_area = np.pi*(3.5/2.0)*(3.5/2.0)*0.82

wfd_columns = ['total_visits_u_wfd',	'total_visits_g_wfd',	'total_visits_r_wfd',	'total_visits_i_wfd',	'total_visits_z_wfd',	'total_visits_y_wfd', 'survey_area'	]

wfd_df = pd.DataFrame(columns=wfd_columns)

survey_path = '/share/data1/csetzer/lsst_cadences/'
survey_dir = os.listdir(survey_path)

for survey in survey_dir:
    survey_name = survey.strip('.db')
    if re.search('minion', survey) is None:
        cadence = oss.OpSimOutput.fromOpSimDB(survey_path + survey, subset='wfd',
                                                   opsimversion='lsstv4').summary
        field_key = 'fieldId'
    else:
        cadence = oss.OpSimOutput.fromOpSimDB(survey_path + survey, subset='wfd',
                                                   opsimversion='lsstv3').summary
        field_key = 'fieldID'
    if survey_name == 'kraken_2044' or survey_name == 'kraken_2042' or survey_name == 'nexus_2097' or survey_name == 'mothra_2049':
        field_key = 'fieldRA'
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:

        filter_cadence = cadence.query('filter == \'{}\''.format(band))

        band_num = len(filter_cadence['expMJD'].unique())

        wfd_df.at[survey_name, 'total_visits_{}_wfd'.format(band)] = band_num


    wfd_df.at[survey_name, 'survey_area'] = len(list(cadence[field_key].unique()))*FOV_area
    print(len(cadence[field_key]))
    print(len(cadence[field_key].unique()))


wfd_df.to_csv('wfd_properties.csv')
