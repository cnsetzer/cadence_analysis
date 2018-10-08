import os
import re
import numpy as np
import pandas as pd
import opsimsummary as oss


ddf_columns = ['median_visits_u_ddf',	'median_visits_g_ddf',	'median_visits_r_ddf',	'median_visits_i_ddf',	'median_visits_z_ddf',	'median_visits_y_ddf','total_visits_u_ddf',	'total_visits_g_ddf',	'total_visits_r_ddf',	'total_visits_i_ddf',	'total_visits_z_ddf',	'total_visits_y_ddf',	'median_single_visit_depth_u_ddf',	'median_single_visit_depth_g_ddf',	'median_single_visit_depth_r_ddf',	'median_single_visit_depth_i_ddf',	'median_single_visit_depth_z_ddf',	'median_single_visit_depth_y_ddf', 'ddf_ra', 'ddf_dec', 'number_ddf_fields']

ddf_df = pd.DataFrame(columns=ddf_columns)

survey_path = '/share/data1/csetzer/lsst_cadences/'
survey_dir = os.listdir(survey_path)

for survey in survey_dir:
    survey_name = survey.strip('.db')
    if re.search('minion', survey) is not None:
        cadence = oss.OpSimOutput.fromOpSimDB(survey_path + survey, subset='ddf',
                                                   opsimversion='lsstv3').summary
        field_key = 'fieldID'
    elif re.serach('alt_sched', survey) is not None:
        cadence = oss.OpSimOutput.fromOpSimDB(survey_path + survey, subset='ddf',
                                                   opsimversion='sstf', ).summary
        field_key = 'fieldRA'
    else:
        cadence = oss.OpSimOutput.fromOpSimDB(survey_path + survey, subset='ddf',
                                                   opsimversion='lsstv4').summary
        field_key = 'fieldId'


    if survey_name == 'kraken_2044' or survey_name == 'kraken_2042' or survey_name == 'nexus_2097' or survey_name == 'mothra_2049' or survey_name == 'colossus_2683' or survey_name == 'astro-lsst-01_2039':
        field_key = 'fieldRA'

    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        band_num = []

        filter_cadence = cadence.query('filter == \'{}\''.format(band))
        for field in list(cadence[field_key].unique()):
            if field in list(filter_cadence[field_key].unique()):
                band_num.append(len(list(filter_cadence.query('{0} == {1}'.format(field_key,field))['expMJD'])))
            else:
                band_num.append(0)

        ddf_df.at[survey_name, 'median_visits_{}_ddf'.format(band)] = np.median(band_num)
        ddf_df.at[survey_name, 'median_single_visit_depth_{}_ddf'.format(band)] = filter_cadence['fiveSigmaDepth'].median()
        ddf_df.at[survey_name, 'total_visits_{}_ddf'.format(band)] = np.sum(band_num)


    ddf_ra = []
    ddf_dec = []
    for field in list(cadence[field_key].unique()):
        if re.search('minion',survey_name) is None:
            ddf_ra.append(np.asscalar(np.mean(cadence.query('{0} == {1}'.format(field_key,field))['fieldRA'].unique())))
            ddf_dec.append(np.asscalar(np.mean(cadence.query('{0} == {1}'.format(field_key,field))['fieldDec'].unique())))
        else:
            ddf_ra.append(np.asscalar(np.rad2deg(np.mean(cadence.query('{0} == {1}'.format(field_key,field))['fieldRA'].unique()))))
            ddf_dec.append(np.asscalar(np.rad2deg(np.mean(cadence.query('{0} == {1}'.format(field_key,field))['fieldDec'].unique()))))


    ddf_df.at[survey_name, 'ddf_ra'] = ddf_ra
    ddf_df.at[survey_name, 'ddf_dec'] = ddf_dec
    ddf_df.at[survey_name,'number_ddf_fields'] = len(list(cadence[field_key].unique()))


ddf_df.to_csv('ddf_properties.csv')
