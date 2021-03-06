import os
import re
import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
from astrotog.functions import scolnic_detections as scd
from astrotog.functions import scolnic_like_detections as scld
import seaborn as sns


def determine_ddf_detections(ddf_properties, cadence_results):
    field_rad = np.deg2rad(3.5/2.0)
    for cadence in cadence_results.keys():
        for model_key in cadence_results[cadence].keys():
            detections1 = cadence_results[cadence][model_key]['data']['scolnic_detections']
            detections2 = cadence_results[cadence][model_key]['data']['scolnic_like_detections']
            detections3 = cadence_results[cadence][model_key]['data']['scolnic_detections_no_coadd']
            detections4 = cadence_results[cadence][model_key]['data']['scolnic_like_detections_no_coadd']

            cad_transient_params = cadence_results[cadence][model_key]['data']['parameters']
            ids_in_ddf = []
            num_ddf_fields = ddf_properties.at[cadence, 'number_ddf_fields']
            for i in range(num_ddf_fields):
                field_ra = np.deg2rad(eval(ddf_properties.at[cadence, 'ddf_ra'])[i])
                field_dec = np.deg2rad(eval(ddf_properties.at[cadence, 'ddf_dec'])[i])
                inter1 = cad_transient_params.query('ra - {0} <= {1} & {0} - ra <= {1}'.format(field_ra,field_rad))
                inter2 = inter1.query('dec - {0} <= {1} & {0} - dec <= {1}'.format(field_dec,field_rad))
                for index, row in inter2.iterrows():
                    ra = row['ra']
                    dec = row['dec']
                    angdist = np.arccos(np.sin(field_dec)*np.sin(dec) +
                                np.cos(field_dec) *
                                np.cos(dec)*np.cos(ra - field_ra))
                    if angdist < field_rad:
                        ids_in_ddf.append(row['transient_id'])

            cadence_results[cadence][model_key]['subset_detections'] = {}
            cadence_results[cadence][model_key]['subset_detections']['total'] = {}
            cadence_results[cadence][model_key]['subset_detections']['wfd'] = {}
            cadence_results[cadence][model_key]['subset_detections']['ddf'] = {}
            cadence_results[cadence][model_key]['subset_detections']['total']['scolnic'] = detections1
            cadence_results[cadence][model_key]['subset_detections']['ddf']['scolnic'] = detections1[detections1['transient_id'].isin(ids_in_ddf)]
            cadence_results[cadence][model_key]['subset_detections']['wfd']['scolnic'] = detections1[~detections1['transient_id'].isin(ids_in_ddf)]
            cadence_results[cadence][model_key]['subset_detections']['total']['scolnic_no_coadd'] = detections2
            cadence_results[cadence][model_key]['subset_detections']['ddf']['scolnic_no_coadd'] = detections2[detections2['transient_id'].isin(ids_in_ddf)]
            cadence_results[cadence][model_key]['subset_detections']['wfd']['scolnic_no_coadd'] = detections2[~detections2['transient_id'].isin(ids_in_ddf)]
            cadence_results[cadence][model_key]['subset_detections']['total']['scolnic_like'] = detections3
            cadence_results[cadence][model_key]['subset_detections']['ddf']['scolnic_like'] = detections3[detections3['transient_id'].isin(ids_in_ddf)]
            cadence_results[cadence][model_key]['subset_detections']['wfd']['scolnic_like'] = detections3[~detections3['transient_id'].isin(ids_in_ddf)]
            cadence_results[cadence][model_key]['subset_detections']['total']['scolnic_like_no_coadd'] = detections4
            cadence_results[cadence][model_key]['subset_detections']['ddf']['scolnic_like_no_coadd'] = detections4[detections4['transient_id'].isin(ids_in_ddf)]
            cadence_results[cadence][model_key]['subset_detections']['wfd']['scolnic_like_no_coadd'] = detections4[~detections4['transient_id'].isin(ids_in_ddf)]
    return cadence_results


def process_counts(results):
    for cadence in results.keys():
        for model in results[cadence].keys():
            results[cadence][model]['detected_counts'] = {}
            results[cadence][model]['detected_counts']['total'] = {}
            results[cadence][model]['detected_counts']['wfd'] = {}
            results[cadence][model]['detected_counts']['ddf'] = {}

            results[cadence][model]['detected_counts']['total']['scolnic'] = len(results[cadence][model]['subset_detections']['total']['scolnic']['transient_id'].unique())
            results[cadence][model]['detected_counts']['wfd']['scolnic'] = len(results[cadence][model]['subset_detections']['wfd']['scolnic']['transient_id'].unique())
            results[cadence][model]['detected_counts']['ddf']['scolnic'] = len(results[cadence][model]['subset_detections']['ddf']['scolnic']['transient_id'].unique())

            results[cadence][model]['detected_counts']['total']['scolnic_no_coadd'] = len(results[cadence][model]['subset_detections']['total']['scolnic_no_coadd']['transient_id'].unique())
            results[cadence][model]['detected_counts']['wfd']['scolnic_no_coadd'] = len(results[cadence][model]['subset_detections']['wfd']['scolnic_no_coadd']['transient_id'].unique())
            results[cadence][model]['detected_counts']['ddf']['scolnic_no_coadd'] = len(results[cadence][model]['subset_detections']['ddf']['scolnic_no_coadd']['transient_id'].unique())

            results[cadence][model]['detected_counts']['total']['scolnic_like'] = len(results[cadence][model]['subset_detections']['total']['scolnic_like']['transient_id'].unique())
            results[cadence][model]['detected_counts']['wfd']['scolnic_like'] = len(results[cadence][model]['subset_detections']['wfd']['scolnic_like']['transient_id'].unique())
            results[cadence][model]['detected_counts']['ddf']['scolnic_like'] = len(results[cadence][model]['subset_detections']['ddf']['scolnic_like']['transient_id'].unique())

            results[cadence][model]['detected_counts']['total']['scolnic_like_no_coadd'] = len(results[cadence][model]['subset_detections']['total']['scolnic_like_no_coadd']['transient_id'].unique())
            results[cadence][model]['detected_counts']['wfd']['scolnic_like_no_coadd'] = len(results[cadence][model]['subset_detections']['wfd']['scolnic_like_no_coadd']['transient_id'].unique())
            results[cadence][model]['detected_counts']['ddf']['scolnic_like_no_coadd'] = len(results[cadence][model]['subset_detections']['ddf']['scolnic_like_no_coadd']['transient_id'].unique())

    return results


def plotting_dataframe(results, wfd_props, ddf_props):
    results_df = pd.DataFrame(index= wfd_props.index, columns=['wfd_ross_scolnic','ddf_ross_scolnic', 'total_ross_scolnic','wfd_scolnic_scolnic','ddf_scolnic_scolnic', 'total_scolnic_scolnic',
                                                                'wfd_ross_scolnic_no_coadd','ddf_ross_scolnic_no_coadd', 'total_ross_scolnic_no_coadd','wfd_scolnic_scolnic_no_coadd','ddf_scolnic_scolnic_no_coadd', 'total_scolnic_scolnic_no_coadd',
                                                                'wfd_ross_scolnic_like','ddf_ross_scolnic_like', 'total_ross_scolnic_like','wfd_scolnic_scolnic_like','ddf_scolnic_scolnic_like', 'total_scolnic_scolnic_like',
                                                                'wfd_ross_scolnic_like_no_coadd','ddf_ross_scolnic_like_no_coadd', 'total_ross_scolnic_like_no_coadd','wfd_scolnic_scolnic_like_no_coadd','ddf_scolnic_scolnic_like_no_coadd', 'total_scolnic_scolnic_like_no_coadd'])
    results_df = results_df.join(wfd_props)
    results_df = results_df.join(ddf_props)
    for cadence in results.keys():
        for model in results[cadence].keys():
            if model == 'rosswog':
                results_df.at[cadence, 'wfd_ross_scolnic'] = results[cadence][model]['detected_counts']['wfd']['scolnic']
                results_df.at[cadence, 'ddf_ross_scolnic'] = results[cadence][model]['detected_counts']['ddf']['scolnic']
                results_df.at[cadence, 'total_ross_scolnic'] = results[cadence][model]['detected_counts']['total']['scolnic']

                results_df.at[cadence, 'wfd_ross_scolnic_no_coadd'] = results[cadence][model]['detected_counts']['wfd']['scolnic_no_coadd']
                results_df.at[cadence, 'ddf_ross_scolnic_no_coadd'] = results[cadence][model]['detected_counts']['ddf']['scolnic_no_coadd']
                results_df.at[cadence, 'total_ross_scolnic_no_coadd'] = results[cadence][model]['detected_counts']['total']['scolnic_no_coadd']

                results_df.at[cadence, 'wfd_ross_scolnic_like'] = results[cadence][model]['detected_counts']['wfd']['scolnic_like']
                results_df.at[cadence, 'ddf_ross_scolnic_like'] = results[cadence][model]['detected_counts']['ddf']['scolnic_like']
                results_df.at[cadence, 'total_ross_scolnic_like'] = results[cadence][model]['detected_counts']['total']['scolnic_like']

                results_df.at[cadence, 'wfd_ross_scolnic_like_no_coadd'] = results[cadence][model]['detected_counts']['wfd']['scolnic_like_no_coadd']
                results_df.at[cadence, 'ddf_ross_scolnic_like_no_coadd'] = results[cadence][model]['detected_counts']['ddf']['scolnic_like_no_coadd']
                results_df.at[cadence, 'total_ross_scolnic_like_no_coadd'] = results[cadence][model]['detected_counts']['total']['scolnic_like_no_coadd']

            elif model == 'scolnic':
                results_df.at[cadence, 'wfd_scolnic_scolnic'] = results[cadence][model]['detected_counts']['wfd']['scolnic']
                results_df.at[cadence, 'ddf_scolnic_scolnic'] = results[cadence][model]['detected_counts']['ddf']['scolnic']
                results_df.at[cadence, 'total_scolnic_scolnic'] = results[cadence][model]['detected_counts']['total']['scolnic']

                results_df.at[cadence, 'wfd_scolnic_scolnic_no_coadd'] = results[cadence][model]['detected_counts']['wfd']['scolnic_no_coadd']
                results_df.at[cadence, 'ddf_scolnic_scolnic_no_coadd'] = results[cadence][model]['detected_counts']['ddf']['scolnic_no_coadd']
                results_df.at[cadence, 'total_scolnic_scolnic_no_coadd'] = results[cadence][model]['detected_counts']['total']['scolnic_no_coadd']

                results_df.at[cadence, 'wfd_scolnic_scolnic_like'] = results[cadence][model]['detected_counts']['wfd']['scolnic_like']
                results_df.at[cadence, 'ddf_scolnic_scolnic_like'] = results[cadence][model]['detected_counts']['ddf']['scolnic_like']
                results_df.at[cadence, 'total_scolnic_scolnic_like'] = results[cadence][model]['detected_counts']['total']['scolnic_like']

                results_df.at[cadence, 'wfd_scolnic_scolnic_like_no_coadd'] = results[cadence][model]['detected_counts']['wfd']['scolnic_like_no_coadd']
                results_df.at[cadence, 'ddf_scolnic_scolnic_like_no_coadd'] = results[cadence][model]['detected_counts']['ddf']['scolnic_like_no_coadd']
                results_df.at[cadence, 'total_scolnic_scolnic_like_no_coadd'] = results[cadence][model]['detected_counts']['total']['scolnic_like_no_coadd']
    return results_df


################################################################################
def plot_trends(results_df, output_path):
    for subset in ['wfd', 'ddf', 'total']:
        for property in list(results_df.columns):
            if (re.search('ross', property) is None) and (re.search('scolnic', property) is None):
                if (re.search('wfd',property) is not None and subset == 'wfd') or (re.search('ddf',property) is not None and subset == 'ddf' and re.search('dec|ra',property) is None) or (re.search('wfd',property) is None and subset == 'total' and re.search('ddf',property) is None):
                    fig = plt.figure()
                    plt.scatter(results_df[property], results_df['{}_ross'.format(subset)], c='r')
                    plt.scatter(results_df[property], results_df['{}_scolnic'.format(subset)], c='k', marker='x')
                    plt.xlabel('{}'.format(property))
                    plt.ylabel('Number of Detections')
                    plt.legend(['Rosswog', 'DES-GW'])
                    fig.savefig(output_path + '{0}_counts_vs_{1}.pdf'.format(subset, property), bbox_inches='tight')
                    plt.close(fig)
    return
################################################################################

################################################################################
def plot_trends_2D(results_df, output_path):
    for subset in ['wfd', 'ddf', 'total']:
        for property1 in list(results_df.columns):
            for property2 in list(results_df.columns):
                if property1 == property2:
                    continue
                else:
                    if (re.search('ross', property1) is None) and (re.search('scolnic', property1) is None) and (re.search('ross', property2) is None) and (re.search('scolnic', property2) is None):
                        if ((re.search('wfd',property1) is not None) and subset == 'wfd') or ((re.search('ddf',property1) is not None) and (subset == 'ddf') and (re.search('dec|ra',property1) is None)) or ((re.search('wfd',property1) is None) and (subset == 'total') and (re.search('ddf',property1) is None)):
                            if ((re.search('wfd',property2) is not None) and subset == 'wfd') or ((re.search('ddf',property2) is not None) and (subset == 'ddf') and (re.search('dec|ra',property2) is None)) or ((re.search('wfd',property2) is None) and (subset == 'total') and (re.search('ddf',property2) is None)):
                                exist1 = os.path.isfile(output_path + '{0}_counts_vs_{1}_and_{2}.pdf'.format(subset, property1, property2))
                                exist2 = os.path.isfile(output_path + '{0}_counts_vs_{1}_and_{2}.pdf'.format(subset, property2, property1))
                                if exist1 or exist2:
                                    continue
                                else:
                                    fig = plt.figure()
                                    plt.scatter(results_df[property1], results_df[property2], c=results_df['{}_ross'.format(subset)], cmap='inferno', s=50)
                                    plt.scatter(results_df[property1], results_df[property2], c=results_df['{}_scolnic'.format(subset)], cmap='inferno', marker='x', s=55)
                                    plt.xlabel('{}'.format(property1))
                                    plt.ylabel('{}'.format(property2))
                                    plt.colorbar(label='Number of Detections')
                                    plt.legend(['Rosswog', 'DES-GW'])
                                    fig.savefig(output_path + '{0}_counts_vs_{1}_and_{2}.pdf'.format(subset, property1, property2), bbox_inches='tight')
                                    plt.close(fig)
    return
################################################################################

def get_cadence_results(results_paths):
    results = {}
    for path in results_paths:
        output_dirs = os.listdir(path)
        for directory in output_dirs:
            if directory == '.DS_Store':
                continue
            else:
                name_split = re.split('_',directory)
                model = name_split[1]
                if re.search('alt_sched', directory) is not None:
                    if re.search('rolling', directory) is not None:
                        name = name_split[2] + '_' + name_split[3] + '_' + name_split[4]
                    else:
                        name = name_split[2] + '_' + name_split[3]
                elif re.search('baseline',name_split[2]) is not None:
                    name = name_split[2]
                elif re.search('astro',directory) is not None:
                    name = 'astro-lsst-01_2039'
                else:
                    name_split2 = re.split("(\d+)", name_split[2])
                    name = name_split2[0] + '_' + name_split2[1]

                if re.search('minion',name) is not None:
                    name += '_desc_dithered_v4'
                for string in name_split:
                    if re.search('var', string) is not None:
                        name += string
                if name not in results.keys():
                    results[name] = {}

                results[name][model] = {}
                results[name][model]['data'] = {}
                #results[name][model]['data']['observations'] = pd.read_csv(path + directory +'/observations.csv',index_col=0)
                results[name][model]['data']['parameters'] = pd.read_csv(path + directory +'/modified_parameters.csv',index_col=0)
                #results[name][model]['data']['coadded_observations'] = pd.read_csv(path + directory +'/coadded_observations.csv',index_col=0)
                results[name][model]['data']['other_observations'] = pd.read_csv(path + directory +'/other_observations.csv',index_col=0)
                results[name][model]['data']['scolnic_detections'] = pd.read_csv(path + directory +'/scolnic_detections.csv',index_col=0)
                results[name][model]['data']['scolnic_detections_no_coadd'] = pd.read_csv(path + directory +'/scolnic_detections_no_coadd.csv',index_col=0)
                results[name][model]['data']['scolnic_like_detections'] = pd.read_csv(path + directory +'/scolnic_like_detections.csv',index_col=0)
                results[name][model]['data']['scolnic_like_detections_no_coadd'] = pd.read_csv(path + directory +'/scolnic_like_detections_no_coadd.csv',index_col=0)
    return results


def get_single_cadence_results(path, directory):
    results = {}
    name_split = re.split('_', directory)
    model = name_split[1]
    if re.search('alt_sched', directory) is not None:
        if re.search('rolling', directory) is not None:
            name = name_split[2] + '_' + name_split[3] + '_' + name_split[4]
        else:
            name = name_split[2] + '_' + name_split[3]
    elif re.search('baseline',name_split[2]) is not None:
        name = name_split[2]
    elif re.search('astro',directory) is not None:
        name = 'astro-lsst-01_2039'
    else:
        name_split2 = re.split("(\d+)", name_split[2])
        name = name_split2[0] + '_' + name_split2[1]

    if re.search('minion',name) is not None:
        name += '_desc_dithered_v4'
    for string in name_split:
        if re.search('var', string) is not None:
            name += string
    if name not in results.keys():
        results[name] = {}

    results[name][model] = {}
    results[name][model]['data'] = {}
    #results[name][model]['data']['observations'] = pd.read_csv(path + directory +'/observations.csv',index_col=0)
    results[name][model]['data']['parameters'] = pd.read_csv(path + directory +'/modified_parameters.csv',index_col=0)
    #results[name][model]['data']['coadded_observations'] = pd.read_csv(path + directory +'/coadded_observations.csv',index_col=0)
    results[name][model]['data']['other_observations'] = pd.read_csv(path + directory +'/other_observations.csv',index_col=0)
    results[name][model]['data']['scolnic_detections'] = pd.read_csv(path + directory +'/scolnic_detections.csv',index_col=0)
    results[name][model]['data']['scolnic_detections_no_coadd'] = pd.read_csv(path + directory +'/scolnic_detections_no_coadd.csv',index_col=0)
    results[name][model]['data']['scolnic_like_detections'] = pd.read_csv(path + directory +'/scolnic_like_detections.csv',index_col=0)
    results[name][model]['data']['scolnic_like_detections_no_coadd'] = pd.read_csv(path + directory +'/scolnic_like_detections_no_coadd.csv',index_col=0)
    return results


def param_subset(param_df, ddf_subset):
    subset_df = pd.DataFrame(index=param_df.index, columns=['subset'])
    in_ddf = list(ddf_subset['transient_id'].unique())
    for id in list(param_df['transient_id']):
        if id in in_ddf:
            subset_df.at[id,'subset'] = 'ddf'
            print(subset_df.at[id,'subset'])
        else:
            subset_df.at[id,'subset'] = 'wfd'

    param_df = param_df.join(subset_df)
    return param_df


def redshift_distribution(param_df):
    z_min = 0.0
    z_max = 0.5
    bin_size = 0.025
    n_bins = int(round((z_max-z_min)/bin_size))
    all_zs = list(param_df['true_redshift'].values)
    wfd_check = param_df.query('subset == \'{}\''.format('wfd'))
    wfd_isnt_detected = wfd_check.query('detected == {}'.format(True)).empty
    if wfd_isnt_detected is True:
        wfd_detect_zs = []
        wfd_max_depth_detect = []
    else:
        wfd_detect_zs = list(param_df.query('subset == \'{0}\' & detected == {1}'.format('wfd', True))['true_redshift'])
        wfd_max_depth_detect = list(param_df[param_df['true_redshift'] <= max(wfd_detect_zs)]['true_redshift'])

    ddf_check = param_df.query('subset == \'{}\''.format('ddf'))
    ddf_isnt_detected = ddf_check.query('detected == {}'.format(True)).empty
    if ddf_isnt_detected is True:
        ddf_detect_zs = []
        ddf_max_depth_detect = []
    else:
        print('DDF!')
        ddf_detect_zs = list(param_df.query('subset == \'{0}\' & detected == {1}'.format('ddf', True))['true_redshift'])
        ddf_max_depth_detect = list(param_df[param_df['true_redshift'] <= max(ddf_detect_zs)]['true_redshift'])
    detect_zs = wfd_detect_zs
    detect_zs.extend(ddf_detect_zs)
    total_eff = (len(detect_zs)/len(all_zs))*100
    max_detect_zs = wfd_max_depth_detect
    max_detect_zs.extend(ddf_max_depth_detect)
    max_depth_eff = (len(detect_zs)/len(max_detect_zs))*100

    print('The redshift range of all sources is {0:.4f} to {1:.4f}.'.format(min(all_zs), max(all_zs)))
    print('The redshift range of the detected sources is {0:.4f} to {1:.4f}.'.format(min(detect_zs), max(detect_zs)))
    print('There are {0} detected transients out of {1}, which is an efficiency of {2:2.2f}%  of the total simulated number.'.format(len(detect_zs), len(all_zs), total_eff))
    print('However, this is an efficiency of {0:2.2f}%  of the total that occur within the range that was detected by {1}.'.format(max_depth_eff, 'LSST'))
    # Create the histogram'
    N_z_dist_fig = plt.figure()
    plt.hist(x=all_zs, bins=n_bins, range=(z_min, z_max), histtype='step', color='red', label='All Sources', linewidth=3.0)
    plt.hist(x=[wfd_detect_zs, ddf_detect_zs], bins=n_bins, range=(z_min, z_max), histtype='stepfilled', alpha=0.3, label='Detected Sources', stacked=True)
    # plt.tick_params(which='both', length=10, width=1.5)
    plt.yscale('log')
    plt.legend(loc=2)
    plt.xlabel('z')
    plt.ylabel(r'$N(z)$')
    plt.title('Redshift Distribution ({0:.3f} bins, {1} WFD, {2} DDF)'.format(bin_size,len(wfd_detect_zs), len(ddf_detect_zs)))
    return N_z_dist_fig

def overlay_redshift_distribution(param_df1, param_df2):
    z_min = 0.0
    z_max = 0.5
    bin_size = 0.025
    n_bins = int(round((z_max-z_min)/bin_size))
    all_zs = list(param_df1['true_redshift'].values)
    wfd_check1 = param_df1.query('subset == \'{}\''.format('wfd'))
    wfd_isnt_detected1 = wfd_check1.query('detected == {}'.format(True)).empty
    if wfd_isnt_detected1 is True:
        wfd_detect_zs1 = []
        wfd_max_depth_detect1 = []
    else:
        wfd_detect_zs1 = list(param_df1.query('subset == \'{0}\' & detected == {1}'.format('wfd', True))['true_redshift'])
        wfd_max_depth_detect1 = list(param_df1[param_df1['true_redshift'] <= max(wfd_detect_zs1)]['true_redshift'])

    ddf_check1 = param_df1.query('subset == \'{}\''.format('ddf'))
    ddf_isnt_detected1 = ddf_check1.query('detected == {}'.format(True)).empty
    if ddf_isnt_detected1 is True:
        ddf_detect_zs1 = []
        ddf_max_depth_detect1 = []
    else:
        print('DDF!')
        ddf_detect_zs1 = list(param_df1.query('subset == \'{0}\' & detected == {1}'.format('ddf', True))['true_redshift'])
        ddf_max_depth_detect1 = list(param_df1[param_df1['true_redshift'] <= max(ddf_detect_zs1)]['true_redshift'])
    detect_zs1 = wfd_detect_zs1
    detect_zs1.extend(ddf_detect_zs1)
    total_eff1 = (len(detect_zs1)/len(all_zs))*100
    max_detect_zs1 = wfd_max_depth_detect1
    max_detect_zs1.extend(ddf_max_depth_detect1)
    max_depth_eff1 = (len(detect_zs1)/len(max_detect_zs1))*100

    wfd_check2 = param_df2.query('subset == \'{}\''.format('wfd'))
    wfd_isnt_detected2 = wfd_check2.query('detected == {}'.format(True)).empty
    if wfd_isnt_detected2 is True:
        wfd_detect_zs2 = []
        wfd_max_depth_detect2 = []
    else:
        wfd_detect_zs2 = list(param_df2.query('subset == \'{0}\' & detected == {1}'.format('wfd', True))['true_redshift'])
        wfd_max_depth_detect2 = list(param_df2[param_df2['true_redshift'] <= max(wfd_detect_zs2)]['true_redshift'])

    ddf_check2 = param_df2.query('subset == \'{}\''.format('ddf'))
    ddf_isnt_detected2 = ddf_check2.query('detected == {}'.format(True)).empty
    if ddf_isnt_detected2 is True:
        ddf_detect_zs2 = []
        ddf_max_depth_detect2 = []
    else:
        print('DDF!')
        ddf_detect_zs2 = list(param_df2.query('subset == \'{0}\' & detected == {1}'.format('ddf', True))['true_redshift'])
        ddf_max_depth_detect2 = list(param_df2[param_df2['true_redshift'] <= max(ddf_detect_zs2)]['true_redshift'])
    detect_zs2 = wfd_detect_zs2
    detect_zs2.extend(ddf_detect_zs2)
    total_eff2 = (len(detect_zs2)/len(all_zs))*100
    max_detect_zs2 = wfd_max_depth_detect2
    max_detect_zs2.extend(ddf_max_depth_detect2)
    max_depth_eff2 = (len(detect_zs2)/len(max_detect_zs2))*100

    print('The redshift range of all sources is {0:.4f} to {1:.4f}.'.format(min(all_zs), max(all_zs)))
    print('The redshift range of the detected sources is {0:.4f} to {1:.4f}.'.format(min(detect_zs1), max(detect_zs1)))
    print('There are {0} detected transients out of {1}, which is an efficiency of {2:2.2f}%  of the total simulated number.'.format(len(detect_zs1), len(all_zs), total_eff1))
    print('However, this is an efficiency of {0:2.2f}%  of the total that occur within the range that was detected by {1}.'.format(max_depth_eff1, 'LSST'))
    # Create the histogram'
    N_z_dist_fig = plt.figure()
    plt.hist(x=all_zs, bins=n_bins, range=(z_min, z_max), histtype='step', color='red', label='All Sources', linewidth=3.0)
    plt.hist(x=[wfd_detect_zs1, ddf_detect_zs1], bins=n_bins, range=(z_min, z_max), histtype='stepfilled', alpha=0.4, label='DES-GW - Detected', stacked=True)
    plt.hist(x=[wfd_detect_zs2, ddf_detect_zs2], bins=n_bins, range=(z_min, z_max), histtype='stepfilled', alpha=0.25, label='SAEE - Detected', stacked=True, color=['k','y'])
    # plt.tick_params(which='both', length=10, width=1.5)
    plt.axvline(x=0.102, linestyle='--', color='k')
    plt.text(0.11, 500, 'LIGO A+ Sensitivity', rotation=90, fontsize=int(13))
    plt.axvline(x=0.204, linestyle='-.', color='k')
    plt.text(0.21, 500, 'LIGO Voyager Sensitivity', rotation=90, fontsize=int(13))
    plt.axvline(x=0.5, linestyle='-', color='k')
    plt.text(0.507, 1100, 'Einstein Telescope Sensitivity', rotation=90, fontsize=int(13))
    plt.yscale('log')
    plt.legend(loc=2, fontsize=int(13))
    plt.xlabel('z', fontsize=int(14))
    plt.ylabel(r'$N(z)$', fontsize=int(14))
    plt.title('({0:.3f} bins, {1}/{2} Detections)'.format(bin_size,len(detect_zs1), len(detect_zs2)), fontsize=int(14))
    return N_z_dist_fig

################################################################################
def band_delta_N(plot_df, results):
    results_df = pd.DataFrame(index= plot_df.index, columns=['delta_N_u_ross', 'delta_N_g_ross', 'delta_N_r_ross', 'delta_N_i_ross', 'delta_N_z_ross', 'delta_N_y_ross', 'delta_N_u_scolnic', 'delta_N_g_scolnic', 'delta_N_r_scolnic', 'delta_N_i_scolnic', 'delta_N_z_scolnic', 'delta_N_y_scolnic'])
    results_df = plot_df.join(results_df)
    for cadence in results.keys():
        print(cadence)
        for model in results[cadence].keys():
            if model == 'rosswog':
                for band in ['u','g','r','i','z','y']:
                    detect_bandless = results[cadence][model]['data']['scolnic_detections'].query('bandfilter != \'{}\''.format(band))
                    new_N = len(scd(results[cadence][model]['data']['parameters'], detect_bandless, results[cadence][model]['data']['other_observations'])['transient_id'].unique())
                    results_df.at[cadence, 'delta_N_{}_ross'] = new_N - results_df.at[cadence, 'total_ross_scolnic']
            elif model == 'scolnic':
                for band in ['u','g','r','i','z','y']:
                    detect_bandless = results[cadence][model]['data']['scolnic_detections'].query('bandfilter != \'{}\''.format(band))
                    new_N = len(scd(results[cadence][model]['data']['parameters'], detect_bandless, results[cadence][model]['data']['other_observations'])['transient_id'].unique())
                    results_df.at[cadence, 'delta_N_{}_scolnic'] = new_N - results_df.at[cadence, 'total_ross_scolnic']
    return results_df


def new_band_delta_N(cadence, model, params_df, other_obs_df, detections_df, num_detect_all_bands, like=True):
    results_df = pd.DataFrame(index=[str(cadence)], columns=['delta_N_u_{}'.format(model), 'delta_N_g_{}'.format(model), 'delta_N_r_{}'.format(model), 'delta_N_i_{}'.format(model), 'delta_N_z_{}'.format(model), 'delta_N_y_{}'.format(model)])
    for band in ['u', 'g', 'r', 'i', 'z', 'y']:
        detect_bandless = detections_df.query('bandfilter != \'{}\''.format(band))
        if like is True:
            new_N = len(scld(params_df, detect_bandless, other_obs_df)['transient_id'].unique())
        else:
            new_N = len(scd(params_df, detect_bandless, other_obs_df)['transient_id'].unique())
        results_df.at[str(cadence), 'delta_N_{}'] = new_N - len(detections_df['transient_id'].unique())
    return results_df

################################################################################

################################################################################
def plot_delta_n(plot_df):
    for i in range(2):
        bands = ['u', 'g', 'r', 'i', 'z', 'y']
        scol_plot = []
        ross_plot = []
        scol_plot_err = []
        ross_plot_err = []
        for band in bands:
            if i == 0:
                scol_plot.append(plot_df['delta_N_{}_scolnic'].median())
                ross_plot.append(plot_df['delta_N_{}_ross'].median())
            if i == 1:
                scol_plot.append(plot_df['delta_N_{}_scolnic'].mean())
                ross_plot.append(plot_df['delta_N_{}_ross'].mean())
                scol_plot_err.append(plot_df['delta_N_{}_scolnic'].std())
                ross_plot_err.append(plot_df['delta_N_{}_ross'].std())

        fig = plt.figure()
        if i == 0:
            plt.scatter(bands, ross_plot, c='r')
            plt.scatter(bands, scol_plot, c='k', marker='x', s=24)
        if i == 1:
            plt.errorbar(bands, ross_plot, ross_plot_err, c='r')
            plt.errorbar(bands, scol_plot, scol_plot_err, c='k', marker='x', s=24)
        plt.xlabel('Bandfilter removed')
        plt.legend(['Rosswog', 'DES-GW'])
        if i == 0:
            plt.ylabel(r'Median $\Delta N$')
            fig.savefig(output_path + 'median_delta_counts_vs_missing_band.pdf', bbox_inches='tight')

        if i == 1:
            plt.ylabel(r'Mean $\Delta N$')
            fig.savefig(output_path + 'mean_delta_counts_vs_missing_band.pdf', bbox_inches='tight')

        plt.close(fig)
    return

################################################################################

def simple_sorted_cadence_plots(results, output_path):
    fiducial_scolnic_total = 74.5
    fiducial_scolnic_wfd = 69.0
    fiducial_scolnic_ddf = 5.5

    #for detect_type in ['scolnic', 'scolnic_no_coadd', 'scolnic_like', 'scolnic_like_no_coadd']:
    sc_raw_cadence_numbers_total = []
    sc_cadence_total = []
    sc_raw_cadence_numbers_wfd = []
    sc_cadence_wfd = []
    sc_raw_cadence_numbers_ddf = []
    sc_cadence_ddf = []

    rw_raw_cadence_numbers_total = []
    rw_cadence_total = []
    rw_raw_cadence_numbers_wfd = []
    rw_cadence_wfd = []
    rw_raw_cadence_numbers_ddf = []
    rw_cadence_ddf = []
    for key in results.keys():
        for key2 in results[key].keys():
            if key2 == 'scolnic' and re.search('var',key) is None:
                if re.search('minion', key) is not None:
                    sc_cadence_total.append('minion1016_descdith')
                    sc_cadence_wfd.append('minion1016_descdith')
                    sc_cadence_ddf.append('minion1016_descdith')
                else:
                    sc_cadence_total.append(key)
                    sc_cadence_wfd.append(key)
                    sc_cadence_ddf.append(key)
                sc_raw_cadence_numbers_total.append(results[key][key2]['detected_counts']['total']['scolnic'])
                sc_raw_cadence_numbers_wfd.append(results[key][key2]['detected_counts']['wfd']['scolnic'])
                sc_raw_cadence_numbers_ddf.append(results[key][key2]['detected_counts']['ddf']['scolnic'])
            elif key2 == 'rosswog' and re.search('var',key) is None:
                if re.search('minion', key) is not None:
                    rw_cadence_total.append('minion1016_descdith')
                    rw_cadence_wfd.append('minion1016_descdith')
                    rw_cadence_ddf.append('minion1016_descdith')
                else:
                    rw_cadence_total.append(key)
                    rw_cadence_wfd.append(key)
                    rw_cadence_ddf.append(key)
                rw_raw_cadence_numbers_total.append(results[key][key2]['detected_counts']['total']['scolnic'])
                rw_raw_cadence_numbers_wfd.append(results[key][key2]['detected_counts']['wfd']['scolnic'])
                rw_raw_cadence_numbers_ddf.append(results[key][key2]['detected_counts']['ddf']['scolnic'])
            elif re.search('var', key) is not None:
                var
    if len(sc_cadence_total) > 0:
        sc_cadence_numbers_total = deepcopy(sc_raw_cadence_numbers_total)
        sc_cadence_numbers_wfd = deepcopy(sc_raw_cadence_numbers_wfd)
        sc_cadence_numbers_ddf = deepcopy(sc_raw_cadence_numbers_ddf)

    if len(rw_cadence_total) > 0:
        rw_cadence_numbers_total = deepcopy(rw_raw_cadence_numbers_total)
        rw_cadence_numbers_wfd = deepcopy(rw_raw_cadence_numbers_wfd)
        rw_cadence_numbers_ddf = deepcopy(rw_raw_cadence_numbers_ddf)
    # sort based on number detected
    if len(sc_cadence_total) > 0:
        sc_sorted_cadences_total = [x for _, x in sorted(zip(sc_raw_cadence_numbers_total, sc_cadence_total), key=lambda pair: pair[0])]
        sc_raw_cadence_numbers_total.sort()
        sc_sorted_cadence_numbers_total = sc_raw_cadence_numbers_total
        sc_sorted_cadences_wfd = [x for _, x in sorted(zip(sc_raw_cadence_numbers_wfd, sc_cadence_wfd), key=lambda pair: pair[0])]
        sc_raw_cadence_numbers_wfd.sort()
        sc_sorted_cadence_numbers_wfd = sc_raw_cadence_numbers_wfd
        sc_sorted_cadences_ddf = [x for _, x in sorted(zip(sc_raw_cadence_numbers_ddf, sc_cadence_ddf), key=lambda pair: pair[0])]
        sc_raw_cadence_numbers_ddf.sort()
        sc_sorted_cadence_numbers_ddf = sc_raw_cadence_numbers_ddf

    if len(rw_cadence_total) > 0:
        rw_sorted_cadences_total = [x for _, x in sorted(zip(rw_raw_cadence_numbers_total, rw_cadence_total), key=lambda pair: pair[0])]
        rw_raw_cadence_numbers_total.sort()
        rw_sorted_cadence_numbers_total = rw_raw_cadence_numbers_total
        rw_sorted_cadences_wfd = [x for _, x in sorted(zip(rw_raw_cadence_numbers_wfd, rw_cadence_wfd), key=lambda pair: pair[0])]
        rw_raw_cadence_numbers_wfd.sort()
        rw_sorted_cadence_numbers_wfd = rw_raw_cadence_numbers_wfd
        rw_sorted_cadences_ddf = [x for _, x in sorted(zip(rw_raw_cadence_numbers_ddf, rw_cadence_ddf), key=lambda pair: pair[0])]
        rw_raw_cadence_numbers_ddf.sort()
        rw_sorted_cadence_numbers_ddf = rw_raw_cadence_numbers_ddf

    fig = plt.figure()
    plt.errorbar(sc_sorted_cadences_total, sc_sorted_cadence_numbers_total, yerr=np.sqrt(sc_sorted_cadence_numbers_total), fmt='o', capsize=6)
    plt.axhline(fiducial_scolnic_total, color='r')
    plt.xticks(rotation='vertical')
    plt.title('Detections vs. Cadence (DES-GW model)')
    plt.ylabel('Number of Detections')
    plt.legend(['Value from Scolnic et. al 2017', 'Number Detected per Cadence'])
    plt.savefig(output_path + 'Sorted_total_scolnic_counts_by_cadence.pdf', bbox_inches='tight')
    plt.close(fig)


    fig = plt.figure()
    plt.errorbar(sc_sorted_cadences_total, sc_sorted_cadence_numbers_total, yerr=np.sqrt(sc_sorted_cadence_numbers_total), fmt='o', capsize=6, color='b')
    plt.errorbar(rw_sorted_cadences_total, rw_sorted_cadence_numbers_total, yerr=np.sqrt(rw_sorted_cadence_numbers_total), fmt='o', capsize=6, color='k')
    plt.axhline(fiducial_scolnic_total, color='r')
    plt.xticks(rotation='vertical', fontsize=int(13))
    plt.ylabel('Number of Detections', fontsize=int(13))
    plt.legend(['Value from Scolnic et. al 2017', 'DES-GW', 'SAEE'], fontsize=int(13))
    plt.savefig(output_path + 'total_detection_counts_by_cadence.pdf', bbox_inches='tight')
    plt.close(fig)


    fig = plt.figure()
    plt.errorbar(rw_sorted_cadences_total, rw_sorted_cadence_numbers_total, yerr=np.sqrt(rw_sorted_cadence_numbers_total), fmt='o', capsize=6)
    plt.axhline(fiducial_scolnic_total, color='r')
    plt.xticks(rotation='vertical')
    plt.title('Detections vs. Cadence (Rosswog model)')
    plt.ylabel('Number of Detections')
    plt.legend(['Value from Scolnic et. al 2017', 'Number Detected per Cadence'], loc=4)
    plt.savefig(output_path + 'Sorted_total_rosswog_counts_by_cadence.pdf', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.errorbar(sc_sorted_cadences_wfd, sc_sorted_cadence_numbers_wfd, yerr=np.sqrt(sc_sorted_cadence_numbers_wfd), fmt='o', capsize=6, color='b')
    plt.errorbar(rw_sorted_cadences_wfd, rw_sorted_cadence_numbers_wfd, yerr=np.sqrt(rw_sorted_cadence_numbers_wfd), fmt='o', capsize=6, color='k')
    plt.axhline(fiducial_scolnic_wfd, color='r')
    plt.xticks(rotation='vertical', fontsize=int(13))
    plt.ylabel('Number of Detections', fontsize=int(13))
    plt.legend(['Value from Scolnic et. al 2017', 'DES-GW', 'SAEE'], loc=2, fontsize=int(13))
    plt.savefig(output_path + 'wfd_detection_counts_by_cadence.pdf', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.errorbar(sc_sorted_cadences_ddf, sc_sorted_cadence_numbers_ddf, yerr=np.sqrt(sc_sorted_cadence_numbers_ddf), fmt='o', capsize=6, color='b')
    plt.errorbar(rw_sorted_cadences_ddf, rw_sorted_cadence_numbers_ddf, yerr=np.sqrt(rw_sorted_cadence_numbers_ddf), fmt='o', capsize=6, color='k')
    plt.axhline(fiducial_scolnic_ddf, color='r')
    plt.xticks(rotation='vertical', fontsize=int(13))
    plt.ylabel('Number of Detections', fontsize=int(13))
    plt.legend(['Value from Scolnic et. al 2017', 'DES-GW', 'SAEE'], loc=2, fontsize=int(13))
    plt.savefig(output_path + 'ddf_detection_counts_by_cadence.pdf', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.errorbar(sc_sorted_cadences_wfd, sc_sorted_cadence_numbers_wfd, yerr=np.sqrt(sc_sorted_cadence_numbers_wfd), fmt='o', capsize=6)
    plt.axhline(fiducial_scolnic_wfd, color='r')
    plt.xticks(rotation='vertical')
    plt.title('Detections vs. Cadence WFD (DES-GW model)')
    plt.ylabel('Number of Detections')
    plt.legend(['Value from Scolnic et. al 2017', 'Number Detected per Cadence'])
    plt.savefig(output_path + 'Sorted_wfd_scolnic_counts_by_cadence.pdf', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.errorbar(rw_sorted_cadences_wfd, rw_sorted_cadence_numbers_wfd, yerr=np.sqrt(rw_sorted_cadence_numbers_wfd), fmt='o', capsize=6)
    plt.axhline(fiducial_scolnic_wfd, color='r')
    plt.xticks(rotation='vertical')
    plt.title('Detections vs. Cadence WFD (Rosswog model)')
    plt.ylabel('Number of Detections')
    plt.legend(['Value from Scolnic et. al 2017', 'Number Detected per Cadence'], loc=4)
    plt.savefig(output_path + 'Sorted_wfd_rosswog_counts_by_cadence.pdf', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.errorbar(sc_sorted_cadences_ddf, sc_sorted_cadence_numbers_ddf, yerr=np.sqrt(sc_sorted_cadence_numbers_ddf), fmt='o', capsize=6)
    plt.axhline(fiducial_scolnic_ddf, color='r')
    plt.xticks(rotation='vertical')
    plt.title('Detections vs. Cadence DDF (DES-GW model)')
    plt.ylabel('Number of Detections')
    plt.legend(['Value from Scolnic et. al 2017', 'Number Detected per Cadence'])
    plt.savefig(output_path + 'Sorted_ddf_scolnic_counts_by_cadence.pdf', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    plt.errorbar(rw_sorted_cadences_ddf, rw_sorted_cadence_numbers_ddf, yerr=np.sqrt(rw_sorted_cadence_numbers_ddf), fmt='o', capsize=6)
    plt.axhline(fiducial_scolnic_ddf, color='r')
    plt.xticks(rotation='vertical')
    plt.title('Detections vs. Cadence DDF(Rosswog model)')
    plt.ylabel('Number of Detections')
    plt.legend(['Value from Scolnic et. al 2017', 'Number Detected per Cadence'], loc=4)
    plt.savefig(output_path + 'Sorted_ddf_rosswog_counts_by_cadence.pdf', bbox_inches='tight')
    plt.close(fig)


    fig = plt.figure()
    plt.errorbar(rw_sorted_cadences_wfd, rw_sorted_cadence_numbers_wfd, yerr=np.sqrt(rw_sorted_cadence_numbers_wfd), fmt='o', capsize=6)
    plt.errorbar(rw_sorted_cadences_ddf, rw_sorted_cadence_numbers_ddf, yerr=np.sqrt(rw_sorted_cadence_numbers_ddf), fmt='o', capsize=6)
    plt.axhline(fiducial_scolnic_ddf, color='r')
    plt.axhline(fiducial_scolnic_wfd, color='r')
    plt.xticks(rotation='vertical')
    plt.title('Detections vs. Cadence (Rosswog model)')
    plt.ylabel('Number of Detections')
    plt.legend(['DDF Value from Scolnic et. al 2017', 'WFD Value from Scolnic et. al 2017', 'WFD', 'DDF'], loc=4)
    plt.savefig(output_path + 'split_rosswog_counts_by_cadence.pdf', bbox_inches='tight')
    plt.close(fig)


    fig = plt.figure()
    plt.errorbar(sc_sorted_cadences_wfd, sc_sorted_cadence_numbers_wfd, yerr=np.sqrt(sc_sorted_cadence_numbers_wfd), fmt='o', capsize=6)
    plt.errorbar(sc_sorted_cadences_ddf, sc_sorted_cadence_numbers_ddf, yerr=np.sqrt(sc_sorted_cadence_numbers_ddf), fmt='o', capsize=6)
    plt.axhline(fiducial_scolnic_ddf, color='r')
    plt.axhline(fiducial_scolnic_wfd, color='r')
    plt.xticks(rotation='vertical')
    plt.title('Detections vs. Cadence (DES-GW model)')
    plt.ylabel('Number of Detections')
    plt.legend(['DDF Value from Scolnic et. al 2017', 'WFD Value from Scolnic et. al 2017', 'WFD', 'DDF'], loc=4)
    plt.savefig(output_path + 'split_desgw_counts_by_cadence.pdf', bbox_inches='tight')
    plt.close(fig)

    return


def hist_params(df, col, nbins):
    fig = plt.figure()
    plt.hist(df[col], bins=nbins)
    plt.xlabel('{}'.format(col))
    plt.ylabel('Counts')
    return fig
