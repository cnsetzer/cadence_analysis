import os
import re
import numpy as np
import pandas as pd
import matplotlib
from astrotog.functions import scolnic_detections as scd
# matplotlib.use('Agg')
import seaborn as sns
# I personally like this style.
sns.set_style('whitegrid')
# Easy to change context from `talk`, `notebook`, `poster`, `paper`.
sns.set_context('paper')
import matplotlib.pyplot as plt
from copy import copy, deepcopy
pd.set_option('display.max_columns', 500)


def determine_ddf_detections(ddf_properties, cadence_results):
    field_rad = np.deg2rad(3.5/2.0)
    for cadence in list(ddf_properties.index):
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
                for index, row in cad_transient_params.iterrows():
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
            cadence_results[cadence][model_key]['subset_detections']['total'] = detections1
            cadence_results[cadence][model_key]['subset_detections']['ddf'] = detections[detections['transient_id'].isin(ids_in_ddf)]
            cadence_results[cadence][model_key]['subset_detections']['wfd'] = detections[~detections['transient_id'].isin(ids_in_ddf)]
            cadence_results[cadence][model_key]['subset_detections']['total'] = detections2
            cadence_results[cadence][model_key]['subset_detections']['ddf'] = detections[detections['transient_id'].isin(ids_in_ddf)]
            cadence_results[cadence][model_key]['subset_detections']['wfd'] = detections[~detections['transient_id'].isin(ids_in_ddf)]
            cadence_results[cadence][model_key]['subset_detections']['total'] = detections3
            cadence_results[cadence][model_key]['subset_detections']['ddf'] = detections[detections['transient_id'].isin(ids_in_ddf)]
            cadence_results[cadence][model_key]['subset_detections']['wfd'] = detections[~detections['transient_id'].isin(ids_in_ddf)]
            cadence_results[cadence][model_key]['subset_detections']['total'] = detections4
            cadence_results[cadence][model_key]['subset_detections']['ddf'] = detections[detections['transient_id'].isin(ids_in_ddf)]
            cadence_results[cadence][model_key]['subset_detections']['wfd'] = detections[~detections['transient_id'].isin(ids_in_ddf)]
    return cadence_results


def process_counts(results):
    for cadence in results.keys():
        for model in results[cadence].keys():
            results[cadence][model]['detected_counts'] = {}
            results[cadence][model]['detected_counts']['total'] = len(results[cadence][model]['data']['scolnic_detections']['transient_id'].unique())
            results[cadence][model]['detected_counts']['wfd'] = len(results[cadence][model]['subset_detections']['wfd']['transient_id'].unique())
            results[cadence][model]['detected_counts']['ddf'] = len(results[cadence][model]['subset_detections']['ddf']['transient_id'].unique())

    return results


def plotting_dataframe(results, wfd_props, ddf_props):
    results_df = pd.DataFrame(index= wfd_props.index, columns=['wfd_ross','ddf_ross', 'total_ross','wfd_scolnic','ddf_scolnic', 'total_scolnic'])
    results_df = results_df.join(wfd_props)
    results_df = results_df.join(ddf_props)
    for cadence in results.keys():
        for model in results[cadence].keys():
            if model == 'rosswog':
                results_df.at[cadence, 'wfd_ross'] = results[cadence][model]['detected_counts']['wfd']
                results_df.at[cadence, 'ddf_ross'] = results[cadence][model]['detected_counts']['ddf']
                results_df.at[cadence, 'total_ross'] = results[cadence][model]['detected_counts']['total']
            elif model == 'scolnic':
                results_df.at[cadence, 'wfd_scolnic'] = results[cadence][model]['detected_counts']['wfd']
                results_df.at[cadence, 'ddf_scolnic'] = results[cadence][model]['detected_counts']['ddf']
                results_df.at[cadence, 'total_scolnic'] = results[cadence][model]['detected_counts']['total']
    return results_df


def plot_trends(results_df, output_path):
    for subset in ['wfd','ddf','total']:
        for property in list(results_df.columns):
            if (re.search('ross', property) is None) and (re.search('scolnic', property) is None):
                if (re.search('wfd',property) is not None and subset == 'wfd') or (re.search('ddf',property) is not None and subset == 'ddf' and re.search('dec|ra',property) is None) or (re.search('wfd',property) is None and subset == 'total' and re.search('ddf',property) is None):
                    fig = plt.figure()
                    plt.scatter(results_df[property], results_df['{}_ross'.format(subset)], c='r')
                    plt.scatter(results_df[property], results_df['{}_scolnic'.format(subset)], c='k', marker='x')
                    plt.xlabel('{}'.format(property))
                    plt.ylabel('Number of Detections')
                    plt.legend(['Rosswog', 'DECAM'])
                    fig.savefig(output_path + '{0}_counts_vs_{1}.pdf'.format(subset, property), bbox_inches='tight')
                    plt.close(fig)
    return


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
                                fig = plt.figure()

                                plt.scatter(results_df[property1], results_df[property2], c=results_df['{}_ross'.format(subset)], cmap='inferno', s=50)
                                plt.scatter(results_df[property1], results_df[property2], c=results_df['{}_scolnic'.format(subset)], cmap='inferno', marker='x', s=55)
                                plt.xlabel('{}'.format(property1))
                                plt.ylabel('{}'.format(property2))
                                plt.colorbar(label='Number of Detections')
                                plt.legend(['Rosswog', 'DECAM'])
                                fig.savefig(output_path + '{0}_counts_vs_{1}_and_{2}.pdf'.format(subset, property1, property2), bbox_inches='tight')
                                plt.close(fig)
    return


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
                if re.search('baseline',name_split[2]) is None:
                    name_split2 = re.split("(\d+)", name_split[2])
                    name = name_split2[0] + '_' + name_split2[1]
                else:
                    name = name_split[2]
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
    return results

def param_subset(param_df, ddf_subset):
    subset_df = pd.DataFrame(index=param_df.index, columns=['subset'])
    in_ddf = list(ddf_subset['transient_id'].unique())
    for id in list(param_df['transient_id']):
        if id in in_ddf:
            subset_df.at[id,'subset'] = 'ddf'
        else:
            subset_df.at[id,'subset'] = 'wfd'

    param_df = param_df.join(subset_df)
    return param_df


def redshift_distribution(param_df):
    z_min = 0.0
    z_max = 0.5
    bin_size = 0.025
    n_bins = int(round((z_max-z_min)/bin_size))
    all_zs = list(param_df['true_redshift'])
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
        ddf_detect_zs = list(param_df.query('subset == \'{0}\' & detected == {1}'.format('ddf', True))['true_redshift'])
        ddf_max_depth_detect = list(param_df[param_df['true_redshift'] <= max(ddf_detect_zs)]['true_redshift'])

    total_eff = (len(wfd_detect_zs.extend(ddf_detect_zs))/len(all_zs))*100
    max_depth_eff = (len(wfd_detect_zs.extend(ddf_detect_zs))/len(wfd_max_depth_detect.extend(ddf_max_depth_detect)))*100

    print('The redshift range of all sources is {0:.4f} to {1:.4f}.'.format(min(all_zs), max(all_zs)))
    print('The redshift range of the detected sources is {0:.4f} to {1:.4f}.'.format(min(wfd_detect_zs.extend(ddf_detect_zs)), max(wfd_detect_zs.extend(ddf_detect_zs))))
    print('There are {0} detected transients out of {1}, which is an efficiency of {2:2.2f}%  of the total simulated number.'.format(len(wfd_detect_zs.extend(ddf_detect_zs)), len(all_zs), total_eff))
    print('However, this is an efficiency of {0:2.2f}%  of the total that occur within the range that was detected by {1}.'.format(max_depth_eff, 'LSST'))
    # Create the histogram'
    N_z_dist_fig = plt.figure()
    plt.hist(x=all_zs, bins=n_bins, range=(z_min, z_max), histtype='step', color='red', label='All Sources', linewidth=3.0)
    plt.hist(x=[wfd_detect_zs,ddf_detect_zs], bins=n_bins, range=(z_min, z_max), histtype='stepfilled', alpha=0.3, label='Detected Sources', stacked=True)
    # plt.tick_params(which='both', length=10, width=1.5)
    plt.yscale('log')
    plt.legend(loc=2)
    plt.xlabel('z')
    plt.ylabel(r'$N(z)$')
    plt.title('Redshift Distribution ({0:.3f} bins, {1} WFD, {2} DDF)'.format(bin_size,len(wfd_detect_zs), len(ddf_detect_zs)))
    return N_z_dist_fig


def band_delta_N(plot_df, results):
    results_df = pd.DataFrame(index= plot_df.index, columns=['delta_N_u_ross', 'delta_N_g_ross', 'delta_N_r_ross', 'delta_N_i_ross', 'delta_N_z_ross', 'delta_N_y_ross', 'delta_N_u_scolnic', 'delta_N_g_scolnic', 'delta_N_r_scolnic', 'delta_N_i_scolnic', 'delta_N_z_scolnic', 'delta_N_y_scolnic'])
    results_df = plot_df.join(results_df)
    for cadence in results.keys():
        for model in results[cadence].keys():
            if model == 'rosswog':
                for band in ['u','g','r','i','z','y']:
                    detect_bandless = results[cadence][model]['data']['scolnic_detections'].query('bandfilter != \'{}\''.format(band))
                    new_N = len(scd(results[cadence][model]['data']['parameters'],detect_bandless,results[cadence][model]['data']['other_observations'])['transient_id'].unique())
                    results_df.at[cadence, 'delta_N_{}_ross'] = new_N - results_df.at[cadence, 'total_ross']
            elif model == 'scolnic':
                for band in ['u','g','r','i','z','y']:
                    detect_bandless = results[cadence][model]['data']['scolnic_detections'].query('bandfilter != \'{}\''.format(band))
                    new_N = len(scd(results[cadence][model]['data']['parameters'],detect_bandless,results[cadence][model]['data']['other_observations'])['transient_id'].unique())
                    results_df.at[cadence, 'delta_N_{}_scolnic'] = new_N - results_df.at[cadence, 'total_ross']
    return results_df


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
        plt.legend(['Rosswog', 'DECAM'])
        if i == 0:
            plt.ylabel(r'Median $\Delta N$')
            fig.savefig(output_path + 'median_delta_counts_vs_missing_band.pdf', bbox_inches='tight')

        if i == 1:
            plt.ylabel(r'Mean $\Delta N$')
            fig.savefig(output_path + 'mean_delta_counts_vs_missing_band.pdf', bbox_inches='tight')

        plt.close(fig)
    return


def simple_sorted_cadence_plots(results):
    fiducial_scolnic = 74.5


    return

if __name__ == "__main__":

    sim_results_path = []

    output_path = '../results_plots/'
    prop_path = '/Users/cnsetzer/Documents/LSST/cadence_analysis/cadence_analysis/cadence_data/'
    sim_results_path.append('/Users/cnsetzer/Documents/LSST/astrotog_output/rosswog_results/binomial_runs/')
    sim_results_path.append('/Users/cnsetzer/Documents/LSST/astrotog_output/scolnic_results/')

    # Get the properties for the different cadence partitions
    ddf_props = pd.read_csv(prop_path + 'ddf_properties.csv', index_col=0)
    wfd_props = pd.read_csv(prop_path + 'wfd_properties.csv', index_col=0)
    cadence_props = pd.read_csv(prop_path + 'cadence_information.csv', index_col=0, sep=';')

    wfd_props = wfd_props.join(cadence_props)

    wfd_props.dropna(inplace=True)

    results = get_cadence_results(sim_results_path)
    results = determine_ddf_detections(ddf_props, results)
    results = process_counts(results)
    results_df = plotting_dataframe(results, wfd_props, ddf_props)
    results_df.dropna(inplace=True)
    plot_trends(results_df, output_path)
    plot_trends_2D(results_df, output_path)

    scolnic_max = results_df['total_scolnic'].astype(np.float64).idxmax()
    rosswog_max = results_df['total_ross'].astype(np.float64).idxmax()
    scolnic_max_param_df = param_subset(results[scolnic_max]['scolnic']['data']['parameters'], results[scolnic_max]['scolnic']['subset_detections']['ddf'])
    ross_max_param_df = param_subset(results[rosswog_max]['rosswog']['data']['parameters'], results[rosswog_max]['rosswog']['subset_detections']['ddf'])

    scolnic_max_nz = redshift_distribution(scolnic_max_param_df)
    scolnic_max_nz.savefig('scolnic_nz_max_{}.pdf'.format(scolnic_max),bbox_inches='tight')
    plt.close(scolnic_max_nz)
    rosswog_max_nz = redshift_distribution(ross_max_param_df)
    rosswog_max_nz.savefig('rosswog_nz_max_{}.pdf'.format(rosswog_max),bbox_inches='tight')
    plt.close(rosswog_max_nz)

    scolnic_min = results_df['total_scolnic'].astype(np.float64).idxmin()
    rosswog_min = results_df['total_ross'].astype(np.float64).idxmin()
    scolnic_min_param_df = param_subset(results[scolnic_min]['scolnic']['data']['parameters'], results[scolnic_min]['scolnic']['subset_detections']['ddf'])
    ross_min_param_df = param_subset(results[rosswog_min]['rosswog']['data']['parameters'], results[rosswog_min]['rosswog']['subset_detections']['ddf'])

    scolnic_min_nz = redshift_distribution(scolnic_min_param_df)
    scolnic_min_nz.savefig('scolnic_nz_min_{}.pdf'.format(scolnic_min),bbox_inches='tight')
    plt.close(scolnic_min_nz)
    rosswog_min_nz = redshift_distribution(ross_min_param_df)
    rosswog_min_nz.savefig('rosswog_nz_min_{}.pdf'.format(rosswog_min),bbox_inches='tight')
    plt.close(rosswog_min_nz)

    scolnic_base = 'kraken_2026'
    rosswog_base = 'kraken_2026'
    scolnic_base_param_df = param_subset(results[scolnic_base]['scolnic']['data']['parameters'], results[scolnic_base]['scolnic']['subset_detections']['ddf'])
    ross_base_param_df = param_subset(results[rosswog_base]['rosswog']['data']['parameters'], results[rosswog_base]['rosswog']['subset_detections']['ddf'])

    scolnic_base_nz = redshift_distribution(scolnic_base_param_df)
    scolnic_base_nz.savefig('scolnic_nz_base_{}.pdf'.format(scolnic_base),bbox_inches='tight')
    plt.close(scolnic_base_nz)
    rosswog_base_nz = redshift_distribution(ross_base_param_df)
    rosswog_base_nz.savefig('rosswog_nz_base_{}.pdf'.format(rosswog_base),bbox_inches='tight')
    plt.close(rosswog_base_nz)

    plot_df = band_delta_N(results_df, results)
    plot_delta_n(plot_df)
