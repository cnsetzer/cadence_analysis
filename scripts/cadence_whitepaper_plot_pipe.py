import os
import re
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from mpi4py import MPI
import multiprocessing as mp
from astrotog.functions import scolnic_detections as scd
from astrotog.functions import scolnic_like_detections as scld
import cadanalysis.functions as ca
# matplotlib.use('Agg')
import seaborn as sns
# I personally like this style.
sns.set_style('whitegrid')
# Easy to change context from `talk`, `notebook`, `poster`, `paper`.
sns.set_context('paper')
pd.set_option('display.max_columns', 500)


if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    sim_results_path = []

    # output_path = '/Users/cnsetzer/Documents/LSST/cadence_analysis/whitepaper_writeup/figures/'
    output_path = '/Users/cnsetzer/Documents/LSST/whitepaper_cadence_analysis/cadence_analysis/results_plots/'
    prop_path = '/Users/cnsetzer/Documents/LSST/whitepaper_cadence_analysis/cadence_analysis/cadence_data/'
    sim_results_path.append('/Users/cnsetzer/Documents/LSST/astrotog_output/rosswog_results/binomial_runs/')
    sim_results_path.append('/Users/cnsetzer/Documents/LSST/astrotog_output/scolnic_results/')

    if rank == 0:
        print('The number of MPI processes is: {}'.format(size))
    # Get the properties for the different cadence partitions
    ddf_props = pd.read_csv(prop_path + 'ddf_properties.csv', index_col=0, sep=';')
    wfd_props = pd.read_csv(prop_path + 'wfd_properties.csv', index_col=0, sep=';')
    cadence_props = pd.read_csv(prop_path + 'cadence_information.csv', index_col=0, sep=';')
    wfd_props = wfd_props.join(cadence_props)
    wfd_props.dropna(inplace=True)

    results = ca.get_cadence_results(sim_results_path)
    print('Done importing the cadence results.')
    results = ca.determine_ddf_detections(ddf_props, results)
    print('Done determining the detections in DDF.')
    results = ca.process_counts(results)
    print('Done getting counts.')
    results_df = ca.plotting_dataframe(results, wfd_props, ddf_props)
    print('Done constructing dataframe for plotting.')
    results_df.dropna(inplace=True)

    # ca.plot_trends(results_df, output_path)
    # ca.plot_trends_2D(results_df, output_path)
    ca.simple_sorted_cadence_plots(results, output_path)
    print('Done with basic plots.')

    # scolnic_max = results_df['total_scolnic'].astype(np.float64).idxmax()
    # rosswog_max = results_df['total_ross'].astype(np.float64).idxmax()
    # scolnic_max_param_df = ca.param_subset(results[scolnic_max]['scolnic']['data']['parameters'], results[scolnic_max]['scolnic']['subset_detections']['ddf']['scolnic'])
    # ross_max_param_df = ca.param_subset(results[rosswog_max]['rosswog']['data']['parameters'], results[rosswog_max]['rosswog']['subset_detections']['ddf']['scolnic'])
    #
    # scolnic_max_nz = ca.redshift_distribution(scolnic_max_param_df)
    # scolnic_max_nz.savefig(output_path + 'scolnic_nz_max_{}.pdf'.format(scolnic_max),bbox_inches='tight')
    # plt.close(scolnic_max_nz)
    # rosswog_max_nz = ca.redshift_distribution(ross_max_param_df)
    # rosswog_max_nz.savefig(output_path + 'rosswog_nz_max_{}.pdf'.format(rosswog_max),bbox_inches='tight')
    # plt.close(rosswog_max_nz)
    #
    # scolnic_min = results_df['total_scolnic'].astype(np.float64).idxmin()
    # rosswog_min = results_df['total_ross'].astype(np.float64).idxmin()
    # scolnic_min_param_df = ca.param_subset(results[scolnic_min]['scolnic']['data']['parameters'], results[scolnic_min]['scolnic']['subset_detections']['ddf']['scolnic'])
    # ross_min_param_df = ca.param_subset(results[rosswog_min]['rosswog']['data']['parameters'], results[rosswog_min]['rosswog']['subset_detections']['ddf']['scolnic'])
    #
    # scolnic_min_nz = ca.redshift_distribution(scolnic_min_param_df)
    # scolnic_min_nz.savefig(output_path + 'scolnic_nz_min_{}.pdf'.format(scolnic_min),bbox_inches='tight')
    # plt.close(scolnic_min_nz)
    # rosswog_min_nz = ca.redshift_distribution(ross_min_param_df)
    # rosswog_min_nz.savefig(output_path + 'rosswog_nz_min_{}.pdf'.format(rosswog_min),bbox_inches='tight')
    # plt.close(rosswog_min_nz)
    #
    scolnic_base = 'kraken_2026'
    rosswog_base = 'kraken_2026'
    scolnic_base_param_df = ca.param_subset(results[scolnic_base]['scolnic']['data']['parameters'], results[scolnic_base]['scolnic']['subset_detections']['ddf']['scolnic'])
    ross_base_param_df = ca.param_subset(results[rosswog_base]['rosswog']['data']['parameters'], results[rosswog_base]['rosswog']['subset_detections']['ddf']['scolnic'])
    # 
    # scolnic_base_nz = ca.redshift_distribution(scolnic_base_param_df)
    # scolnic_base_nz.savefig(output_path + 'scolnic_nz_base_{}.pdf'.format(scolnic_base),bbox_inches='tight')
    # plt.close(scolnic_base_nz)
    # rosswog_base_nz =ca.redshift_distribution(ross_base_param_df)
    # rosswog_base_nz.savefig(output_path + 'rosswog_nz_base_{}.pdf'.format(rosswog_base),bbox_inches='tight')
    # plt.close(rosswog_base_nz)

    scolnic_base_nz = ca.overlay_redshift_distribution(scolnic_base_param_df, ross_base_param_df)
    scolnic_base_nz.savefig(output_path + 'both_nz_base_{}.pdf'.format(scolnic_base),bbox_inches='tight')
    plt.close(scolnic_base_nz)
    print('Done with redshift distribution plots.')

    # rkey = 'rosswog'
    # nbins = 8
    # param_plot_list = ['m_ej', 'v_ej', 'kappa', 'explosion_time']
    # for cadence in results.keys():
    #     params = results[cadence][rkey]['data']['parameters']
    #     detections = results[cadence][rkey]['data']['scolnic_detections']
    #     sub_params = params[params['transient_id'].isin(detections['transient_id'].unique())]
    #
    #     for par in param_plot_list:
    #         hist_fig = ca.hist_params(sub_params, par, nbins)
    #         hist_fig.savefig(output_path +'detect_hist_{0}_{1}_{2}.pdf'.format(cadence,rkey,par),bbox_inches='tight')
    #         plt.close(hist_fig)
