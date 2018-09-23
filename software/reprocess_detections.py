import os
import pandas as pd
import multiprocessing as mp
from astrotog.functions import scolnic_detections as scd
from astrotog.functions import scolnic_like_detections as scld


def process_detections(path, directory):
    if directory == '.DS_Store':
        return
    else:
        file1_exist = os.path.isfile(path + directory + '/scolnic_detections_no_coadd.csv')
        file2_exist = os.path.isfile(path + directory + '/scolnic_like_detections_no_coadd.csv')
        file3_exist = os.path.isfile(path + directory + '/scolnic_like_detections.csv')
        if file1_exist and file2_exist and file3_exist:
            print('{} is done reprocessing for detections.'.format(directory))
            return
        else:
            parameters = pd.read_csv(path + directory + '/modified_parameters.csv', index_col=0)
            single_observations = pd.read_csv(path + directory + '/observations.csv', index_col=0)
            other_observations = pd.read_csv(path + directory + '/other_observations.csv', index_col=0)
            if not file1_exist:
                single_detections = scd(parameters, single_observations, other_observations)
                single_detections.to_csv(path + directory + '/scolnic_detections_no_coadd.csv')
                single_detections = None
            if not file2_exist:
                single_like_detections = scld(parameters, single_observations, other_observations)
                single_like_detections.to_csv(path + directory + '/scolnic_like_detections_no_coadd.csv')
                single_observations = None
                single_like_detections = None
            if not file3_exist:
                coadded_observations = pd.read_csv(path + directory + '/coadded_observations.csv', index_col=0)
                coadded_like_detections = scld(parameters, coadded_observations, other_observations)
                coadded_like_detections.to_csv(path + directory + '/scolnic_like_detections.csv')
                coadded_observations = None
                parameters = None
                other_observations = None
        print('{} is done reprocessing for detections.'.format(directory))
    return


if __name__ == '__main__':

    sim_results_paths = []
    # sim_results_paths.append('/Users/cnsetzer/Documents/LSST/astrotog_output/rosswog_results/binomial_runs/')
    # sim_results_paths.append('/Users/cnsetzer/Documents/LSST/astrotog_output/scolnic_results/')
    sim_results_paths.append('share/data1/csetzer/lsst_kne_sims_outputs/')

    for path in sim_results_paths:
        output_dirs = os.listdir(path)
        map_inputs = list(zip(repeat(path), output_dirs))
        p = mp.Pool()
        p.starmap(process_detections, map_inputs)
        p.close()
