import os
import re
import time
import datetime
import numpy as np
import pandas as pd
from mpi4py import MPI
from astrotog.functions import scolnic_detections as scd
from astrotog.functions import scolnic_like_detections as scld
import cadanalysis.functions as ca
pd.set_option('display.max_columns', 500)

if __name__ == "__main__":
    t0 = time.time()
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    sim_results_path = []

    output_path = '/home/csetzer/LSST/whitepaper/detect_delta_N/'
    sim_results_path.append('/share/data1/csetzer/lsst_kne_sims_outputs/')

    # output_path = '/Users/cnsetzer/Documents/LSST/astrotog_output/'
    # sim_results_path.append('/Users/cnsetzer/Documents/LSST/astrotog_output/scolnic_results/')

    df1 = pd.DataFrame(columns=['delta_N_u_rosswog', 'delta_N_g_rosswog', 'delta_N_r_rosswog', 'delta_N_i_rosswog', 'delta_N_z_rosswog', 'delta_N_y_rosswog', 'delta_N_u_scolnic', 'delta_N_g_scolnic', 'delta_N_r_scolnic', 'delta_N_i_scolnic', 'delta_N_z_scolnic', 'delta_N_y_scolnic'])
    df2 = pd.DataFrame(columns=['delta_N_u_rosswog', 'delta_N_g_rosswog', 'delta_N_r_rosswog', 'delta_N_i_rosswog', 'delta_N_z_rosswog', 'delta_N_y_rosswog', 'delta_N_u_scolnic', 'delta_N_g_scolnic', 'delta_N_r_scolnic', 'delta_N_i_scolnic', 'delta_N_z_scolnic', 'delta_N_y_scolnic'])
    df3 = pd.DataFrame(columns=['delta_N_u_rosswog', 'delta_N_g_rosswog', 'delta_N_r_rosswog', 'delta_N_i_rosswog', 'delta_N_z_rosswog', 'delta_N_y_rosswog', 'delta_N_u_scolnic', 'delta_N_g_scolnic', 'delta_N_r_scolnic', 'delta_N_i_scolnic', 'delta_N_z_scolnic', 'delta_N_y_scolnic'])
    df4 = pd.DataFrame(columns=['delta_N_u_rosswog', 'delta_N_g_rosswog', 'delta_N_r_rosswog', 'delta_N_i_rosswog', 'delta_N_z_rosswog', 'delta_N_y_rosswog', 'delta_N_u_scolnic', 'delta_N_g_scolnic', 'delta_N_r_scolnic', 'delta_N_i_scolnic', 'delta_N_z_scolnic', 'delta_N_y_scolnic'])

    results = None
    cadence_keys = None
    model_keys = None
    if rank == 0:
        print('The number of MPI processes is: {}'.format(size))
        results = ca.get_cadence_results(sim_results_path)
        print('Done importing the cadence results.')
        cadence_keys = list(results.keys())
        model_keys = list(results[cadence_keys[0]].keys())

    if size > 1:
        cadence_keys = comm.bcast(cadence_keys, root=0)
        model_keys = comm.bcast(model_keys, root=0)

    for cadence in cadence_keys:
        if rank == 0:
            print(cadence)
        else:
            results = {}
            results[cadence] = {}

        for model in model_keys:
            results_cadence_model = None
            if rank != 0:
                results[cadence][model] = None
            if size > 1:
                comm.barrier()
                results_cadence_model = comm.bcast(results[cadence][model], root=0)

            other_obs = results_cadence_model['data']['other_observations']
            params = results_cadence_model['data']['parameters']
            detections1 = results_cadence_model['data']['scolnic_detections']
            detections2 = results_cadence_model['data']['scolnic_detections_no_coadd']
            detections3 = results_cadence_model['data']['scolnic_like_detections']
            detections4 = results_cadence_model['data']['scolnic_like_detections_no_coadd']

            num_detected1 = len(detections1['transient_id'].unique())
            num_detected2 = len(detections2['transient_id'].unique())
            num_detected3 = len(detections3['transient_id'].unique())
            num_detected4 = len(detections4['transient_id'].unique())

            if rank == 0:
                id_list1 = np.asarray(detections1['transient_id'].unique(), dtype='i4')
                id_list2 = np.asarray(detections2['transient_id'].unique(), dtype='i4')
                id_list3 = np.asarray(detections3['transient_id'].unique(), dtype='i4')
                id_list4 = np.asarray(detections4['transient_id'].unique(), dtype='i4')
            else:
                id_list1 = None
                id_list2 = None
                id_list3 = None
                id_list4 = None

            num_trans_pprocess1 = int(np.ceil(num_detected1/size))
            num_trans_pprocess2 = int(np.ceil(num_detected2/size))
            num_trans_pprocess3 = int(np.ceil(num_detected3/size))
            num_trans_pprocess4 = int(np.ceil(num_detected4/size))

            receive_array1 = np.zeros(num_trans_pprocess1, dtype='i4')
            receive_array2 = np.zeros(num_trans_pprocess2, dtype='i4')
            receive_array3 = np.zeros(num_trans_pprocess3, dtype='i4')
            receive_array4 = np.zeros(num_trans_pprocess4, dtype='i4')

            comm.barrier()

            comm.Scatter([id_list1, num_trans_pprocess1, MPI.INT],
                     [receive_array1, num_trans_pprocess1, MPI.INT], root=0)

            comm.Scatter([id_list2, num_trans_pprocess2, MPI.INT],
                     [receive_array2, num_trans_pprocess2, MPI.INT], root=0)

            comm.Scatter([id_list3, num_trans_pprocess3, MPI.INT],
                     [receive_array3, num_trans_pprocess3, MPI.INT], root=0)

            comm.Scatter([id_list4, num_trans_pprocess4, MPI.INT],
                     [receive_array4, num_trans_pprocess4, MPI.INT], root=0)

            # Trim the nonsense from the process arrays
            id_del1 = []
            for i in range(num_trans_pprocess1):
                if abs(receive_array1[i]) < 1e-250:
                    id_del1.append(i)
            receive_array1 = np.delete(receive_array1, id_del1, 0)

            id_del2 = []
            for i in range(num_trans_pprocess2):
                if abs(receive_array2[i]) < 1e-250:
                    id_del2.append(i)
            receive_array2 = np.delete(receive_array2, id_del2, 0)

            id_del3 = []
            for i in range(num_trans_pprocess3):
                if abs(receive_array3[i]) < 1e-250:
                    id_del3.append(i)
            receive_array3 = np.delete(receive_array3, id_del3, 0)

            id_del4 = []
            for i in range(num_trans_pprocess4):
                if abs(receive_array4[i]) < 1e-250:
                    id_del4.append(i)
            receive_array4 = np.delete(receive_array4, id_del4, 0)

            id_list_pprocess1 = receive_array1.tolist()
            id_list_pprocess2 = receive_array2.tolist()
            id_list_pprocess3 = receive_array3.tolist()
            id_list_pprocess4 = receive_array4.tolist()

            receive_array1 = None
            receive_array2 = None
            receive_array3 = None
            receive_array4 = None

            detections_pp_1 = detections1[detections1['transient_id'].isin(id_list_pprocess1)]
            detections_pp_2 = detections2[detections2['transient_id'].isin(id_list_pprocess2)]
            detections_pp_3 = detections3[detections3['transient_id'].isin(id_list_pprocess3)]
            detections_pp_4 = detections4[detections4['transient_id'].isin(id_list_pprocess4)]

            detections1 = None
            detections2 = None
            detections3 = None
            detections4 = None
            id_list_pprocess1 = None
            id_list_pprocess2 = None
            id_list_pprocess3 = None
            id_list_pprocess4 = None

            num_pp1 = len(detections_pp_1['transient_id'].unique())
            num_pp2 = len(detections_pp_2['transient_id'].unique())
            num_pp3 = len(detections_pp_3['transient_id'].unique())
            num_pp4 = len(detections_pp_4['transient_id'].unique())

            df_pp1 = ca.new_band_delta_N(cadence, model, params, other_obs, detections_pp_1, num_pp1, like=False)
            df_pp2 = ca.new_band_delta_N(cadence, model, params, other_obs, detections_pp_2, num_pp2, like=False)
            df_pp3 = ca.new_band_delta_N(cadence, model, params, other_obs, detections_pp_3, num_pp3, like=True)
            df_pp4 = ca.new_band_delta_N(cadence, model, params, other_obs, detections_pp_4, num_pp4, like=True)

            other_obs = None
            params = None
            detections_pp_1 = None
            detections_pp_2 = None
            detections_pp_3 = None
            detections_pp_4 = None

            if size > 1:
                df1_receive = comm.allgather(df_pp1)
                df2_receive = comm.allgather(df_pp2)
                df3_receive = comm.allgather(df_pp3)
                df4_receive = comm.allgather(df_pp4)

                for i in range(size):
                    df1 = df1.add(df1_receive[i], fill_value=0.0)
                    df2 = df2.add(df2_receive[i], fill_value=0.0)
                    df3 = df3.add(df3_receive[i], fill_value=0.0)
                    df4 = df4.add(df4_receive[i], fill_value=0.0)

            df1_receive = None
            df2_receive = None
            df3_receive = None
            df4_receive = None
            df_pp1 = None
            df_pp2 = None
            df_pp3 = None
            df_pp4 = None

    if rank == 0:
        df1.to_csv(output_path + 'band_delta_N_scolnic_coadd.csv')
        df2.to_csv(output_path + 'band_delta_N_scolnic_no_coadd.csv')
        df3.to_csv(output_path + 'band_delta_N_scolnic_like_coadd.csv')
        df4.to_csv(output_path + 'band_delta_N_scolnic_like_no_coadd.csv')
        # plot_delta_n(plot_df)
        print('Finished delta N calculations.')
        print('The recalculation of detections for all bands took {} seconds.'.format(datetime.timedelta(seconds=time.time()-t0)))
