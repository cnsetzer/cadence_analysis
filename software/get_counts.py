import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import astrotog
from copy import copy, deepcopy
# I personally like this style.
sns.set_style('whitegrid')
# Easy to change context from `talk`, `notebook`, `poster`, `paper`.
sns.set_context('paper')
output_path = '/Users/cnsetzer/Documents/LSST/astrotog_output/scolnic_results/'
output_dirs = os.listdir(output_path)

results = {}
for directory in output_dirs:
    if directory == '.DS_Store':
        continue
    else:
        name_split = re.split('_', directory)
        model = name_split[1]
        name = name_split[2]
        if re.search('minion', name) is not None:
            name += name_split[3]
        for string in name_split:
            if re.search('var', string) is not None:
                name += string

        results[name] = {}
        results[name][model] = {}
        results[name][model]['data'] = {}
        results[name][model]['data']['observations'] = pd.read_csv(output_path + directory + '/observations.csv', index_col=0)
        results[name][model]['data']['parameters'] = pd.read_csv(output_path + directory + '/modified_parameters.csv', index_col=0)
        results[name][model]['data']['coadded observations'] = pd.read_csv(output_path + directory + '/coadded_observations.csv', index_col=0)
        results[name][model]['data']['other observations'] = pd.read_csv(output_path + directory + '/other_observations.csv', index_col=0)
        results[name][model]['data']['scolnic detections'] = pd.read_csv(output_path + directory + '/scolnic_detections.csv', index_col=0)
        results[name][model]['number scolnic'] = len(results[name][model]['data']['scolnic detections']['transient_id'].unique())

fiducial_scolnic = 74.5

sc_raw_cadence_numbers = []
sc_cadence = []

rw_raw_cadence_numbers = []
rw_cadence = []

sc_var_cadence_numbers = []
sc_var_cadence = []

rw_var_cadence_numbers = []
rw_var_cadence = []

sc_min_var_cadence = []
sc_min_var_cadence_numbers = []

for key in results.keys():
    for key2 in results[key].keys():
        if key2 == 'scolnic' and re.search('var', key) is None:
            sc_cadence.append(key)
            sc_raw_cadence_numbers.append(results[key][key2]['number scolnic'])
        elif key2 == 'rosswog' and re.search('var', key) is None:
            rw_cadence.append(key)
            rw_raw_cadence_numbers.append(results[key][key2]['number scolnic'])
        elif re.search('var', key) is not None:
            if re.search('scolnic', key2) is not None and re.search('minion', key) is None:
                sc_var_cadence.append(key)
                sc_var_cadence_numbers.append(results[key][key2]['number scolnic'])
            elif re.search('rosswog', key2) is not None:
                rw_var_cadence.append(key)
                rw_var_cadence_numbers.append(results[key][key2]['number scolnic'])
            elif re.search('minion', key) is not None:
                sc_min_var_cadence.append(key)
                sc_min_var_cadence_numbers.append(results[key][key2]['number scolnic'])


# sort based on number detected
if len(sc_cadence) > 0:
    sc_sorted_cadences = [x for _, x in sorted(zip(sc_raw_cadence_numbers, sc_cadence), key=lambda pair: pair[0])]
    sc_raw_cadence_numbers.sort()
    sc_sorted_cadence_numbers = sc_raw_cadence_numbers

if len(rw_cadence) > 0:
    rw_sorted_cadences = [x for _, x in sorted(zip(rw_raw_cadence_numbers, rw_cadence), key=lambda pair: pair[0])]
    rw_raw_cadence_numbers.sort()
    rw_sorted_cadence_numbers = rw_raw_cadence_numbers

if len(sc_var_cadence) > 0:
    sc_var_sorted_cadences = [x for _, x in sorted(zip(sc_var_cadence_numbers, sc_var_cadence), key=lambda pair: pair[0])]
    sc_var_cadence_numbers.sort()
    sc_var_sorted_cadence_numbers = sc_var_cadence_numbers

if len(rw_var_cadence) > 0:
    rw_var_sorted_cadences = [x for _, x in sorted(zip(rw_var_cadence_numbers, rw_var_cadence), key=lambda pair: pair[0])]
    rw_var_cadence_numbers.sort()
    rw_var_sorted_cadence_numbers = rw_var_cadence_numbers

if len(sc_min_var_cadence) > 0:
    sc_min_var_sorted_cadences = [x for _, x in sorted(zip(sc_min_var_cadence_numbers, sc_min_var_cadence), key=lambda pair: pair[0])]
    sc_min_var_cadence_numbers.sort()
    sc_min_sorted_cadence_numbers = sc_min_var_cadence_numbers

print(list(zip(sc_sorted_cadences, sc_sorted_cadence_numbers)))
print(list(zip(rw_sorted_cadences, rw_sorted_cadence_numbers)))
print(list(zip(sc_var_sorted_cadences, sc_var_sorted_cadence_numbers)))
print(list(zip(rw_var_sorted_cadences, rw_var_sorted_cadence_numbers)))
print(list(zip(sc_min_var_sorted_cadences, sc_min_var_sorted_cadence_numbers)))










fig = plt.figure()
plt.scatter(sc_sorted_cadences, sc_sorted_cadence_numbers)
plt.ylim([50, 100])
plt.axhline(fiducial_scolnic,color='r')
plt.xticks(rotation='vertical')
plt.title('Number of Detections vs. Cadence (Scolnic et. al 2017)')
plt.ylabel('Number of Detected KNe')
plt.legend(['Value from Scolnic et. al 2017','Number Detected per Cadence'])
plt.show()
plt.savefig('Sorted_raw_counts.pdf',bbox_inches='tight')
