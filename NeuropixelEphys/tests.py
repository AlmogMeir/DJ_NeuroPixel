import numpy as np
import pandas as pd
# import datajoint as dj
import mat73

data = mat73.loadmat('simplified_spike_data.mat')
data_simplified = data['simplified_data']

print(data_simplified['example_waveforms'][0])


# trial_spikes_cols = ["session", "subject_id", "electrode_group", "unit", "trial", "spike_times"]
# df_trial_spikes = pd.DataFrame(columns=trial_spikes_cols)

# # print(len(data_simplified['trial_spikes'][1][0]['cluster_ids']))

# # for i in range(len(data_simplified['trial_spikes']) - 1):
#     # TODO: Add the relevant data to the i trial row.
#     # TODO: Create a new row for the tables.
#     # current_session_trial_key_string = f'session=1 & subject_id=101104 & trial={i + 1}'
#     # session_trial_key = (EXPt.SessionTrial10 & 'session=1' & 'subject_id=101104' & f'trial={i+1}').fetch1('session', 'subject_id', 'trial')
#     # print(f"SessionTrial key for trial {i + 1}: {session_trial_key}")

#     # unit_key = (schema_module.Unit & 'session=1' & 'subject_id=101104' & 'electrode_group=1' & f'unit={data_simplified['trial_spikes'][0][0]['cluster_ids'][i]}').fetch1('session', 'subject_id', 'electrode_group', 'unit')
#     # print(f"Unit key for unit {data_simplified['trial_spikes'][0][0]['cluster_ids'][i]}: {unit_key}")

#     # schema_module.TrialSpikes.insert1(dict(session=session_trial_key[0], subject_id=session_trial_key[1],
#     #                                        trial=session_trial_key[2], electrode_group=unit_key[2], unit=unit_key[3],
#     #                                        spike_times=data_simplified['trial_spikes'][0][0]['spike_times_sec'][i]), skip_duplicates=True, allow_direct_insert=True)
    
# for i in range(5):
#     # TODO: Add the relevant data to the i trial row.
#     # TODO: Create a new row for the tables.
#     # current_session_trial_key_string = f'session=1 & subject_id=101104 & trial={i + 1}'
#     # session_trial_key = (EXPt.SessionTrial10 & 'session=1' & 'subject_id=101104' & f'trial={i+1}').fetch1('session', 'subject_id', 'trial')
#     # print(f"SessionTrial key for trial {i + 1}: {session_trial_key}")

#     # unit_key = (schema_module.Unit & 'session=1' & 'subject_id=101104' & 'electrode_group=1' & f'unit={data_simplified['trial_spikes'][0][0]['cluster_ids'][i]}').fetch1('session', 'subject_id', 'electrode_group', 'unit')
#     # print(f"Unit key for unit {data_simplified['trial_spikes'][0][0]['cluster_ids'][i]}: {unit_key}")

#     # schema_module.TrialSpikes.insert1(dict(session=session_trial_key[0], subject_id=session_trial_key[1],
#     #                                        trial=session_trial_key[2], electrode_group=unit_key[2], unit=unit_key[3],
#     #                                        spike_times=data_simplified['trial_spikes'][0][0]['spike_times_sec'][i]), skip_duplicates=True, allow_direct_insert=True)
    
#     num_of_units = data_simplified['num_clusters']
#     num_of_spikes = len(data_simplified['trial_spikes'][i][0]['spike_times_sec'])

#     print(f"trial {i+1} num of spikes: {len(data_simplified['trial_spikes'][i][0]['cluster_ids'])}")

#     # create empty dict for spike times of all units in the trial
#     spike_times_dict = {unit: [] for unit in range(int(num_of_units))}

    
#     # Populate the spike times for each unit in the trial
#     for spike in range(len(data_simplified['trial_spikes'][i][0]['cluster_ids'])):
#         unit = data_simplified['trial_spikes'][i][0]['cluster_ids'][spike]
#         spike_times_dict[unit].append(data_simplified['trial_spikes'][i][0]['spike_times_sec'][spike])
        
#     for j in range(int(num_of_units)):
#         if j in spike_times_dict:
#             spike_times = spike_times_dict[j]
#         else:
#             spike_times = []

#         df_trial_spikes.loc[int(num_of_units * i + j)] = {
#             "session": 1,
#             "subject_id": 111,
#             "electrode_group": 1,
#             "unit": j,
#             "trial": i + 1,  # Trials are 1-indexed
#             "spike_times": spike_times
#         }
    
# print("TrialSpikes DataFrame:\n" , df_trial_spikes)

# unit_spikes_cols = ["session", "subject_id", "electrode_group", "unit", "spike_times"]
# df_unit_spikes = pd.DataFrame(columns=unit_spikes_cols)

# # Initialize a dictionary to collect spike times for each unit across all trials
# unit_spike_times_dict = {unit: [] for unit in range(int(data_simplified['num_clusters']))}

# for i in range(30):
#     for spike in range(len(data_simplified['trial_spikes'][i][0]['cluster_ids'])):
#         unit = data_simplified['trial_spikes'][i][0]['cluster_ids'][spike]
#         spike_time = data_simplified['trial_spikes'][i][0]['spike_times_sec'][spike]
#         unit_spike_times_dict[unit].append(spike_time)

# for unit in range(int(data_simplified['num_clusters'])):
#     df_unit_spikes.loc[unit] = {
#         "session": 1,
#         "subject_id": 111,
#         "electrode_group": 1,
#         "unit": unit,
#         "spike_times": unit_spike_times_dict[unit]
#     }

# print("UnitSpikes DataFrame:\n", df_unit_spikes)
# print(f"Spike train length for unit 0: {len(unit_spike_times_dict[0])}")


    # print(f"Trial {i + 1} spike times: {spike_times_dict}")