import numpy as np
import pandas as pd
import datajoint as dj
import dj_connect
import getSchema
import mat73

data = mat73.loadmat('simplified_spike_data.mat')
data_simplified = data['simplified_data']
conn = dj_connect.connectToDataJoint("talch012", "simple")

schema = getSchema.getSchema("EPHYS_TEST")
schema_module = dj.VirtualModule("schema_module", "talch012_EPHYS_TEST", create_tables=True)
print(schema.list_tables())
exp_schema = getSchema.getSchema("EXPt")
EXPt = dj.VirtualModule("EXPt", "talch012_expt", create_tables=True)
# print(schema_module.ElectrodeGroup())


print("Schema:", schema)
print("Tables in the schema:", schema.list_tables())

print("EXPt schema:", EXPt)
print("EXPt tables:", exp_schema.list_tables())
print("EXPt.SessionTrial10 table:", (EXPt.SessionTrial10 & 'session=1' & 'subject_id=101104').fetch())

session = 1
subject_id = 101104

schema_module.Probe.insert1({1, "Neuropixels 2.0", "Test probe"}, skip_duplicates=True)

session_key = (EXPt.Session & 'session=1' & 'subject_id=101104').fetch1('session', 'subject_id')
probe_key = (schema_module.Probe & 'probe_part_no=1').fetch1('probe_part_no')

schema_module.ElectrodeGroup.insert1(dict(session=session_key[0], subject_id=session_key[1], electrode_group=1, probe_part_no=probe_key), skip_duplicates=True)

# Create empty DataFrames for the relevant tables in the schema

# Using columns from the foreign keys (Session and ElectrodeGroup) plus unit fields.
unit_cols = ["session", "subject_id", "electrode_group", "unit", "unit_uid", "unit_quality", "unit_channel"]
df_unit = pd.DataFrame(columns=unit_cols)

# Waveform set columns
waveform_cols = ["session", "subject_id", "electrode_group", "unit", "waveform", "spk_width_ms", "sampling_fq", "waveform_amplitude"]
df_waveform = pd.DataFrame(columns=waveform_cols)

# (includes columns from Unit and SessionTrial plus spike_times).
trial_spikes_cols = ["session", "subject_id", "electrode_group", "unit", "trial", "spike_times"]
df_trial_spikes = pd.DataFrame(columns=trial_spikes_cols)

for i in range(len(data_simplified['example_waveforms'])):
    # TODO: Add the relevant data to the i unit row.
    # TODO: Create a new row for the Unit table.
    df_waveform.loc[i] = {
        "session": session,
        "subject_id": subject_id,
        "electrode_group": 1,
        "unit": i,
        "waveform": data_simplified['example_waveforms'][i][0],
        "spk_width_ms": 0.0,  # Placeholder, adjust as needed
        "sampling_fq": data_simplified['fs'],
        "waveform_amplitude": data_simplified['example_waveforms'][i][0].max(),  # Example calculation
    }
    
    df_unit.loc[i] = {
        "session": session,
        "subject_id": subject_id,
        "electrode_group": 1,
        "unit": i,
        "unit_uid": f"{subject_id}{session}{i}",
        "unit_quality": "good",  # Placeholder, adjust as needed
        "unit_channel": ""  # Placeholder, adjust as needed
    }


schema_module.Unit.insert(df_unit, skip_duplicates=True, allow_direct_insert=True)
schema_module.UnitWaveform.insert(df_waveform, skip_duplicates=True, allow_direct_insert=True)

print("Unit table populated with data.")
print(schema_module.Unit())

# Show the cloumns of the table Trial_spikes
print("TrialSpikes table columns:", schema_module.TrialSpikes().heading)

for i in range(183):
    # TODO: Add the relevant data to the i trial row.
    # TODO: Create a new row for the tables.
    # current_session_trial_key_string = f'session=1 & subject_id=101104 & trial={i + 1}'
    session_trial_key = (EXPt.SessionTrial10 & 'session=1' & 'subject_id=101104' & f'trial={i+1}').fetch1('session', 'subject_id', 'trial')
    print(f"SessionTrial key for trial {i + 1}: {session_trial_key}")

    unit_key = (schema_module.Unit & 'session=1' & 'subject_id=101104' & 'electrode_group=1' & f'unit={data_simplified['trial_spikes'][0][0]['cluster_ids'][i]}').fetch1('session', 'subject_id', 'electrode_group', 'unit')
    print(f"Unit key for unit {data_simplified['trial_spikes'][0][0]['cluster_ids'][i]}: {unit_key}")

    # schema_module.TrialSpikes.insert1(dict(session=session_trial_key[0], subject_id=session_trial_key[1],
    #                                        trial=session_trial_key[2], electrode_group=unit_key[2], unit=unit_key[3],
    #                                        spike_times=data_simplified['trial_spikes'][0][0]['spike_times_sec'][i]), skip_duplicates=True, allow_direct_insert=True)

    df_trial_spikes.loc[i] = {
        "session": session,
        "subject_id": subject_id,
        "electrode_group": 1,
        "unit": data_simplified['trial_spikes'][0][0]['cluster_ids'][i],
        "trial": i + 1,  # Assuming trial is indexed by i + 1
        "spike_times": data_simplified['trial_spikes'][0][0]['spike_times_sec'][i]  # Convert to list
    }


print("TrialSpikes DataFrame:\n" , df_trial_spikes)
schema_module.TrialSpikes.insert(df_trial_spikes, skip_duplicates=True, allow_direct_insert=True)

print("TrialSpikes table populated with data.")
print(f"Number of rows in TrialSpikes: {len(schema_module.TrialSpikes())}")
print("First few rows of TrialSpikes:")
print(schema_module.TrialSpikes().fetch())

'''
dict_keys(['example_waveforms', 'file_names', 'fs', 'num_clusters', 'num_trials', 'trial_durations', 'trial_spikes', 'unique_clusters', 'unit_locations'])
len of example waveforms: 595
num_trials: 184.0
len of trial_durations: 184
number of clusters: 595.0
fs: 30000.0
len of trial_spikes: 184
len of file_names: 184

Schema `almog_EPHYS_TEST3`

Tables in the schema:
#cell_type
#cluster_quality_label
probe_insertion
_ephys_recording
_ephys_recording__channel
_ephys_recording__ephys_file
insertion_location
_l_f_p
_l_f_p__electrode
_curated_clustering
_curated_clustering__unit
_trial_spikes
_waveform_set
_waveform_set__waveform
_waveform_set__peak_waveform

Probe Columns: [probe_part_no, probe_type, probe_comment]
Session Columns: [subject_id, session_date, session_time, session_uid]
SessionTrial Columns: [subject_id, session_date, session_time, trial, trial_uid, trial_start_time, trial_end_time]
ElectrodeGroup Columns: [subject_id, session_date, session_time, electrode_group, probe_part_no] 
CellType Columns: [cell_type, cell_type_description]
Unit Columns: [subject_id, session_date, session_time, electrode_group, probe_part_no, unit, unit_uid, unit_quality, unit_channel]
TrialSpikes Columns: [subject_id, session_date, session_time, electrode_group, probe_part_no, unit, trial, spike_times]

'''
'''
print(data_simplified.keys())
print(f"len of example waveforms: {len(data_simplified['example_waveforms'])}")
print(f"num_trials: {data_simplified['num_trials']}")
print(f"len of trial_durations: {len(data_simplified['trial_durations'])}")
print(f"number of clusters: {data_simplified['num_clusters']}")
print(f"fs: {data_simplified['fs']}")
print(f"len of trial_spikes: {len(data_simplified['trial_spikes'])}")
print(f"len of file_names: {len(data_simplified['file_names'])}")
'''

'''
print("Connected to DataJoint schema:", schema)
# Check if the schema is empty
if not schema.list_tables():
    print("Schema is empty. Proceeding to create tables.")
else:
    print("Schema is not empty. Tables already exist.")

# Print the tables in the schema
print("Tables in the schema:")
for table_name in schema.list_tables():
    print(table_name)

probe_cols = ["probe_part_no", "probe_type", "probe_comment"]
df_probe = pd.DataFrame(columns=probe_cols)

session_cols = ["subject_id", "session_date", "session_time", "session_uid"]
df_session = pd.DataFrame(columns=session_cols)

# Note: the foreign key to Session yields subject_id, session_date, session_time.
session_trial_cols = ["subject_id", "session_date", "session_time", "trial", "trial_uid", "trial_start_time", "trial_end_time"]
df_session_trial = pd.DataFrame(columns=session_trial_cols)

# The foreign key to Session gives subject_id, session_date, session_time, plus electrode_group and the Probe's primary key.
electrode_group_cols = ["subject_id", "session_date", "session_time", "electrode_group", "probe_part_no"]
df_electrode_group = pd.DataFrame(columns=electrode_group_cols)

cell_type_cols = ["cell_type", "cell_type_description"]
df_cell_type = pd.DataFrame(columns=cell_type_cols)

# Using columns from the foreign keys (Session and ElectrodeGroup) plus unit fields.
unit_cols = ["subject_id", "session_date", "session_time", "electrode_group", "probe_part_no", "unit", "unit_uid", "unit_quality", "unit_channel"]
df_unit = pd.DataFrame(columns=unit_cols)

# (includes columns from Unit and SessionTrial plus spike_times).
trial_spikes_cols = ["subject_id", "session_date", "session_time", "electrode_group", "probe_part_no", "unit", "trial", "spike_times"]
df_trial_spikes = pd.DataFrame(columns=trial_spikes_cols)

print("\nEmpty DataFrames:")
print("Probe:\n", df_probe)
print("Session:\n", df_session)
print("SessionTrial:\n", df_session_trial)
print("ElectrodeGroup:\n", df_electrode_group)
print("CellType:\n", df_cell_type)
print("Unit:\n", df_unit)
print("TrialSpikes:\n", df_trial_spikes)

# for i in range(len(data_simplified['example_waveforms'])):
#     # TODO: Add the relevant data to the i unit row.
#     # TODO: Create a new row for the Unit table.
#     continue

# for i in range(len(data_simplified['trial_spikes'])):
#     # TODO: Add the relevant data to the i trial row.
#     # TODO: Create a new row for the tables.
#     continue
'''