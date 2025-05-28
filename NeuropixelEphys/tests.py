import numpy as np
import pandas as pd
import datajoint as dj
import os
import re
import dj_connect
import getSchema

# arseny_schema = dj.VirtualModule("arseny_schema", "arseny_s1alm_experiment")

# exp = dj.VirtualModule("exp", "arseny_s1alm_experiment")
# # schema = dj.Schema("arseny_s1alm_experiment")
# # schema.spawn_missing_classes()
# print(schema.list_tables())
# schema = dj.VirtualModule("ephys", "almog_EPHYS")
# schema.ProbeInsertion.drop()
# Create a new schema for the copy
# almog_schema = dj.Schema('almog_EPHYS')

# Copy tables from arseny_schema to almog_schema
# for table_name in arseny_schema.list_tables():
#     table = getattr(arseny_schema, table_name)
#     class Copy(table.__class__):
#         definition = table.definition
#         _schema = almog_schema

# print(arseny_schema)
# print(almog_schema.list_tables())

# show the virtual module
# schema = dj.VirtualModule("arseny", "arseny_s1alm_experiment")
# query = schema.Session
# data = query.fetch()
# print(query)
# schema = dj.VirtualModule('EPHYS', conn)
# dj.Diagram(schema).draw()

import mat73
data = mat73.loadmat('simplified_spike_data.mat')
data_simplified = data['simplified_data']

print(data_simplified.keys())
print(f"len of example waveforms: {len(data_simplified['example_waveforms'])}")
print(f"num_trials: {data_simplified['num_trials']}")
print(f"len of trial_durations: {len(data_simplified['trial_durations'])}")
print(f"number of clusters: {data_simplified['num_clusters']}")
print(f"fs: {data_simplified['fs']}")
print(f"len of trial_spikes: {len(data_simplified['trial_spikes'])}")
print(f"len of file_names: {len(data_simplified['file_names'])}")

conn = dj_connect.connectToDataJoint("almog", "simple")
schema = getSchema.getSchema("EPHYS_TEST")

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

for i in range(data_simplified['num_clusters']):
    # TODO: Add the relevant data to the i unit row.
    # TODO: Create a new row for the Unit table.

for i in range(data_simplified['num_trials']):
    # TODO: Add the relevant data to the i trial row.
    # TODO: Create a new row for the tables.