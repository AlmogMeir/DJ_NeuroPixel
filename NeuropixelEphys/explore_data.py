import mat73

data = mat73.loadmat('simplified_spike_data.mat')
data_simplified = data['simplified_data']

print(data_simplified['example_waveforms'][0][0].max())
print(data_simplified['trial_spikes'][0][0])

print(f"first unit in first trial: {data_simplified['trial_spikes'][0][0]['cluster_ids'][0]}")
print(f"first time in first trial: {data_simplified['trial_spikes'][0][0]['spike_times_sec'][0]}")