import datajoint as dj
import numpy as np
import pandas as pd
import pathlib
import importlib
import inspect
import re

import dj_connect
import getSchema
import probe
from readers import kilosort, spikeglx


# from element_interface.utils import dict_to_uuid, find_full_path, find_root_directory

# Below is an alternative to define the functions that not imported properly from element_interface.
# This is for the case that element_interface failed to install on the env.

# """
from uuid import uuid4
from pathlib import Path

def dict_to_uuid(input_dict):
    # Generate a UUID based on a dictionary.
    return str(uuid4())

def find_full_path(root_directories, relative_path):
    # Find the full path given a list of root directories and a relative path.
    for root_dir in root_directories:
        full_path = Path(root_dir) / relative_path
        if full_path.exists():
            return full_path
    raise FileNotFoundError(f"File {relative_path} not found in any of the root directories.")

def find_root_directory(root_directories, target_path):
    # Find the root directory containing the target path.
    target_path = Path(target_path).resolve()
    for root_dir in root_directories:
        root_dir = Path(root_dir).resolve()
        if target_path.is_relative_to(root_dir):
            return root_dir
    raise FileNotFoundError(f"Root directory for {target_path} not found.")
# """

conn = dj_connect.connectToDataJoint("almog", "simple")
schema = dj.Schema("almog_EPHYS")
exp = dj.VirtualModule("EXP", "arseny_s1alm_experiment")
# schema = dj.Schema("arseny_s1alm_experiment")
# schema.spawn_missing_classes()
logger = dj.logger
_linking_module = None

def activate(
    ephys_schema_name: str,
    *,
    create_schema: bool = True,
    create_tables: bool = True,
    linking_module: str = None,
):
    """Activates the `ephys` and `probe` schemas.

    Args:
        ephys_schema_name (str): A string containing the name of the ephys schema.
        create_schema (bool): If True, schema will be created in the database.
        create_tables (bool): If True, tables related to the schema will be created in the database.
        linking_module (str): A string containing the module name or module containing the required dependencies to activate the schema.

    Dependencies:
    Upstream tables:
        Session: A parent table to ProbeInsertion
        Probe: A parent table to EphysRecording. Probe information is required before electrophysiology data is imported.

    Functions:
        get_ephys_root_data_dir(): Returns absolute path for root data director(y/ies) with all electrophysiological recording sessions, as a list of string(s).
        get_session_direction(session_key: dict): Returns path to electrophysiology data for the a particular session as a list of strings.
        get_processed_data_dir(): Optional. Returns absolute path for processed data. Defaults to root directory.
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(
        linking_module
    ), "The argument 'dependency' must be a module's name or a module"

    global _linking_module
    _linking_module = linking_module

    if not probe.schema.is_activated():
        raise RuntimeError("Please activate the `probe` schema first.")

    schema.activate(
        ephys_schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=_linking_module.__dict__,
    )

# -------------- Functions required by the elements-ephys  ---------------


def get_ephys_root_data_dir() -> list:
    """Fetches absolute data path to ephys data directories.

    The absolute path here is used as a reference for all downstream relative paths used in DataJoint.

    Returns:
        A list of the absolute path(s) to ephys data directories.
    """
    root_directories = _linking_module.get_ephys_root_data_dir()
    if isinstance(root_directories, (str, pathlib.Path)):
        root_directories = [root_directories]

    if hasattr(_linking_module, "get_processed_root_data_dir"):
        root_directories.append(_linking_module.get_processed_root_data_dir())

    return root_directories


def get_session_directory(session_key: dict) -> str:
    """Retrieve the session directory with Neuropixels for the given session.

    Args:
        session_key (dict): A dictionary mapping subject to an entry in the subject table, and session_datetime corresponding to a session in the database.

    Returns:
        A string for the path to the session directory.
    """
    return _linking_module.get_session_directory(session_key)


def get_processed_root_data_dir() -> str:
    """Retrieve the root directory for all processed data.

    Returns:
        A string for the full path to the root directory for processed data.
    """

    if hasattr(_linking_module, "get_processed_root_data_dir"):
        return _linking_module.get_processed_root_data_dir()
    else:
        return get_ephys_root_data_dir()[0]

# ----------------------------- Table declarations ----------------------
@schema
class Session(dj.Manual):
    """Session information.

    Attributes:
        session_datetime (datetime): Date and time of the session.
        session_directory (varchar(255) ): Relative path to the session directory.
    """

    definition = """
    # Session information
    session_datetime: datetime
    ---
    session_directory: varchar(255)
    """

@schema
class ProbeInsertion(dj.Manual):
    """Information about probe insertion across subjects and sessions.

    Attributes:
        Session (foreign key): Session primary key.
        insertion_number (foreign key, str): Unique insertion number for each probe insertion for a given session.
        probe.Probe (str): probe.Probe primary key.
    """

    definition = """
    # Probe insertion implanted into an animal for a given session.
    -> Session
    insertion_number: tinyint unsigned
    ---
    -> probe.Probe
    """

    @classmethod
    def auto_generate_entries(cls, session_key):
        """Automatically populate entries in ProbeInsertion table for a session."""
        session_dir = find_full_path(get_ephys_root_data_dir(), get_session_directory(session_key))
        # search session dir and determine acquisition software
        # for ephys_pattern in ("*.ap.meta"):
        #     ephys_meta_filepaths = list(session_dir.rglob(ephys_pattern))
        #     if ephys_meta_filepaths:
        #         acq_software = "SpikeGLX"
        #         break
        # else:
        #     raise FileNotFoundError(
        #         f"Ephys recording data not found!"
        #         f"SpikeGLX recording files not found in: {session_dir}"
        #     )
        
        # This row is instead of the for loop above
        ephys_meta_filepaths = list(session_dir.rglob("*.ap.meta"))

        probe_list, probe_insertion_list = [], []
        for meta_fp_idx, meta_filepath in enumerate(ephys_meta_filepaths):
            spikeglx_meta = spikeglx.SpikeGLXMeta(meta_filepath)

            probe_key = {
                "probe_type": spikeglx_meta.probe_model,
                "probe": spikeglx_meta.probe_SN,
            }
            if probe_key["probe"] not in [p["probe"] for p in probe_list]:
                probe_list.append(probe_key)

            probe_dir = meta_filepath.parent
            try:
                probe_number = re.search(r"(imec)?\d{1}$", probe_dir.name).group()
                probe_number = int(probe_number.replace("imec", ""))
            except AttributeError:
                probe_number = meta_fp_idx

            probe_insertion_list.append(
                {
                    **session_key,
                    "probe": spikeglx_meta.probe_SN,
                    "insertion_number": int(probe_number),
                }
            )

        probe.Probe.insert(probe_list, skip_duplicates=True)
        cls.insert(probe_insertion_list, skip_duplicates=True)


@schema
class InsertionLocation(dj.Manual):
    """Stereotaxic location information for each probe insertion.

    Attributes:
        ProbeInsertion (foreign key): ProbeInsertion primary key.
        ap_location (decimal (6, 2) ): Anterior-posterior location in micrometers. Reference is 0 with anterior values positive.
        ml_location (decimal (6, 2) ): Medial-lateral location in micrometers. Reference is zero with right side values positive.
        depth (decimal (6, 2) ): Manipulator depth relative to the surface of the brain at zero. Ventral is negative.
        Theta (decimal (5, 2) ): elevation - rotation about the ml-axis in degrees relative to positive z-axis.
        phi (decimal (5, 2) ): azimuth - rotation about the dv-axis in degrees relative to the positive x-axis

    """

    definition = """
    # Brain Location of a given probe insertion.
    -> ProbeInsertion
    ---
    ap_location: decimal(6, 2) # (um) anterior-posterior; ref is 0; more anterior is more positive
    ml_location: decimal(6, 2) # (um) medial axis; ref is 0 ; more right is more positive
    depth:       decimal(6, 2) # (um) manipulator depth relative to surface of the brain (0); more ventral is more negative
    theta=null:  decimal(5, 2) # (deg) - elevation - rotation about the ml-axis [0, 180] - w.r.t the z+ axis
    phi=null:    decimal(5, 2) # (deg) - azimuth - rotation about the dv-axis [0, 360] - w.r.t the x+ axis
    beta=null:   decimal(5, 2) # (deg) rotation about the shank of the probe [-180, 180] - clockwise is increasing in degree - 0 is the probe-front facing anterior
    """


@schema
class EphysRecording(dj.Imported):
    """Automated table with electrophysiology recording information for each probe inserted during an experimental session.

    Attributes:
        ProbeInsertion (foreign key): ProbeInsertion primary key.
        probe.ElectrodeConfig (dict): probe.ElectrodeConfig primary key.
        sampling_rate (float): sampling rate of the recording in Hertz (Hz).
        recording_datetime (datetime): datetime of the recording from this probe.
        recording_duration (float): duration of the entire recording from this probe in seconds.
    """

    definition = """
    # Ephys recording from a probe insertion for a given session.
    -> ProbeInsertion      
    ---
    -> probe.ElectrodeConfig
    sampling_rate: float # (Hz) 
    recording_datetime: datetime # datetime of the recording from this probe
    recording_duration: float # (seconds) duration of the recording from this probe
    """

    class Channel(dj.Part):
        definition = """
        -> master
        channel_idx: int  # channel index (index of the raw data)
        ---
        -> probe.ElectrodeConfig.Electrode
        channel_name="": varchar(64)  # alias of the channel
        """

    class EphysFile(dj.Part):
        """Paths of electrophysiology recording files for each insertion.

        Attributes:
            EphysRecording (foreign key): EphysRecording primary key.
            file_path (varchar(255) ): relative file path for electrophysiology recording.
        """

        definition = """
        # Paths of files of a given EphysRecording round.
        -> master
        file_path: varchar(255)  # filepath relative to root data directory
        """

    def make(self, key):
        """Populates table with electrophysiology recording information."""
        session_dir = find_full_path(
            get_ephys_root_data_dir(), get_session_directory(key)
        )
        inserted_probe_serial_number = (ProbeInsertion * probe.Probe & key).fetch1(
            "probe"
        )

        # Search session dir and determine acquisition software
        # for ephys_pattern in ("*.ap.meta"):
        #     ephys_meta_filepaths = list(session_dir.rglob(ephys_pattern))
        #     if ephys_meta_filepaths:
        #         acq_software = "SpikeGLX"
        #         break
        # else:
        #     raise FileNotFoundError(
        #         f"Ephys recording data not found in {session_dir}."
        #         "Neither SpikeGLX nor Open Ephys recording files found"
        #     )

        # The row below is instead of the loop above
        ephys_meta_filepaths = list(session_dir.rglob("*.ap.meta"))

        supported_probe_types = probe.ProbeType.fetch("probe_type")

        for meta_filepath in ephys_meta_filepaths:
            spikeglx_meta = spikeglx.SpikeGLXMeta(meta_filepath)
            if str(spikeglx_meta.probe_SN) == inserted_probe_serial_number:
                spikeglx_meta_filepath = meta_filepath
                break
        else:
            raise FileNotFoundError(
                "No SpikeGLX data found for probe insertion: {}".format(key)
            )

        if spikeglx_meta.probe_model not in supported_probe_types:
            raise NotImplementedError(
                f"Processing for neuropixels probe model {spikeglx_meta.probe_model} not yet implemented."
            )

        probe_type = spikeglx_meta.probe_model
        electrode_query = probe.ProbeType.Electrode & {"probe_type": probe_type}

        probe_electrodes = {
            (shank, shank_col, shank_row): key
            for key, shank, shank_col, shank_row in zip(
                *electrode_query.fetch("KEY", "shank", "shank_col", "shank_row")
            )
        }  # electrode configuration
        electrode_group_members = [
            probe_electrodes[(shank, shank_col, shank_row)]
            for shank, shank_col, shank_row, _ in spikeglx_meta.shankmap["data"]
        ]  # recording session-specific electrode configuration

        econfig_entry, econfig_electrodes = generate_electrode_config_entry(
            probe_type, electrode_group_members
        )

        ephys_recording_entry = {
            **key,
            "electrode_config_hash": econfig_entry["electrode_config_hash"],
            "acq_software": "SpikeGLX",
            "sampling_rate": spikeglx_meta.meta["imSampRate"],
            "recording_datetime": spikeglx_meta.recording_time,
            "recording_duration": (
                spikeglx_meta.recording_duration
                or spikeglx.retrieve_recording_duration(spikeglx_meta_filepath)
            ),
        }

        root_dir = find_root_directory(
            get_ephys_root_data_dir(), spikeglx_meta_filepath
        )

        ephys_file_entries = [
            {
                **key,
                "file_path": spikeglx_meta_filepath.relative_to(
                    root_dir
                ).as_posix(),
            }
        ]

        # Insert channel information
        # Get channel and electrode-site mapping
        channel2electrode_map = {
            recorded_site: probe_electrodes[(shank, shank_col, shank_row)]
            for recorded_site, (shank, shank_col, shank_row, _) in enumerate(
                spikeglx_meta.shankmap["data"]
            )
        }

        ephys_channel_entries = [
            {
                **key,
                "electrode_config_hash": econfig_entry["electrode_config_hash"],
                "channel_idx": channel_idx,
                **channel_info,
            }
            for channel_idx, channel_info in channel2electrode_map.items()
        ]

        # Insert into probe.ElectrodeConfig (recording configuration)
        if not probe.ElectrodeConfig & {
            "electrode_config_hash": econfig_entry["electrode_config_hash"]
        }:
            probe.ElectrodeConfig.insert1(econfig_entry)
            probe.ElectrodeConfig.Electrode.insert(econfig_electrodes)

        self.insert1(ephys_recording_entry)
        self.EphysFile.insert(ephys_file_entries)
        self.Channel.insert(ephys_channel_entries)


@schema
class LFP(dj.Imported):
    """Extracts local field potentials (LFP) from an electrophysiology recording.

    Attributes:
        EphysRecording (foreign key): EphysRecording primary key.
        lfp_sampling_rate (float): Sampling rate for LFPs in Hz.
        lfp_time_stamps (longblob): Time stamps with respect to the start of the recording.
        lfp_mean (longblob): Overall mean LFP across electrodes.
    """

    definition = """
    # Acquired local field potential (LFP) from a given Ephys recording.
    -> EphysRecording
    ---
    lfp_sampling_rate: float   # (Hz)
    lfp_time_stamps: longblob  # (s) timestamps with respect to the start of the recording (recording_timestamp)
    lfp_mean: longblob         # (uV) mean of LFP across electrodes - shape (time,)
    """

    class Electrode(dj.Part):
        """Saves local field potential data for each electrode.

        Attributes:
            LFP (foreign key): LFP primary key.
            probe.ElectrodeConfig.Electrode (foreign key): probe.ElectrodeConfig.Electrode primary key.
            lfp (longblob): LFP recording at this electrode in microvolts.
        """

        definition = """
        -> master
        -> probe.ElectrodeConfig.Electrode  
        ---
        lfp: longblob               # (uV) recorded lfp at this electrode 
        """

    # Only store LFP for every 9th channel, due to high channel density,
    # close-by channels exhibit highly similar LFP
    _skip_channel_counts = 9

    def make(self, key):
        """Populates the LFP tables."""
        electrode_keys, lfp = [], []

        spikeglx_meta_filepath = get_spikeglx_meta_filepath(key)
        spikeglx_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)

        lfp_channel_ind = spikeglx_recording.lfmeta.recording_channels[
            -1 :: -self._skip_channel_counts
        ]

        # Extract LFP data at specified channels and convert to uV
        lfp = spikeglx_recording.lf_timeseries[
            :, lfp_channel_ind
        ]  # (sample x channel)
        lfp = (
            lfp * spikeglx_recording.get_channel_bit_volts("lf")[lfp_channel_ind]
        ).T  # (channel x sample)

        self.insert1(
            dict(
                key,
                lfp_sampling_rate=spikeglx_recording.lfmeta.meta["imSampRate"],
                lfp_time_stamps=(
                    np.arange(lfp.shape[1])
                    / spikeglx_recording.lfmeta.meta["imSampRate"]
                ),
                lfp_mean=lfp.mean(axis=0),
            )
        )

        electrode_query = (
            probe.ProbeType.Electrode
            * probe.ElectrodeConfig.Electrode
            * EphysRecording
            & key
        )
        probe_electrodes = {
            (shank, shank_col, shank_row): key
            for key, shank, shank_col, shank_row in zip(
                *electrode_query.fetch("KEY", "shank", "shank_col", "shank_row")
            )
        }

        for recorded_site in lfp_channel_ind:
            shank, shank_col, shank_row, _ = spikeglx_recording.apmeta.shankmap[
                "data"
            ][recorded_site]
            electrode_keys.append(probe_electrodes[(shank, shank_col, shank_row)])
        
        # single insert in loop to mitigate potential memory issue
        for electrode_key, lfp_trace in zip(electrode_keys, lfp):
            self.Electrode.insert1({**key, **electrode_key, "lfp": lfp_trace})


@schema
class ClusterQualityLabel(dj.Lookup):
    """Quality label for each spike sorted cluster.

    Attributes:
        cluster_quality_label (foreign key, varchar(100) ): Cluster quality type.
        cluster_quality_description (varchar(4000) ): Description of the cluster quality type.
    """

    definition = """
    # Quality
    cluster_quality_label:  varchar(100)  # cluster quality type - e.g. 'good', 'MUA', 'noise', etc.
    ---
    cluster_quality_description:  varchar(4000)
    """
    contents = [
        ("good", "single unit"),
        ("ok", "probably a single unit, but could be contaminated"),
        ("mua", "multi-unit activity"),
        ("noise", "bad unit"),
        ("n.a.", "not available"),
    ]

@schema
class CuratedClustering(dj.Imported):
    """Clustering results after curation.

    Attributes:
        EphysRecording (foreign key): EphysRecording primary key.
    """

    definition = """
    # Clustering results of the spike sorting step.
    -> EphysRecording
    """

    class Unit(dj.Part):
        """Single unit properties after clustering and curation.

        Attributes:
            CuratedClustering (foreign key): CuratedClustering primary key.
            unit (int): Unique integer identifying a single unit.
            probe.ElectrodeConfig.Electrode (foreign key): probe.ElectrodeConfig.Electrode primary key.
            ClusteringQualityLabel (foreign key): CLusteringQualityLabel primary key.
            spike_count (int): Number of spikes in this recording for this unit.
            spike_times (longblob): Spike times of this unit, relative to start time of EphysRecording.
            spike_sites (longblob): Array of electrode associated with each spike.
            spike_depths (longblob): Array of depths associated with each spike, relative to each spike.
        """

        definition = """   
        # Properties of a given unit from a round of clustering (and curation)
        -> master
        unit: int
        ---
        -> probe.ElectrodeConfig.Electrode  # electrode with highest waveform amplitude for this unit
        -> ClusterQualityLabel
        spike_count: int         # how many spikes in this recording for this unit
        spike_times: longblob    # (s) spike times of this unit, relative to the start of the EphysRecording
        spike_sites : longblob   # array of electrode associated with each spike
        spike_depths=null : longblob  # (um) array of depths associated with each spike, relative to the (0, 0) of the probe    
        """

        class UnitPosition(dj.Part):
            definition = """
                # Estimated unit position in the brain
                -> master
                -> CF.CFAnnotationType
                ---
                -> LAB.Hemisphere
                -> LAB.BrainArea
                -> LAB.SkullReference
                unit_ml_location= null: decimal(8,3)    # um from ref; right is positive; based on manipulator coordinates (or histology) & probe config
                unit_ap_location= null: decimal(8,3)    # um from ref; anterior is positive; based on manipulator coordinates (or histology) & probe config
                unit_dv_location= null: decimal(8,3)    # um from dura; ventral is positive; based on manipulator coordinates (or histology) & probe config
                """

        class UnitComment(dj.Manual):
            """Single unit comments.

            Attributes:
                Unit (foreign key): Unit primary key.
                unit_comment (varchar(767)): free text comment for the unit.
            """

            definition = """
            #
            -> master
            unit_comment: varchar(767) 
            ---
            """

        class UnitCellType(dj.Computed):
            """Single unit cell type.

            Attributes:
                Unit (foreign key): Unit primary key.
                CellType (foreign key): name of the cell type.
            """
            
            definition = """
                #
                -> master
                ---
                -> CellType
                """

    def make(self, key):
        """Automated population of Unit information."""
        output_dir = (
            key # TODO: update path? 
        ).fetch1("clustering_output_dir")
        output_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        # Get channel and electrode-site mapping
        electrode_query = (EphysRecording.Channel & key).proj(..., "-channel_name")
        channel2electrode_map: dict[int, dict] = {
            chn.pop("channel_idx"): chn for chn in electrode_query.fetch(as_dict=True)
        }

        # Read from kilosort outputs
        kilosort_dataset = kilosort.Kilosort(output_dir)
        _, sample_rate = (EphysRecording & key).fetch1(
            "acq_software", "sampling_rate"
        )

        sample_rate = kilosort_dataset.data["params"].get(
            "sample_rate", sample_rate
        )

        # ---------- Unit ----------
        # -- Remove 0-spike units
        withspike_idx = [
            i
            for i, u in enumerate(kilosort_dataset.data["cluster_ids"])
            if (kilosort_dataset.data["spike_clusters"] == u).any()
        ]
        valid_units = kilosort_dataset.data["cluster_ids"][withspike_idx]
        valid_unit_labels = kilosort_dataset.data["cluster_groups"][withspike_idx]

        # -- Spike-times --
        # spike_times_sec_adj > spike_times_sec > spike_times
        spike_time_key = (
            "spike_times_sec_adj"
            if "spike_times_sec_adj" in kilosort_dataset.data
            else (
                "spike_times_sec"
                if "spike_times_sec" in kilosort_dataset.data
                else "spike_times"
            )
        )
        spike_times = kilosort_dataset.data[spike_time_key]
        kilosort_dataset.extract_spike_depths()

        # -- Spike-sites and Spike-depths --
        spike_sites = np.array(
            [
                channel2electrode_map[s]["electrode"]
                for s in kilosort_dataset.data["spike_sites"]
            ]
        )
        spike_depths = kilosort_dataset.data["spike_depths"]

        # -- Insert unit, label, peak-chn
        units = []
        for unit, unit_lbl in zip(valid_units, valid_unit_labels):
            if (kilosort_dataset.data["spike_clusters"] == unit).any():
                unit_channel, _ = kilosort_dataset.get_best_channel(unit)
                unit_spike_times = (
                    spike_times[kilosort_dataset.data["spike_clusters"] == unit]
                    / sample_rate
                )
                spike_count = len(unit_spike_times)

                units.append(
                    {
                        **key,
                        "unit": unit,
                        "cluster_quality_label": unit_lbl,
                        **channel2electrode_map[unit_channel],
                        "spike_times": unit_spike_times,
                        "spike_count": spike_count,
                        "spike_sites": spike_sites[
                            kilosort_dataset.data["spike_clusters"] == unit
                        ],
                        "spike_depths": spike_depths[
                            kilosort_dataset.data["spike_clusters"] == unit
                        ],
                    }
                )

        self.insert1(key)
        self.Unit.insert(units, ignore_extra_fields=True)

@schema
class SessionTrial(dj.Manual):
    """Trial information for each session.

    Attributes:
        Session (foreign key): Session primary key.
        trial_number (int): Unique trial number for each session.
        trial_start_time (datetime): Start time of the trial.
        trial_end_time (datetime): End time of the trial.
    """

    definition = """
    # Trial information for each session
    -> Session
    trial_number: int
    ---
    trial_start_time: datetime
    trial_end_time: datetime
    """

@schema
class TrialSpikes(dj.Imported):
    """Single trial spikes.

            Attributes:
                Unit (foreign key): Unit primary key.
                SessionTrial (foreign key): SessionTrial primary key.
                spike_times (longblob): Spike times for each trial, relative to the beginning of the trial.
            """
    
    definition = """
        #
        -> CuratedClustering.Unit
        -> SessionTrial
        ---
        spike_times: longblob   #(s) spike times for each trial, relative to the beginning of the trial" \
        """
    
@schema
class WaveformSet(dj.Imported):
    """A set of spike waveforms for units out of a given CuratedClustering.

    Attributes:
        CuratedClustering (foreign key): CuratedClustering primary key.
    """

    definition = """
    # A set of spike waveforms for units out of a given CuratedClustering
    -> CuratedClustering
    """

    class PeakWaveform(dj.Part):
        """Mean waveform across spikes for a given unit.

        Attributes:
            WaveformSet (foreign key): WaveformSet primary key.
            CuratedClustering.Unit (foreign key): CuratedClustering.Unit primary key.
            peak_electrode_waveform (longblob): Mean waveform for a given unit at its representative electrode.
        """

        definition = """
        # Mean waveform across spikes for a given unit at its representative electrode
        -> master
        -> CuratedClustering.Unit
        ---
        peak_electrode_waveform: longblob  # (uV) mean waveform for a given unit at its representative electrode
        """

    class Waveform(dj.Part):
        """Spike waveforms for a given unit.

        Attributes:
            WaveformSet (foreign key): WaveformSet primary key.
            CuratedClustering.Unit (foreign key): CuratedClustering.Unit primary key.
            probe.ElectrodeConfig.Electrode (foreign key): probe.ElectrodeConfig.Electrode primary key.
            waveform_mean (longblob): mean waveform across spikes of the unit in microvolts.
            waveforms (longblob): waveforms of a sampling of spikes at the given electrode and unit.
        """

        definition = """
        # Spike waveforms and their mean across spikes for the given unit
        -> master
        -> CuratedClustering.Unit
        -> probe.ElectrodeConfig.Electrode  
        --- 
        waveform_mean: longblob   # (uV) mean waveform across spikes of the given unit
        waveforms=null: longblob  # (uV) (spike x sample) waveforms of a sampling of spikes at the given electrode for the given unit
        """

    def make(self, key):
        """Populates waveform tables."""
        output_dir = (
            key # TODO: update path?
        ).fetch1("clustering_output_dir")
        output_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        self.insert1(key)
        if not len(CuratedClustering.Unit & key):
            logger.info(
                f"No CuratedClustering.Unit found for {key}, skipping Waveform ingestion."
            )
            return

        # Get channel and electrode-site mapping
        electrode_query = (EphysRecording.Channel & key).proj(..., "-channel_name")
        channel2electrode_map: dict[int, dict] = {
            chn.pop("channel_idx"): chn for chn in electrode_query.fetch(as_dict=True)
        }

        
        kilosort_dataset = kilosort.Kilosort(output_dir)

        # acq_software, probe_serial_number = (
        #     EphysRecording * ProbeInsertion & key
        # ).fetch1("acq_software", "probe")

        # Get all units
        units = {
            u["unit"]: u
            for u in (CuratedClustering.Unit & key).fetch(
                as_dict=True, order_by="unit"
            )
        }

        if (output_dir / "mean_waveforms.npy").exists():
            unit_waveforms = np.load(
                output_dir / "mean_waveforms.npy"
            )  # unit x channel x sample

            def yield_unit_waveforms():
                for unit_no, unit_waveform in zip(
                    kilosort_dataset.data["cluster_ids"], unit_waveforms
                ):
                    unit_peak_waveform = {}
                    unit_electrode_waveforms = []
                    if unit_no in units:
                        for channel, channel_waveform in zip(
                            kilosort_dataset.data["channel_map"], unit_waveform
                        ):
                            unit_electrode_waveforms.append(
                                {
                                    **units[unit_no],
                                    **channel2electrode_map[channel],
                                    "waveform_mean": channel_waveform,
                                }
                            )
                            if (
                                channel2electrode_map[channel]["electrode"]
                                == units[unit_no]["electrode"]
                            ):
                                unit_peak_waveform = {
                                    **units[unit_no],
                                    "peak_electrode_waveform": channel_waveform,
                                }
                    yield unit_peak_waveform, unit_electrode_waveforms

        else:
            spikeglx_meta_filepath = get_spikeglx_meta_filepath(key)
            neuropixels_recording = spikeglx.SpikeGLX(
                spikeglx_meta_filepath.parent
                )

            def yield_unit_waveforms():
                for unit_dict in units.values():
                    unit_peak_waveform = {}
                    unit_electrode_waveforms = []

                    spikes = unit_dict["spike_times"]
                    waveforms = neuropixels_recording.extract_spike_waveforms(
                        spikes, kilosort_dataset.data["channel_map"]
                    )  # (sample x channel x spike)
                    waveforms = waveforms.transpose(
                        (1, 2, 0)
                    )  # (channel x spike x sample)
                    for channel, channel_waveform in zip(
                        kilosort_dataset.data["channel_map"], waveforms
                    ):
                        unit_electrode_waveforms.append(
                            {
                                **unit_dict,
                                **channel2electrode_map[channel],
                                "waveform_mean": channel_waveform.mean(axis=0),
                                "waveforms": channel_waveform,
                            }
                        )
                        if (
                            channel2electrode_map[channel]["electrode"]
                            == unit_dict["electrode"]
                        ):
                            unit_peak_waveform = {
                                **unit_dict,
                                "peak_electrode_waveform": channel_waveform.mean(
                                    axis=0
                                ),
                            }

                    yield unit_peak_waveform, unit_electrode_waveforms

        # insert waveform on a per-unit basis to mitigate potential memory issue
        for unit_peak_waveform, unit_electrode_waveforms in yield_unit_waveforms():
            if unit_peak_waveform:
                self.PeakWaveform.insert1(unit_peak_waveform, ignore_extra_fields=True)
            if unit_electrode_waveforms:
                self.Waveform.insert(unit_electrode_waveforms, ignore_extra_fields=True)
                
@schema
class CellType(dj.Lookup):
    """Types of cells.

            Attributes:
                cell_type: cell type name.
                cell_type_description: text description for the unit.
            """
    
    definition = """
        #
        cell_type: varchar(100)
        ---
        cell_type_description: varchar(4000)
        """


# TODO: For non commented tables, decide which & how to include in the schemas above
    
# @schema
# class UnitQualityType(dj.Lookup):
#     definition = """
#         #
#         unit_quality: varchar(100)
#         ---
#         unit_quality_description: varchar(4000)
#         """
#     contents = [
#         ['good', 'single unit'],
#         ['ok', 'probably a single unit, but could be contaminated'],
#         ['multi', 'multi unit'],
#         ['all', 'all units'],
#         ['ok or good', 'include both ok and good unit']
#         ]

# @schema
# class UnitWaveform(dj.Part):
#     definition = """
#         # Estimated unit position in the brain
#         -> EPHYS.Unit
#         ---
#         waveform: blob              # unit average waveform. time in samples, amplitude in microvolts.
#         spk_width_ms: float         # unit average spike width, in ms
#         sampling_fq: float          # Hz
#         waveform_amplitude: float   # unit amplitude (peak) in microvolts
#         """
    
# @schema
# class UnitSpikes(dj.Part):
#     definition = """
#         #
#         -> EPHYS.Unit
#         ---
#         spike_times: longblob   #(s) spike times for the entire session (relative to the beginning of the session) 
#         """
    
    
# @schema
# class ElectrodeGroup(dj.Manual):
#     definition = """
#         # Electrode
#         -> EXP.Session
#         electrode_group : tinyint   # shank number
#         ---
#         -> EPHYS.Probe
#         """

# @schema
# class ElectrodeGroupPosition(dj.Part):
#     definition = """
#         -> EPHYS.ElectrodeGroup
#         -> CF.CFAnnotationType
#         ---
#         -> LAB.SkullReference
#         ml_location= null: decimal(8,3)     # um from ref ; right is positive; based on manipulator coordinates/reconstructed track
#         ap_location= null: decimal(8,3)     # um from ref; anterior is positive; based on manipulator coordinates/reconstructed track
#         dv_location= null: decimal(8,3)     # um from dura; ventral is positive; based on manipulator coordinates/reconstructed track
#         ml_angle = null: decimal(8,3)       # Angle between the mainipulator/reconstructed track and the Medio-Lateral axis. A tilt towards the right hemishpere is positive.
#         ap_angle = null: decimal(8,3)       # Angle between the mainipulator/reconstructed track and the Anterior-Posterior axis. An anterior tilt is positive." \
#         """
    
# @schema
# class Jobs(dj.Jobs):
#     definition = """
#         # the job reservation table for +EPHYS
#         table_name: varchar(255)    # className of the table
#         key_hash: char(32)          # key hash
#         -----
#         status: enum("reserved","error","ignore") # if tuple is missing, the job is available
#         key=null: blob                              # structure containing the key
#         error_message="": varchar(1023)             # error message returned if failed
#         error_stack=null: blob                      # error stack if failed
#         host="": varchar(255)                       # system hostname
#         pid=0: int unsigned                         # system process id
#         timestamp=CURRENT_TIMESTAMP: timestamp      # automatic timestamp
#         """
    
# ---------------- HELPER FUNCTIONS ----------------


def get_spikeglx_meta_filepath(ephys_recording_key: dict) -> str:
    """Get spikeGLX data filepath."""
    # attempt to retrieve from EphysRecording.EphysFile
    spikeglx_meta_filepath = pathlib.Path(
        (
            EphysRecording.EphysFile
            & ephys_recording_key
            & 'file_path LIKE "%.ap.meta"'
        ).fetch1("file_path")
    )

    try:
        spikeglx_meta_filepath = find_full_path(
            get_ephys_root_data_dir(), spikeglx_meta_filepath
        )
    except FileNotFoundError:
        # if not found, search in session_dir again
        if not spikeglx_meta_filepath.exists():
            session_dir = find_full_path(
                get_ephys_root_data_dir(), get_session_directory(ephys_recording_key)
            )
            inserted_probe_serial_number = (
                ProbeInsertion * probe.Probe & ephys_recording_key
            ).fetch1("probe")

            spikeglx_meta_filepaths = [fp for fp in session_dir.rglob("*.ap.meta")]
            for meta_filepath in spikeglx_meta_filepaths:
                spikeglx_meta = spikeglx.SpikeGLXMeta(meta_filepath)
                if str(spikeglx_meta.probe_SN) == inserted_probe_serial_number:
                    spikeglx_meta_filepath = meta_filepath
                    break
            else:
                raise FileNotFoundError(
                    "No SpikeGLX data found for probe insertion: {}".format(
                        ephys_recording_key
                    )
                )

    return spikeglx_meta_filepath

def get_recording_channels_details(ephys_recording_key: dict) -> np.array:
    """Get details of recording channels for a given recording."""
    channels_details = {}

    _, sample_rate = (EphysRecording & ephys_recording_key).fetch1(
        "acq_software", "sampling_rate"
    )

    probe_type = (ProbeInsertion * probe.Probe & ephys_recording_key).fetch1(
        "probe_type"
    )
    channels_details["probe_type"] = {
        "neuropixels 1.0 - 3A": "3A",
        "neuropixels 1.0 - 3B": "NP1",
        "neuropixels UHD": "NP1100",
        "neuropixels 2.0 - SS": "NP21",
        "neuropixels 2.0 - MS": "NP24",
    }[probe_type]

    electrode_config_key = (
        probe.ElectrodeConfig * EphysRecording & ephys_recording_key
    ).fetch1("KEY")
    (
        channels_details["channel_ind"],
        channels_details["x_coords"],
        channels_details["y_coords"],
        channels_details["shank_ind"],
    ) = (
        probe.ElectrodeConfig.Electrode * probe.ProbeType.Electrode
        & electrode_config_key
    ).fetch(
        "electrode", "x_coord", "y_coord", "shank"
    )
    channels_details["sample_rate"] = sample_rate
    channels_details["num_channels"] = len(channels_details["channel_ind"])

    spikeglx_meta_filepath = get_spikeglx_meta_filepath(ephys_recording_key)
    spikeglx_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)
    channels_details["uVPerBit"] = spikeglx_recording.get_channel_bit_volts("ap")[0]
    channels_details["connected"] = np.array(
        [v for *_, v in spikeglx_recording.apmeta.shankmap["data"]]
    )

    return channels_details


def generate_electrode_config_entry(probe_type: str, electrode_keys: list) -> dict:
    """Generate and insert new ElectrodeConfig

    Args:
        probe_type (str): probe type (e.g. neuropixels 2.0 - SS)
        electrode_keys (list): list of keys of the probe.ProbeType.Electrode table

    Returns:
        dict: representing a key of the probe.ElectrodeConfig table
    """
    # compute hash for the electrode config (hash of dict of all ElectrodeConfig.Electrode)
    electrode_config_hash = dict_to_uuid({k["electrode"]: k for k in electrode_keys})

    electrode_list = sorted([k["electrode"] for k in electrode_keys])
    electrode_gaps = (
        [-1]
        + np.where(np.diff(electrode_list) > 1)[0].tolist()
        + [len(electrode_list) - 1]
    )
    electrode_config_name = "; ".join(
        [
            f"{electrode_list[start + 1]}-{electrode_list[end]}"
            for start, end in zip(electrode_gaps[:-1], electrode_gaps[1:])
        ]
    )
    electrode_config_key = {"electrode_config_hash": electrode_config_hash}
    econfig_entry = {
        **electrode_config_key,
        "probe_type": probe_type,
        "electrode_config_name": electrode_config_name,
    }
    econfig_electrodes = [
        {**electrode, **electrode_config_key} for electrode in electrode_keys
    ]

    return econfig_entry, econfig_electrodes


