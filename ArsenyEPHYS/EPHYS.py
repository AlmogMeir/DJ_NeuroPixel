import datajoint as dj
import numpy as np
import pandas as pd

import dj_connect
import getSchema

conn = dj_connect.connectToDataJoint("almog", "simple")
schema = getSchema.getSchema()


# ----------------------------- Table declarations ----------------------

@schema
class Probe(dj.Lookup):
    definition = """
        #
        probe_part_no   :  varchar(20)
        ---
        probe_type      :  varchar(32)
        probe_comment   :  varchar(4000)      
        """
    
@schema
class Unit(dj.Imported):
    definition = """
        -> EPHYS.ElectrodeGroup
        unit: smallint
        ---
        unit_uid: int               # unique across sessions/animals
        -> EPHYS.UnitQualityType
        unit_channel = null: float  # channel on the electrode for which the unit has the largest amplitude
        """

@schema
class UnitCellType(dj.Computed):
    definition = """
        #
        -> EPHYS.Unit
        ---
        -> EPHYS.CellType
        """
    
@schema
class UnitPosition(dj.Part):
    definition = """
        # Estimated unit position in the brain
        -> EPHYS.Unit
        -> CF.CFAnnotationType
        ---
        -> LAB.Hemisphere
        -> LAB.BrainArea
        -> LAB.SkullReference
        unit_ml_location= null: decimal(8,3)    # um from ref; right is positive; based on manipulator coordinates (or histology) & probe config
        unit_ap_location= null: decimal(8,3)    # um from ref; anterior is positive; based on manipulator coordinates (or histology) & probe config
        unit_dv_location= null: decimal(8,3)    # um from dura; ventral is positive; based on manipulator coordinates (or histology) & probe config
        """
    
@schema
class UnitComment(dj.Manual):
    definition = """
        #
        -> EPHYS.Unit
        unit_comment :  varchar(767) 
        ---
        """
    
@schema
class UnitQualityType(dj.Lookup):
    definition = """
        #
        unit_quality: varchar(100)
        ---
        unit_quality_description: varchar(4000)
        """
    contents = [
        ['good', 'single unit'],
        ['ok', 'probably a single unit, but could be contaminated'],
        ['multi', 'multi unit'],
        ['all', 'all units'],
        ['ok or good', 'include both ok and good unit']
        ]

@schema
class UnitWaveform(dj.Part):
    definition = """
        # Estimated unit position in the brain
        -> EPHYS.Unit
        ---
        waveform: blob              # unit average waveform. time in samples, amplitude in microvolts.
        spk_width_ms: float         # unit average spike width, in ms
        sampling_fq: float          # Hz
        waveform_amplitude: float   # unit amplitude (peak) in microvolts
        """
    
@schema
class UnitSpikes(dj.Part):
    definition = """
        #
        -> EPHYS.Unit
        ---
        spike_times: longblob   #(s) spike times for the entire session (relative to the beginning of the session) 
        """
    
@schema
class TrialSpikes(dj.Imported):
    definition = """
        #
        -> EPHYS.Unit
        -> EXP.SessionTrial
        ---
        spike_times: longblob   #(s) spike times for each trial, relative to the beginning of the trial" \
        """
    
@schema
class ElectrodeGroup(dj.Manual):
    definition = """
        # Electrode
        -> EXP.Session
        electrode_group : tinyint   # shank number
        ---
        -> EPHYS.Probe
        """

@schema
class ElectrodeGroupPosition(dj.Part):
    definition = """
        -> EPHYS.ElectrodeGroup
        -> CF.CFAnnotationType
        ---
        -> LAB.SkullReference
        ml_location= null: decimal(8,3)     # um from ref ; right is positive; based on manipulator coordinates/reconstructed track
        ap_location= null: decimal(8,3)     # um from ref; anterior is positive; based on manipulator coordinates/reconstructed track
        dv_location= null: decimal(8,3)     # um from dura; ventral is positive; based on manipulator coordinates/reconstructed track
        ml_angle = null: decimal(8,3)       # Angle between the mainipulator/reconstructed track and the Medio-Lateral axis. A tilt towards the right hemishpere is positive.
        ap_angle = null: decimal(8,3)       # Angle between the mainipulator/reconstructed track and the Anterior-Posterior axis. An anterior tilt is positive." \
        """
    
@schema
class LabeledTrack(dj.Manual):
    definition = """
        #
        -> EPHYS.ElectrodeGroup
        ---
        labeling_date: date         # in case we labeled the track not during a recorded session we can specify the exact date here
        dye_color  : varchar(32)
        """
    
@schema
class CellType(dj.Lookup):
    definition = """
        #
        cell_type: varchar(100)
        ---
        cell_type_description: varchar(4000)
        """

@schema
class Jobs(dj.Jobs):
    definition = """
        # the job reservation table for +EPHYS
        table_name: varchar(255)    # className of the table
        key_hash: char(32)          # key hash
        -----
        status: enum("reserved","error","ignore") # if tuple is missing, the job is available
        key=null: blob                              # structure containing the key
        error_message="": varchar(1023)             # error message returned if failed
        error_stack=null: blob                      # error stack if failed
        host="": varchar(255)                       # system hostname
        pid=0: int unsigned                         # system process id
        timestamp=CURRENT_TIMESTAMP: timestamp      # automatic timestamp
        """