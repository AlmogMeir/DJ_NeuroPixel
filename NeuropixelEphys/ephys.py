import datajoint as dj
import numpy as np
import pandas as pd
# import NeuropixelEphys.dj_connect as dj_connect
# import NeuropixelEphys.getSchema as getSchema
import dj_connect
import getSchema
# from scipy.io import loadmat

# schema = getSchema.getSchema()

conn = dj_connect.connectToDataJoint("shany", "shany1906")

schema = dj.Schema("shany_EPHYS")
exp = dj.VirtualModule("EXP", "shany_exp2", create_tables=True)
lab = dj.VirtualModule("LAB", "shany_lab", create_tables=True)

# ----------------------------- Table declarations ----------------------

@schema
class Probe(dj.Lookup):
    """Probe table.
    Attributes:
        probe_part_no (varchar): Probe part number.
        probe_type (varchar): Probe type.
        probe_comment (varchar): Probe comment.
    """

    definition = """
        #
        probe_part_no   :  varchar(20)
        ---
        probe_type      :  varchar(32)
        probe_comment   :  varchar(4000)      
        """
    
    contents = [{'probe_part_no': '', 'probe_type': 'NeuroPixel2.0', 'probe_comment': ''}]
    # TODO: Change to the actual probe types

@schema
class ProbeMapping(dj.Imported):
    """Probe mapping table.
    Attributes:
        Probe (foreign key): Probe primary key.
        probe_mapping (longblob): Probe mapping file.
    """

    definition = """
        -> Probe
        ---
        probe_mapping: longblob
        """


@schema
class ElectrodeGroup(dj.Manual):
    """Electrode group table.
    Attributes:
        exp.SessionEpoch (foreign key): SessionEpoch primary key.
        electrode_group (tinyint): Shank number.
        -> Probe (foreign key): Probe primary key.
    """

    definition = """
        # Electrode
        -> exp.SessionEpoch
        electrode_group : tinyint   # shank number
        -> Probe
        ---
        """

@schema
class ElectrodeGroupPosition(dj.Imported):
    """Electrode group position in the brain.
    Attributes:
        ElectrodeGroup (foreign key): ElectrodeGroup primary key.
        ml_location (float): Electrode medio-lateral location in micrometers.
        ap_location (float): Electrode anterior-posterior location in micrometers.
        dv_location (float): Electrode dorsal-ventral location in micrometers.
        ml_angle (float): Angle between the mainipulator/reconstructed track and the Medio-Lateral axis. A tilt towards the right hemishpere is positive.
        ap_angle (float): Angle between the mainipulator/reconstructed track and the Anterior-Posterior axis. An anterior tilt is positive.
    """

    definition = """
        -> ElectrodeGroup
        ---
        ml_location= null: decimal(8,3)     # um from ref ; right is positive; based on manipulator coordinates/reconstructed track
        ap_location= null: decimal(8,3)     # um from ref; anterior is positive; based on manipulator coordinates/reconstructed track
        dv_location= null: decimal(8,3)     # um from dura; ventral is positive; based on manipulator coordinates/reconstructed track
        ml_angle = null: decimal(8,3)       # Angle between the mainipulator/reconstructed track and the Medio-Lateral axis. A tilt towards the right hemishpere is positive.
        ap_angle = null: decimal(8,3)       # Angle between the mainipulator/reconstructed track and the Anterior-Posterior axis. An anterior tilt is positive." \
        """
    
def make(self, key):
    """Populates table with Electrode information."""

@schema
class CellType(dj.Lookup):
    """Cell type table.
    Attributes:
        cell_type (varchar): Cell type name.
        cell_type_description (varchar): Cell type description.
    """

    definition = """
        #
        cell_type: varchar(100)
        ---
        cell_type_description: varchar(4000)
        """
    
    contents = [('Pyr','putative pyramidal'),
            ('FS','fast spiking'),
            ('not classified','intermediate spike-width that falls between spike-width thresholds for FS or Putative pyramidal cells'),
            ('all','all types')]
    
@schema
class UnitQualityType(dj.Lookup):
        """Unit quality type.
        Attributes:
            unit_quality (varchar): Unit quality type name.
            unit_quality_description (varchar): Unit quality type description.
        """
        
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
class Unit(dj.Imported):
    # ElectrodeGroup is computed location + id (shank number)
    """Single unit table.
    Attributes:
        ElectrodeGroup (foreign key): Electrode group primary key.
        unit (smallint): Unit number.
        unit_uid (int): Unique unit identifier across sessions/animals. ###
        UnitQualityType (foreign key): Unit quality type.
        unit_channel (float): Channel on the electrode for which the unit has the largest amplitude.
        roi_number (smallint): unit number, for compatibility with imaging tables.
    """

    definition = """
        -> ElectrodeGroup
        unit: smallint
        ---
        unit_uid: int               # unique across sessions/animals
        -> UnitQualityType
        unit_channel = null: float  # channel on the electrode for which the unit has the largest amplitude
        roi_number: smallint
        """

@schema
class UnitCellType(dj.Manual):
    """Single unit cell type.

    Attributes:
        Unit (foreign key): Unit primary key.
        CellType (foreign key): name of the cell type.
    """

    definition = """
        #
        -> Unit
        ---
        -> CellType
        """

@schema
class UnitPosition(dj.Manual):
    """Estimated unit position in the brain.
    Attributes:
        Unit (foreign key): Unit primary key.
        lab.Hemisphere (foreign key): Hemisphere primary key.
        lab.BrainArea (foreign key): Brain area primary key.
        lab.SkullReference (foreign key): Skull reference primary key.
        unit_ml_location (float): Unit medio-lateral location in micrometers.
        unit_ap_location (float): Unit anterior-posterior location in micrometers.
        unit_dv_location (float): Unit dorsal-ventral location in micrometers.
        
    """

    definition = """
        # Estimated unit position in the brain
        -> Unit
        ---
        -> lab.Hemisphere
        -> lab.BrainArea
        -> lab.SkullReference
        unit_ml_location= null: decimal(8,3)    # um from ref; right is positive; based on manipulator coordinates (or histology) & probe config
        unit_ap_location= null: decimal(8,3)    # um from ref; anterior is positive; based on manipulator coordinates (or histology) & probe config
        unit_dv_location= null: decimal(8,3)    # um from dura; ventral is positive; based on manipulator coordinates (or histology) & probe config
        """
    
    #TODO: Add absolute position

@schema
class UnitComment(dj.Manual):
    """Single unit comments.

    Attributes:
        Unit (foreign key): Unit primary key.
        unit_comment (varchar(767)): free text comment for the unit.
    """

    definition = """
        #
        -> Unit
        unit_comment: varchar(767) 
        ---
        """

# Take from original files of kilosort/bombcell
@schema
class UnitWaveform(dj.Manual):
    """Mean waveform across spikes for a given unit.

    Attributes:
        Unit (foreign key): Unit primary key.
        waveform (blob): Unit average waveform. time in samples, amplitude in microvolts.
        spk_width_ms (float): Unit average spike width, in ms.
        sampling_fq (float): Sampling frequency in Hz.
        waveform_amplitude (float): Unit amplitude (peak) in microvolts.
    """

    definition = """
        # Estimated unit position in the brain
        -> Unit
        ---
        waveform: blob              # unit average waveform. time in samples, amplitude in microvolts.
        spk_width_ms: float         # unit average spike width, in ms
        sampling_fq: float          # Hz
        waveform_amplitude: float   # unit amplitude (peak) in microvolts
        """

# Take from original files of kilosort or merge Ben's
@schema
class UnitSpikes(dj.Manual):
    """Spikes for each unit.
    Attributes:
        Unit (foreign key): Unit primary key.
        spike_times (longblob): Spike times for the entire session (relative to the beginning of the session).
    """

    definition = """
    #
    -> Unit
    ---
    spike_times: longblob   #(s) spike times for the entire session (relative to the beginning of the session) 
    """
    
def make(self, key):
    """Automated population of Unit information."""

# Example from the matlab code
# TODO: Implement the makeTuples function
"""
methods(Access=protected)
    function makeTuples(self, key)

        obj = EXP.getObj(key);
        counter=0;
        for iUnits = 1:size(obj.eventSeriesHash.value,2)
            unit_channel = mode(obj.eventSeriesHash.value{iUnits}.channel);

            if unit_channel<=32 && key.electrode_group ==1
                Insert_Unit(self, key, iUnits, unit_channel);
                counter=counter+1;
            elseif unit_channel>32 && key.electrode_group ==2
                unit_channel = unit_channel-32;
                Insert_Unit(self, key, iUnits, unit_channel);
                counter=counter+1;
            else
            end
            
        end
        fprintf('Populated %d units recorded from animal %d  on %s', counter, key.subject_id, fetch1(EXP.SessionEpoch & key,'session_date'))
    end
end
"""

@schema
class TrialSpikes(dj.Manual):
    definition = """
        # Spikes for each trial
        -> Unit (foreign key): Unit primary key.
        -> exp.SessionTrial (foreign key): Session trial primary key.
        ---
        spike_times: longblob   #(s) spike times for each trial, relative to the beginning of the trial" \
        """
    definition = """
        #
        -> Unit
        -> exp.SessionTrial
        ---
        spike_times: longblob   #(s) spike times for each trial, relative to the beginning of the trial" \
        """
    def make(self, key):
        """Automated population of TrialSpikes information."""
        # # Get the session key
        # session_key = key.copy()
        # session_key.pop('unit')
        
        # # Fetch the trial spikes for the given session
        # trial_spikes = (EXP.SessionTrial & session_key).fetch('spike_times')
        
        # # Store the trial spikes in the Unit table
        # self.insert1({**key, 'spike_times': trial_spikes})
