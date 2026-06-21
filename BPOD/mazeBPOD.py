import datajoint as dj
import numpy as np
import pandas as pd
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from NeuropixelEphys import dj_connect
from NeuropixelEphys import getSchema

# Connect to DataJoint
conn = dj_connect.connectToDataJoint("talch012", "simple")

schema = dj.Schema("talch012_mazeBPOD")
exp = dj.VirtualModule("EXP", "talch012_expt", create_tables=True)

# ----------------------------- Table declarations ----------------------

@schema  
class Port(dj.Manual):
    """Port information for the session.
    
    Attributes:
        exp.Session (foreign key): Session primary key.
        port_number (tinyint): Port number identifier.
        port_reward_size (float): Reward size for the port in seconds.
    """
    
    definition = """
        # Port information for the session
        -> exp.Session
        port_number: tinyint            # port number (1, 2, 3, etc.)
        ---
        port_reward_size: float         # (s) reward size for the port
        """
    
@schema  
class Block(dj.Manual):
    """Block information for the session.
    
    Attributes:
        exp.Session (foreign key): Session primary key.
        block_id (tinyint): Block identifier within the session.
        -> Port: Foreign key to Port table.
    """
    
    definition = """
        # Block information for the session
        -> exp.Session
        block_id: tinyint               # block ID within session
        -> Port
        ---
        """

@schema
class SessionTrial(dj.Manual):
    """Trial information for the session.
    
    Attributes:
        exp.Session (foreign key): Session primary key.
        trial (smallint): Trial number within the session.
        start_time (float): Start time of the trial in seconds.
    """
    
    definition = """
        # Trial information for the session
        -> exp.Session
        trial: smallint                 # trial number within session
        ---
        start_time = null: float        # (s) start time of the trial
        """

@schema  
class Reward(dj.Manual):
    """Reward info in the trial.
    
    Attributes:
        -> SessionTrial: Trial behavior primary key.
        -> Port
        reward_size (float): Size of the reward in seconds.
        abs_time (float): Absolute time in BPOD where reward was given.
        depletion_step (tinyint): Depletion step number for the port. 0 for no depletion.
        depletion_size (float): Depletion size in seconds for the port. port_reward_size / reward_size.

    """
    
    definition = """
        # Individual reward events
        -> SessionTrial
        ---
        -> Port
        reward_size: float              # (s) size of the reward
        abs_time: float                 # (s) absolute time in BPOD where reward was given
        depletion_step: tinyint         # depletion step number for the port. 0 for no depletion.
        depletion_size: float           # depletion size in seconds for the port. port_reward_size / reward_size.  
 
        """



@schema  
class LickEvent(dj.Manual):
    """Individual lick events for detailed analysis.
    
    Attributes:
        -> SessionTrial: Trial behavior primary key.
        lick_id (smallint): Lick event identifier within the trial.
        -> Port
        lick_time (float): Time of lick start relative to trial start.
        lick_duration (float): Duration of lick (PortOut time - PortIn time).
        lick_start_absolute (float): Absolute start time of lick.
        lick_end_absolute (float): Absolute end time of lick.
        is_early (boolean): Is this an early lick, based on IRI time (before end of IRI is early)
    """
    
    definition = """
        # Individual lick events
        -> SessionTrial
        lick_id: smallint               # lick event ID within trial
        ---
        -> Port
        lick_time: float                # (s) time relative to trial start (start of lick)
        lick_duration: float            # (s) duration of lick (out_time - in_time)
        lick_start_absolute = null: float  # (s) absolute start time of lick
        lick_end_absolute = null: float    # (s) absolute end time of lick
        is_early: boolean                  # is before IRI end or after
        """

# @schema
# class TrialBehavior(dj.Manual):
#     """Behavioral data for each trial.
    
#     Attributes:
#         exp.Session (foreign key): Session trial primary key.
#         trial_start_time (float): Trial start time in seconds.
#         trial_end_time (float): Trial end time in seconds.
#         reward_size (float): Reward size calculated as end_time - start_time.
#         licks (longblob): Array of lick events, each containing port number and time.
#         blocks (longblob): Tuple representing which ports are currently open.
#     """
    
#     definition = """
#         # Behavioral data per trial
#         -> exp.Session
#         trial: smallint                 # trial number within session (primary key component)
#         ---
#         trial_start_time: float         # (s) trial start time relative to session start
#         trial_end_time: float           # (s) trial end time relative to session start
#         reward_size: float              # (s) reward size = end_time - start_time
#         licks: longblob                 # array of lick events [port: int, time: float, ...]
#         blocks: blob                # tuple of currently open ports (port1, port2, ...)
#         """
    
#     def make(self, key):
#         """Automated population of TrialBehavior information."""
#         # TODO: Implement population logic based on BPOD data files
#         pass