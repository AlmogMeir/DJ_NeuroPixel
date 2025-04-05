import numpy as np
import pandas as pd
import datajoint as dj
import os
import re
import dj_connect
import getSchema

conn = dj_connect.connectToDataJoint("almog", "simple")
schema = getSchema.getSchema()

@schema # The `@schema` decorator for DataJoint classes creates the table on the server.
class TrialSpikes(dj.Imported):
    definition = """
        #
        -> EPHYS.Unit
        -> EXP.SessionTrial
        ---
        spike_times: longblob   #(s) spike times for each trial, relative to the beginning of the trial" \
        """