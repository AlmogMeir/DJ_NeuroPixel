import numpy as np
import pandas as pd
import datajoint as dj
import cv2
import os
import re
import dj_connect
import getSchema

conn = dj_connect.connectToDataJoint()
schema = getSchema.getSchema()

@schema # The `@schema` decorator for DataJoint classes creates the table on the server.
class ElectrodeGroup(dj.Manual):
    definition = """
        # Electrode
        -> EXP.Session
        electrode_group : tinyint   # shank number
        ---
        -> EPHYS.Probe
        """