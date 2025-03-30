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
class LabeledTrack(dj.Manual):
    definition = """
        #
        -> EPHYS.ElectrodeGroup
        ---
        labeling_date: date         # in case we labeled the track not during a recorded session we can specify the exact date here
        dye_color  : varchar(32)
        """