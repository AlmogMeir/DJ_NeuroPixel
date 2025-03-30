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