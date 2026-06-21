import numpy as np
import pandas as pd
import datajoint as dj
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from NeuropixelEphys import dj_connect
from NeuropixelEphys import getSchema

conn = dj_connect.connectToDataJoint("talch012", "simple")

schema = dj.Schema("talch012_mazeBPOD")
schema_module = dj.VirtualModule("schema_module", "talch012_mazeBPOD", create_tables=True)

print("Schema:", schema)
print("Tables in the schema:", schema.list_tables())

# session = 1
# subject_id = 101104

schema_module.Port.drop()
schema_module.Block.drop()
schema_module.Reward.drop()
schema_module.LickEvent.drop()



