import numpy as np
import pandas as pd
import datajoint as dj
# import NeuropixelEphys.dj_connect as dj_connect
# import NeuropixelEphys.getSchema as getSchema
import dj_connect
import getSchema
import mat73

conn = dj_connect.connectToDataJoint("shany", "shany1906")

schema = getSchema.getSchema("EPHYS_TEST")
schema_module = dj.VirtualModule("schema_module", "shany_EPHYS_TEST", create_tables=True)

print("Schema:", schema)
print("Tables in the schema:", schema.list_tables())

session = 1
subject_id = 103008

# schema_module.Probe.drop()
# schema_module.ElectrodeGroup.drop()
# schema_module.Unit.drop()
# schema_module.WaveformSet.drop()
# schema_module.TrialSpikes.drop()
# schema_module.CellType.drop()
schema_module.UnitQualityType.drop()

