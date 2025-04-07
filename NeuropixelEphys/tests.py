import numpy as np
import pandas as pd
import datajoint as dj
import os
import re
import dj_connect
import getSchema

conn = dj_connect.connectToDataJoint("almog", "simple")
schema = getSchema.getSchema()

print(dj.list_schemas())
print(schema.list_tables())
schema = dj.VirtualModule('EPHYS', conn)
dj.Diagram(schema).draw()