import numpy as np
import pandas as pd
import datajoint as dj
import os
import re
import dj_connect
import getSchema

# conn = dj_connect.connectToDataJoint("almog", "simple")
# schema = getSchema.getSchema()
# arseny_schema = dj.VirtualModule("arseny_schema", "arseny_s1alm_experiment")

# schema = dj.Schema("almog_EPHYS")
# exp = dj.VirtualModule("exp", "arseny_s1alm_experiment")
# # schema = dj.Schema("arseny_s1alm_experiment")
# # schema.spawn_missing_classes()
# print(schema.list_tables())
# schema = dj.VirtualModule("ephys", "almog_EPHYS")
# schema.ProbeInsertion.drop()
# Create a new schema for the copy
# almog_schema = dj.Schema('almog_EPHYS')

# Copy tables from arseny_schema to almog_schema
# for table_name in arseny_schema.list_tables():
#     table = getattr(arseny_schema, table_name)
#     class Copy(table.__class__):
#         definition = table.definition
#         _schema = almog_schema

# print(arseny_schema)
# print(almog_schema.list_tables())

# show the virtual module
# schema = dj.VirtualModule("arseny", "arseny_s1alm_experiment")
# query = schema.Session
# data = query.fetch()
# print(query)
# schema = dj.VirtualModule('EPHYS', conn)
# dj.Diagram(schema).draw()

from scipy.io import loadmat

mat_data = loadmat('your_matlab_file.mat')
print(mat_data.keys())
