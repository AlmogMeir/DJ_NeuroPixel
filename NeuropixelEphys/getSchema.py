import datajoint as dj
import dj_connect

# Connect to DataJoint database

def getSchema(schema_name):
    return dj.Schema(f"{dj.config['database.user']}_{schema_name}") 
