import datajoint as dj
import dj_connect

# Connect to DataJoint database

def getSchema():
    return dj.Schema(f"{dj.config['database.user']}_EXP") 
