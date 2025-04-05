import os
import datajoint as dj

def connectToDataJoint(username, password):
    """
    Connect to DataJoint database.
    """
    # Set the DataJoint configuration
    dj.config['database.host'] = "arseny-lab.cmte3q4ziyvy.il-central-1.rds.amazonaws.com"
    dj.config['database.user'] = username
    dj.config['database.password'] = password

    # Connect to the database    
    return dj.conn()