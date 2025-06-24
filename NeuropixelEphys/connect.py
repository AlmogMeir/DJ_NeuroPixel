# import datajoint as dj
# import dj_connect

# def connect():
#     """Connect to DataJoint database."""
#     # Check if the connection is already established
#     if dj.config.get("database.host") is None:
#         raise ValueError("Database host is not set in DataJoint configuration.")
#     if dj.config.get("database.user") is None:
#         raise ValueError("Database user is not set in DataJoint configuration.")
#     # Connect to the DataJoint database
#     # This assumes that the connection details are set in the DataJoint config

    # conn = dj_connect.connectToDataJoint("almog", "simple")
#     return conn

import datajoint as dj
import dj_connect

def connect():
    """Connect to DataJoint database using provided credentials if not already connected."""
    # If a connection exists and the user matches, return it.
    existing_conn = dj.config if dj.config is not None else None
    required_user = "talch012"
    
    if existing_conn and existing_conn.user == required_user:
        return existing_conn

    # Check if the necessary configurations are set.
    if dj.config.get("database.host") is None:
        raise ValueError("Database host is not set in DataJoint configuration.")
    if dj.config.get("database.user") is None:
        raise ValueError("Database user is not set in DataJoint configuration.")

    # Otherwise, create a new connection with the provided credentials.
    conn = dj_connect.connectToDataJoint("talch012", "simple")
    return conn
