import datajoint as dj
import dj_connect
import getSchema

conn = dj_connect.connectToDataJoint("almog", "simple")
schema = getSchema.getSchema()

@schema # The `@schema` decorator for DataJoint classes creates the table on the server.
class UnitSpikes(dj.Part):
    definition = """
        #
        -> EPHYS.Unit
        ---
        spike_times: longblob   #(s) spike times for the entire session (relative to the beginning of the session) 
        """