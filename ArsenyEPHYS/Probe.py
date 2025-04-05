import datajoint as dj
import dj_connect
import getSchema

conn = dj_connect.connectToDataJoint("almog", "simple")
schema = getSchema.getSchema()

@schema # The `@schema` decorator for DataJoint classes creates the table on the server.
class Probe(dj.Lookup):
    definition = """
        #
        probe_part_no   :  varchar(20)
        ---
        probe_type      :  varchar(32)
        probe_comment   :  varchar(4000)      
        """