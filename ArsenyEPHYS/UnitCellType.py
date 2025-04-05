import datajoint as dj
import dj_connect
import getSchema

conn = dj_connect.connectToDataJoint("almog", "simple")
schema = getSchema.getSchema()

@schema # The `@schema` decorator for DataJoint classes creates the table on the server.
class UnitCellType(dj.Computed):
    definition = """
        #
        -> EPHYS.Unit
        ---
        -> EPHYS.CellType
        """