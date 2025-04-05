import datajoint as dj
import dj_connect
import getSchema

conn = dj_connect.connectToDataJoint("almog", "simple")
schema = getSchema.getSchema()

@schema # The `@schema` decorator for DataJoint classes creates the table on the server.
class CellType(dj.Lookup):
    definition = """
        #
        cell_type: varchar(100)
        ---
        cell_type_description: varchar(4000)
        """