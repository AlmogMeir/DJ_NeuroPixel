import datajoint as dj
import dj_connect
import getSchema

conn = dj_connect.connectToDataJoint("almog", "simple")
schema = getSchema.getSchema()

@schema # The `@schema` decorator for DataJoint classes creates the table on the server.
class Unit(dj.Imported):
    definition = """
        -> EPHYS.ElectrodeGroup
        unit: smallint
        ---
        unit_uid: int               # unique across sessions/animals
        -> EPHYS.UnitQualityType
        unit_channel = null: float  # channel on the electrode for which the unit has the largest amplitude
        """